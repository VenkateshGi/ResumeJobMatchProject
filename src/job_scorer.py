import os
import sys
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from resume_parser import ResumeProfile
from job_searcher import JobPosting

load_dotenv()

#pydantic schema for Confidence Scoring

class JobMatchScore(BaseModel):
    overall_score: int = Field(
        description="Overall match score from 0 to 100"
    )
    skill_match_score: int = Field(
        description="How well candidate skills match job requirements, 0-100"
    )
    experience_match_score: int = Field(
        description="How well experience level matches, 0-100"
    )
    # domain_match_score: int = Field(
    #     description="How well domain/industry background matches, 0-100"
    # )
    matching_skills: List[str] = Field(
        description="Skills the candidate has that the job requires"
    )
    missing_skills: List[str] = Field(
        description="Skills the job requires that the candidate lacks"
    )
    strengths: str = Field(
        description="1-2 lines on why candidate is a good fit"
    )
    gaps: str = Field(
        description="1-2 lines on key gaps or concerns"
    )
    recommendation: str = Field(
        description="One of: 'Strong Apply', 'Apply', 'Consider', 'Skip'"
    )
    recommendation_reason: str = Field(
        description="One line reason for the recommendation"
    )


# ── Scored job — combines posting + score ────────────────────────────────────

class ScoredJob(BaseModel):
    job: JobPosting
    score: JobMatchScore

    def display(self, rank: int):
        bar_filled = "█" * (self.score.overall_score // 10)
        bar_empty = "░" * (10 - self.score.overall_score // 10)
        bar = f"{bar_filled}{bar_empty} {self.score.overall_score}/100"

        rec_emoji = {
            "Strong Apply": "🟢",
            "Apply": "🟡",
            "Consider": "🟠",
            "Skip": "🔴"
        }.get(self.score.recommendation, "⚪")

        print(f"""
┌─ [{rank}] {self.job.title} @ {self.job.company}
│  📍 {self.job.location} | ⏳ {self.job.experience_required} | 🌐 {self.job.source}
│  Match: {bar}
│  {rec_emoji} {self.score.recommendation} — {self.score.recommendation_reason}
│
│  Skill Match    : {self.score.skill_match_score}/100
│  Experience     : {self.score.experience_match_score}/100
│
│  ✅ Matching    : {', '.join(self.score.matching_skills[:6])}
│  ❌ Missing     : {', '.join(self.score.missing_skills[:5]) or 'None'}
│
│  💪 Strengths   : {self.score.strengths}
│  ⚠️  Gaps        : {self.score.gaps}
│  🔗 {self.job.apply_url}
└{'─' * 65}""")
        

def score_job_match(profile: ResumeProfile, job: JobPosting) -> JobMatchScore:
    """Use LLM to score how well a candidate profile matches a job posting."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    structured_llm = llm.with_structured_output(JobMatchScore)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical recruiter and career coach.
Evaluate how well a candidate's resume profile matches a job posting.
Be realistic and specific — not every candidate is a perfect match.
Score strictly: 90+ only for near-perfect matches, 70-89 for strong fits,
50-69 for partial fits, below 50 for weak matches."""),

        ("human", """
## Candidate Profile
Name: {name}
Current Role: {current_role}
Total Experience: {experience} years
Education: {education}
Skills: {skills}
Tools & Frameworks: {tools}
Summary: {summary}

## Job Posting
Title: {job_title}
Company: {company}
Location: {location}
Experience Required: {exp_required}
Skills Required: {job_skills}
Job Summary: {job_summary}
Source: {source}

Evaluate the match and provide detailed scoring.
""")
    ])
    chain = prompt | structured_llm
    result = chain.invoke({
        "name": profile.name,
        "current_role": profile.current_role,
        "experience": profile.total_experience_years,
        "education": profile.education,
        "skills": ", ".join(profile.skills),
        "tools": ", ".join(profile.tools_and_frameworks),
        "summary": profile.summary,
        "job_title": job.title,
        "company": job.company,
        "location": job.location,
        "exp_required": job.experience_required,
        "job_skills": ", ".join(job.skills_required),
        "job_summary": job.job_summary,
        "source": job.source,
    })

    return result

def score_all_jobs(
    profile: ResumeProfile,
    jobs: List[JobPosting]
) -> List[ScoredJob]:
    """Score all jobs and return sorted by overall_score descending."""

    print(f"\n🎯 Scoring {len(jobs)} jobs against your profile...\n")

    scored_jobs = []

    for i, job in enumerate(jobs, 1):
        print(f"  [{i}/{len(jobs)}] Scoring: {job.title} @ {job.company}...")
        try:
            score = score_job_match(profile, job)
            scored_jobs.append(ScoredJob(job=job, score=score))
            print(f"  ✅ Score: {score.overall_score}/100 | {score.recommendation}")
        except Exception as e:
            print(f"  ⚠️  Scoring failed: {e}")

    # Sort best matches first
    scored_jobs.sort(key=lambda x: x.score.overall_score, reverse=True)

    return scored_jobs

# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from resume_parser import extract_resume_profile
    from job_searcher import search_jobs_for_profile

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/GarnepudiVenkateshResume.pdf"

    print("📄 Step 1: Parsing resume...")
    profile = extract_resume_profile(pdf_path)
    print(f"✅ {profile.name} | {profile.current_role} | {profile.total_experience_years} yrs")

    print("\n🌐 Step 2: Searching jobs...")
    jobs = search_jobs_for_profile(profile, max_results_per_query=3)

    print("\n🎯 Step 3: Scoring matches...")
    scored_jobs = score_all_jobs(profile, jobs)

    print("\n" + "=" * 65)
    print(f"  🏆 TOP JOB MATCHES FOR {profile.name.upper()}")
    print("=" * 65)

    for rank, scored in enumerate(scored_jobs, 1):
        scored.display(rank)

    # Quick summary
    strong = [s for s in scored_jobs if s.score.recommendation in ["Strong Apply", "Apply"]]
    print(f"\n📊 Summary: {len(strong)} jobs worth applying | "
          f"{len(scored_jobs) - len(strong)} to skip\n")
    
