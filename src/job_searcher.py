# src/job_searcher.py

import time
import os
import sys
from typing import List, Optional

from serpapi import GoogleSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from resume_parser import ResumeProfile

load_dotenv()


# ── Pydantic schema for a Job Posting ────────────────────────────────────────

class JobPosting(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name, 'Unknown' if not found")
    location: str = Field(description="Job location")
    experience_required: str = Field(description="Experience required e.g. '3-5 years'")
    skills_required: List[str] = Field(description="Key skills mentioned in the job")
    job_summary: str = Field(description="2-3 line summary of the role")
    apply_url: str = Field(description="URL where job was found")
    source: str = Field(description="Platform e.g. LinkedIn, Naukri, Indeed")


# ── Query builder from resume profile ────────────────────────────────────────

def build_search_queries(profile: ResumeProfile) -> List[str]:
    """Generate targeted search queries from resume profile."""

    # role = profile.current_role
    top_skills = " ".join(profile.tools_and_frameworks[:3])
    job_role = profile.current_role
    queries = [
        f"GenAI jobs Hyderabad {top_skills} site:naukri.com",
        f"{job_role} LangChain LangGraph jobs Hyderabad site:naukri.com",
        f"GenAI openings Hyderabad site:linkedin.com",
        f"AI Engineer RAG LLM jobs Hyderabad 2025",
        f"GenAI Engineer Python jobs Hyderabad site:indeed.co.in",
    ]
    return queries


# ── SerpAPI search ────────────────────────────────────────────────────────────

def serpapi_search(query: str, max_results: int = 5) -> List[dict]:
    """Search Google via SerpAPI, return list of organic results."""
    try:
        params = {
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "num": max_results,
            "hl": "en",
            "gl": "in",          # India-focused results
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("organic_results", [])

    except Exception as e:
        print(f"  ⚠️  SerpAPI error: {e}")
        return []


# ── LLM-based job extractor from snippet ─────────────────────────────────────

def extract_job_from_snippet(result: dict) -> Optional[JobPosting]:
    """Use Gemini to extract a structured JobPosting from a SerpAPI result."""

    snippet_text = f"""
Title: {result.get('title', '')}
URL: {result.get('link', '')}
Snippet: {result.get('snippet', '')}
"""

    if len(result.get('snippet', '')) < 20:
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    structured_llm = llm.with_structured_output(JobPosting)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a job posting extractor.
Extract structured job details from the search result provided.
Infer company from URL or title if not explicitly stated.
Set apply_url to the link provided.
Set source based on domain: naukri.com → Naukri, linkedin.com → LinkedIn, indeed.co.in → Indeed.
If a field cannot be determined, use 'Unknown'."""),
        ("human", "{snippet_text}")
    ])

    chain = prompt | structured_llm

    try:
        job = chain.invoke({"snippet_text": snippet_text})
        return job
    except Exception as e:
        print(f"  ⚠️  Extraction failed: {e}")
        return None


# ── Main search orchestrator ──────────────────────────────────────────────────

def search_jobs_for_profile(
    profile: ResumeProfile,
    max_results_per_query: int = 5
) -> List[JobPosting]:
    """
    Full pipeline: Profile → Queries → SerpAPI → Extract jobs via Gemini.
    Returns a deduplicated list of JobPosting objects.
    """

    queries = build_search_queries(profile)
    all_jobs: List[JobPosting] = []
    seen_urls = set()

    print(f"\n🔍 Running {len(queries)} queries via SerpAPI...\n")

    for i, query in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {query}")

        results = serpapi_search(query, max_results=max_results_per_query)

        if not results:
            print(f"  ⚠️  No results returned")
            continue

        for result in results:
            url = result.get('link', '')
            if url in seen_urls:
                continue
            seen_urls.add(url)

            print(f"    → {result.get('title', url)[:65]}...")
            job = extract_job_from_snippet(result)

            if job:
                all_jobs.append(job)
                print(f"    ✅ {job.title} @ {job.company} | {job.source}")

            time.sleep(0.5)  # Polite delay between Gemini calls

        time.sleep(1)  # Delay between SerpAPI queries

    print(f"\n📋 Total jobs extracted: {len(all_jobs)}\n")
    return all_jobs


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from resume_parser import extract_resume_profile

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/GarnepudiVenkateshResume.pdf"

    print("📄 Step 1: Parsing resume...")
    profile = extract_resume_profile(pdf_path)
    print(f"✅ {profile.name} | {profile.current_role} | {profile.total_experience_years} yrs")

    print("\n🌐 Step 2: Searching jobs via SerpAPI...")
    jobs = search_jobs_for_profile(profile, max_results_per_query=4)

    print("=" * 60)
    for idx, job in enumerate(jobs, 1):
        print(f"\n[{idx}] {job.title} @ {job.company}")
        print(f"    📍 {job.location} | ⏳ {job.experience_required}")
        print(f"    🛠  {', '.join(job.skills_required[:5])}")
        print(f"    📝 {job.job_summary}")
        print(f"    🔗 {job.apply_url}")
    print("\n" + "=" * 60)