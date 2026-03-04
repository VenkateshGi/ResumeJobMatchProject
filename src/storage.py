# src/storage.py

import sqlite3
import json
import os
import sys
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))
from job_searcher import JobPosting
from job_scorer import JobMatchScore, ScoredJob


# ── DB path ───────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jobs.db")


# ── Init DB ───────────────────────────────────────────────────────────────────

def init_db():
    """Create the jobs table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            title                 TEXT NOT NULL,
            company               TEXT,
            location              TEXT,
            experience_required   TEXT,
            skills_required       TEXT,       -- JSON list
            job_summary           TEXT,
            apply_url             TEXT UNIQUE, -- prevents duplicates
            source                TEXT,

            -- Score fields
            overall_score         INTEGER,
            skill_match_score     INTEGER,
            experience_match_score INTEGER,
            matching_skills       TEXT,        -- JSON list
            missing_skills        TEXT,        -- JSON list
            strengths             TEXT,
            gaps                  TEXT,
            recommendation        TEXT,
            recommendation_reason TEXT,

            -- Tracker fields
            status                TEXT DEFAULT 'new',
            notes                 TEXT DEFAULT '',
            found_at              TEXT,
            applied_at            TEXT,
            updated_at            TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized at:", DB_PATH)


# ── Save scored job ───────────────────────────────────────────────────────────

def save_scored_job(scored: ScoredJob) -> bool:
    """
    Insert a scored job into the DB.
    Returns True if inserted, False if URL already exists (duplicate).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    try:
        cursor.execute("""
            INSERT INTO jobs (
                title, company, location, experience_required,
                skills_required, job_summary, apply_url, source,
                overall_score, skill_match_score, experience_match_score,
                matching_skills, missing_skills,
                strengths, gaps, recommendation, recommendation_reason,
                status, found_at, updated_at
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, 
                ?, ?, ?, ?,
                'new', ?, ?
            )
        """, (
            scored.job.title,
            scored.job.company,
            scored.job.location,
            scored.job.experience_required,
            json.dumps(scored.job.skills_required),
            scored.job.job_summary,
            scored.job.apply_url,
            scored.job.source,
            scored.score.overall_score,
            scored.score.skill_match_score,
            scored.score.experience_match_score,
            json.dumps(scored.score.matching_skills),
            json.dumps(scored.score.missing_skills),
            scored.score.strengths,
            scored.score.gaps,
            scored.score.recommendation,
            scored.score.recommendation_reason,
            now,
            now
        ))
        conn.commit()
        return True

    except sqlite3.IntegrityError:
        # Duplicate URL — skip silently
        return False

    finally:
        conn.close()


# ── Save all scored jobs ──────────────────────────────────────────────────────

def save_all_scored_jobs(scored_jobs: List[ScoredJob]) -> dict:
    """Save a list of scored jobs. Returns insert/duplicate counts."""
    inserted = 0
    duplicates = 0

    for scored in scored_jobs:
        if save_scored_job(scored):
            inserted += 1
        else:
            duplicates += 1

    return {"inserted": inserted, "duplicates": duplicates}


# ── Query jobs ────────────────────────────────────────────────────────────────

def get_all_jobs(
    min_score: int = 0,
    status: Optional[str] = None,
    recommendation: Optional[str] = None
) -> List[dict]:
    """Fetch jobs from DB with optional filters."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM jobs WHERE overall_score >= ?"
    params = [min_score]

    if status:
        query += " AND status = ?"
        params.append(status)

    if recommendation:
        query += " AND recommendation = ?"
        params.append(recommendation)

    query += " ORDER BY overall_score DESC"

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Deserialize JSON fields
    for row in rows:
        row['skills_required'] = json.loads(row['skills_required'] or '[]')
        row['matching_skills'] = json.loads(row['matching_skills'] or '[]')
        row['missing_skills'] = json.loads(row['missing_skills'] or '[]')

    return rows


# ── Update application status ─────────────────────────────────────────────────

def update_job_status(job_id: int, status: str, notes: str = "") -> bool:
    """
    Update the status of a job.
    Valid statuses: new, applied, interviewing, rejected, offered
    """
    valid_statuses = {"new", "applied", "interviewing", "rejected", "offered"}
    if status not in valid_statuses:
        print(f"⚠️  Invalid status '{status}'. Use: {valid_statuses}")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    applied_at = now if status == "applied" else None

    cursor.execute("""
        UPDATE jobs
        SET status = ?, notes = ?, updated_at = ?,
            applied_at = COALESCE(CASE WHEN ? = 'applied' THEN ? ELSE applied_at END, applied_at)
        WHERE id = ?
    """, (status, notes, now, status, applied_at, job_id))

    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


# ── Pretty print jobs ─────────────────────────────────────────────────────────

def print_jobs_table(jobs: List[dict]):
    """Print jobs in a clean table format."""

    if not jobs:
        print("  No jobs found.")
        return

    status_emoji = {
        "new": "🆕", "applied": "📤", "interviewing": "🎯",
        "rejected": "❌", "offered": "🎉"
    }
    rec_emoji = {
        "Strong Apply": "🟢", "Apply": "🟡",
        "Consider": "🟠", "Skip": "🔴"
    }

    print(f"\n{'─'*70}")
    print(f"  {'#':<4} {'Score':<7} {'Title':<30} {'Company':<18} {'Status'}")
    print(f"{'─'*70}")

    for job in jobs:
        se = status_emoji.get(job['status'], '⚪')
        re = rec_emoji.get(job['recommendation'], '⚪')
        title = job['title'][:28]
        company = job['company'][:16]
        print(f"  {job['id']:<4} {re}{job['overall_score']:<5} {title:<30} {company:<18} {se} {job['status']}")

    print(f"{'─'*70}\n")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from resume_parser import extract_resume_profile
    from job_searcher import search_jobs_for_profile
    from job_scorer import score_all_jobs

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/GarnepudiVenkateshResume.pdf"

    # Init DB
    init_db()

    print("📄 Step 1: Parsing resume...")
    profile = extract_resume_profile(pdf_path)
    print(f"✅ {profile.name} | {profile.current_role} | {profile.total_experience_years} yrs")

    print("\n🌐 Step 2: Searching jobs...")
    jobs = search_jobs_for_profile(profile, max_results_per_query=3)

    print("\n🎯 Step 3: Scoring matches...")
    scored_jobs = score_all_jobs(profile, jobs)

    print("\n💾 Step 4: Saving to database...")
    result = save_all_scored_jobs(scored_jobs)
    print(f"  ✅ Inserted: {result['inserted']} | Duplicates skipped: {result['duplicates']}")

    print("\n📋 All jobs in DB (sorted by score):")
    all_jobs = get_all_jobs(min_score=0)
    print_jobs_table(all_jobs)

    print("\n🟢 Strong Apply + Apply jobs only:")
    top_jobs = get_all_jobs(min_score=70)
    print_jobs_table(top_jobs)

    # Demo: update a status
    if all_jobs:
        first_id = all_jobs[0]['id']
        update_job_status(first_id, "applied", notes="Applied via Naukri on portal")
        print(f"  📤 Marked job #{first_id} as 'applied'")

    print("\n📤 Jobs marked as applied:")
    applied = get_all_jobs(status="applied")
    print_jobs_table(applied)