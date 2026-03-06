import os
import sys 
from typing import List,TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.dirname(__file__))
from resume_parser import ResumeProfile, extract_resume_profile
from job_searcher import JobPosting, search_jobs_for_profile
from job_scorer import ScoredJob, score_all_jobs
from storage import init_db, save_all_scored_jobs, get_all_jobs, print_jobs_table
from vector_store import (
    store_resume,
    store_jobs_direct,
    build_langchain_vectorstore,
    load_langchain_vectorstore,
    build_rag_chain,
    find_similar_jobs_direct,
    print_semantic_results
)

load_dotenv()


class AgentState(TypedDict):
    pdf_path: str
    # Intermediate results
    profile:          Optional[ResumeProfile]
    raw_jobs:         List[JobPosting]
    scored_jobs:      List[ScoredJob]

    # RAG outputs
    rag_answers:      List[dict]          # [{question, answer}]
    semantic_hits:    List[dict]

    # Control flow
    error:            Optional[str]
    status:           str                 # current stage
    jobs_found:       int
    jobs_saved:       int


def node_parse_resume(state: AgentState) -> AgentState:
    """Node 1 — Parse PDF resume into structured profile."""

    print("\n" + "─"*60)
    print("🤖 [Node 1/6] Parsing Resume...")
    print("─"*60)

    try:
        profile = extract_resume_profile(state["pdf_path"])
        print(f"  ✅ {profile.name}")
        print(f"  📌 Role      : {profile.current_role}")
        print(f"  ⏳ Experience: {profile.total_experience_years} years")
        print(f"  🛠  Top skills: {', '.join(profile.tools_and_frameworks[:5])}")

        return {
            **state,
            "profile": profile,
            "status":  "resume_parsed",
            "error":   None
        }

    except Exception as e:
        print(f"  ❌ Parse failed: {e}")
        return {**state, "error": str(e), "status": "failed"}
    
def node_search_jobs(state: AgentState) -> AgentState:
    """Node 2 — Search for live job postings via SerpAPI."""

    print("\n" + "─"*60)
    print("🤖 [Node 2/6] Searching Jobs...")
    print("─"*60)

    try:
        jobs = search_jobs_for_profile(
            state["profile"],
            max_results_per_query=3
        )

        print(f"  ✅ Found {len(jobs)} job postings")

        return {
            **state,
            "raw_jobs":   jobs,
            "jobs_found": len(jobs),
            "status":     "jobs_found" if jobs else "no_jobs_found",
            "error":      None
        }

    except Exception as e:
        print(f"  ❌ Search failed: {e}")
        return {**state, "error": str(e), "status": "failed"}

def node_score_jobs(state: AgentState) -> AgentState:
    """Node 3 — Score each job against resume with Gemini."""

    print("\n" + "─"*60)
    print("🤖 [Node 3/6] Scoring Job Matches...")
    print("─"*60)

    try:
        scored_jobs = score_all_jobs(
            state["profile"],
            state["raw_jobs"]
        )

        strong = [s for s in scored_jobs
                  if s.score.recommendation in ["Strong Apply", "Apply"]]

        print(f"  ✅ Scored {len(scored_jobs)} jobs")
        print(f"  🟢 Worth applying: {len(strong)}")
        print(f"  🔴 Skip          : {len(scored_jobs) - len(strong)}")

        return {
            **state,
            "scored_jobs": scored_jobs,
            "status":      "jobs_scored",
            "error":       None
        }

    except Exception as e:
        print(f"  ❌ Scoring failed: {e}")
        return {**state, "error": str(e), "status": "failed"}
    

def node_save_jobs(state: AgentState) -> AgentState:
    """Node 4 — Persist jobs to SQLite + ChromaDB vector store."""

    print("\n" + "─"*60)
    print("🤖 [Node 4/6] Saving to SQLite + ChromaDB...")
    print("─"*60)

    try:
        # SQLite
        init_db()
        result = save_all_scored_jobs(state["scored_jobs"])
        print(f"  💾 SQLite   → inserted: {result['inserted']} | "
              f"duplicates: {result['duplicates']}")

        # ChromaDB direct
        store_resume(state["profile"])
        store_jobs_direct(state["scored_jobs"])

        # LangChain vectorstore
        build_langchain_vectorstore(state["scored_jobs"])
        print(f"  🧠 ChromaDB → LangChain vectorstore built")

        return {
            **state,
            "jobs_saved": result["inserted"],
            "status":     "jobs_saved",
            "error":      None
        }

    except Exception as e:
        print(f"  ❌ Save failed: {e}")
        return {**state, "error": str(e), "status": "failed"}
    
def node_rag_insights(state: AgentState) -> AgentState:
    """Node 5 — Run RAG Q&A over stored job data."""

    print("\n" + "─"*60)
    print("🤖 [Node 5/6] Running RAG Insights...")
    print("─"*60)

    try:
        # Semantic search (direct ChromaDB)
        hits = find_similar_jobs_direct(state["profile"], top_k=5)
        print_semantic_results(hits)

        # LangChain RAG Q&A
        vectorstore = load_langchain_vectorstore()
        rag_chain   = build_rag_chain(vectorstore)

        questions = [
            f"Which jobs best match my {state['profile'].current_role} background?",
            "What are the most common missing skills I should upskill in?",
            "Which companies are hiring and what experience do they require?"
        ]

        rag_answers = []
        for q in questions:
            print(f"\n  ❓ {q}")
            answer = rag_chain.invoke(q)
            print(f"  💬 {answer[:200]}...")
            rag_answers.append({"question": q, "answer": answer})

        return {
            **state,
            "rag_answers":   rag_answers,
            "semantic_hits": hits,
            "status":        "rag_complete",
            "error":         None
        }

    except Exception as e:
        print(f"  ❌ RAG failed: {e}")
        return {**state, "error": str(e), "status": "failed"}
    
def node_summarise(state: AgentState) -> AgentState:
    """Node 6 — Print final summary report."""

    print("\n" + "═"*60)
    print("🤖 [Node 6/6] Final Summary Report")
    print("═"*60)

    profile = state["profile"]
    scored  = state["scored_jobs"]

    strong_apply = [s for s in scored
                    if s.score.recommendation == "Strong Apply"]
    apply        = [s for s in scored
                    if s.score.recommendation == "Apply"]
    skip         = [s for s in scored
                    if s.score.recommendation == "Skip"]

    print(f"""
  👤 Candidate  : {profile.name}
  🏢 Target Role: {profile.current_role}
  ⏳ Experience : {profile.total_experience_years} years

  📊 Job Search Results:
  ─────────────────────────────────────────
  🟢 Strong Apply : {len(strong_apply)} jobs
  🟡 Apply        : {len(apply)} jobs
  🔴 Skip         : {len(skip)} jobs
  💾 Saved to DB  : {state['jobs_saved']} new entries

  🏆 Top 3 Jobs to Apply:""")

    top3 = sorted(scored,
                  key=lambda x: x.score.overall_score,
                  reverse=True)[:3]

    for i, s in enumerate(top3, 1):
        print(f"""
  [{i}] {s.job.title} @ {s.job.company}
       Score : {s.score.overall_score}/100
       Gap   : {', '.join(s.score.missing_skills[:3]) or 'None'}
       URL   : {s.job.apply_url}""")

    print(f"""
  📚 Key Skill Gaps to Close:""")

    # Aggregate missing skills across all jobs
    all_missing = []
    for s in scored:
        all_missing.extend(s.score.missing_skills)

    from collections import Counter
    top_gaps = Counter(all_missing).most_common(5)
    for skill, count in top_gaps:
        print(f"     • {skill} (missing in {count} jobs)")

    print(f"\n  🤖 RAG Insights:")
    for qa in state.get("rag_answers", []):
        print(f"\n  ❓ {qa['question']}")
        print(f"  💬 {qa['answer'][:300]}")

    print(f"\n{'═'*60}")
    print("  ✅ Agent pipeline complete!")
    print(f"{'═'*60}\n")

    return {**state, "status": "complete"}

def should_continue_after_search(state: AgentState) -> str:
    """
    After job search:
    - If jobs found → proceed to scoring
    - If no jobs    → end with message
    - If error      → end with error
    """
    if state.get("error"):
        return "end"
    if state.get("jobs_found", 0) == 0:
        print("\n  ⚠️  No jobs found. Try different search queries.")
        return "end"
    return "continue"

def should_continue_after_parse(state: AgentState) -> str:
    """
    After resume parse:
    - If parsed successfully → search jobs
    - If error              → end
    """
    if state.get("error"):
        print(f"\n  ❌ Pipeline stopped: {state['error']}")
        return "end"
    return "continue"

def build_agent() -> StateGraph:
    """
    Wire all nodes and edges into a LangGraph StateGraph.

    Graph structure:
    START
      ↓
    parse_resume ──(error)──→ END
      ↓ (success)
    search_jobs ──(no jobs)──→ END
      ↓ (jobs found)
    score_jobs
      ↓
    save_jobs
      ↓
    rag_insights
      ↓
    summarise
      ↓
    END
    """

    graph = StateGraph(AgentState)

    graph.add_node("parse_resume",  node_parse_resume)
    graph.add_node("search_jobs",   node_search_jobs)
    graph.add_node("score_jobs",    node_score_jobs)
    graph.add_node("save_jobs",     node_save_jobs)
    graph.add_node("rag_insights",  node_rag_insights)
    graph.add_node("summarise",     node_summarise)

    # ── Entry point ───────────────────────────────────────────────
    graph.set_entry_point("parse_resume")

    # ── Conditional edges ─────────────────────────────────────────
    graph.add_conditional_edges(
        "parse_resume",
        should_continue_after_parse,
        {
            "continue": "search_jobs",
            "end":      END
        }
    )

    graph.add_conditional_edges(
        "search_jobs",
        should_continue_after_search,
        {
            "continue": "score_jobs",
            "end":      END
        }
    )

    # ── Linear edges ──────────────────────────────────────────────
    graph.add_edge("score_jobs",   "save_jobs")
    graph.add_edge("save_jobs",    "rag_insights")
    graph.add_edge("rag_insights", "summarise")
    graph.add_edge("summarise",    END)

    return graph.compile()

if __name__ == "__main__":

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/GarnepudiVenkateshResume.pdf"

    print("\n" + "═"*60)
    print("  🚀 Resume Job Matcher — LangGraph Agent")
    print("═"*60)

    # Build the agent graph
    agent = build_agent()

    # Initial state
    initial_state: AgentState = {
        "pdf_path":      pdf_path,
        "profile":       None,
        "raw_jobs":      [],
        "scored_jobs":   [],
        "rag_answers":   [],
        "semantic_hits": [],
        "error":         None,
        "status":        "starting",
        "jobs_found":    0,
        "jobs_saved":    0
    }

    # Run the agent
    final_state = agent.invoke(initial_state)

    print(f"\n  Pipeline status: {final_state['status']}")