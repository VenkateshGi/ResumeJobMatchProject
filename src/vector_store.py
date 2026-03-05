# src/vector_store.py

import os
import sys
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from resume_parser import ResumeProfile
from job_searcher import JobPosting
from job_scorer import ScoredJob

load_dotenv()


# ── Paths & collection names ──────────────────────────────────────────────────

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", ".chroma")
JOBS_COLLECTION      = "job_postings"        # used by direct ChromaDB client
LANGCHAIN_COLLECTION = "langchain_jobs"      # used by LangChain Chroma


# ── Document builders ─────────────────────────────────────────────────────────

def build_resume_document(profile: ResumeProfile) -> str:
    """Convert ResumeProfile to rich text for embedding."""
    return f"""
Name: {profile.name}
Role: {profile.current_role}
Experience: {profile.total_experience_years} years
Education: {profile.education}
Summary: {profile.summary}
Skills: {', '.join(profile.skills)}
Tools & Frameworks: {', '.join(profile.tools_and_frameworks)}
""".strip()


def build_job_document(job: JobPosting) -> str:
    """Convert JobPosting to rich text for embedding."""
    return f"""
Title: {job.title}
Company: {job.company}
Location: {job.location}
Experience Required: {job.experience_required}
Skills Required: {', '.join(job.skills_required)}
Summary: {job.job_summary}
Source: {job.source}
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Direct ChromaDB (raw vector store operations)
# ══════════════════════════════════════════════════════════════════════════════

def get_chroma_client() -> chromadb.PersistentClient:
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_embedding_function():
    """Google Generative AI embedding function for direct ChromaDB."""
    return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model_name="models/gemini-embedding-001"
    )


def store_resume(profile: ResumeProfile):
    """Embed and store resume profile in ChromaDB."""
    client = get_chroma_client()
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name="resume_profiles",
        embedding_function=ef
    )

    doc    = build_resume_document(profile)
    doc_id = f"resume_{profile.name.replace(' ', '_').lower()}"

    collection.upsert(
        ids=[doc_id],
        documents=[doc],
        metadatas=[{
            "name":       profile.name,
            "role":       profile.current_role,
            "experience": str(profile.total_experience_years),
            "skills":     ", ".join(profile.skills[:10])
        }]
    )
    print(f"  ✅ Resume stored  →  id: {doc_id}")
    return doc_id


def store_jobs_direct(scored_jobs: List[ScoredJob]):
    """
    Embed and store scored jobs using direct ChromaDB client.
    Skips duplicates by URL.
    """
    client = get_chroma_client()
    ef     = get_embedding_function()
    collection = client.get_or_create_collection(
        name=JOBS_COLLECTION,
        embedding_function=ef
    )

    inserted = skipped = 0

    for scored in scored_jobs:
        job    = scored.job
        doc    = build_job_document(job)
        doc_id = job.apply_url.replace("https://", "").replace("/", "_")[:80]

        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            skipped += 1
            continue

        collection.add(
            ids=[doc_id],
            documents=[doc],
            metadatas=[{
                "title":          job.title,
                "company":        job.company,
                "location":       job.location,
                "source":         job.source,
                "overall_score":  str(scored.score.overall_score),
                "recommendation": scored.score.recommendation,
                "apply_url":      job.apply_url,
                "missing_skills": ", ".join(scored.score.missing_skills[:5])
            }]
        )
        inserted += 1

    print(f"  ✅ Direct ChromaDB  →  inserted: {inserted} | skipped: {skipped}")
    return inserted


def find_similar_jobs_direct(profile: ResumeProfile, top_k: int = 5) -> List[dict]:
    """
    Semantic search — resume profile vs stored job vectors.
    Returns top-K most similar jobs with similarity score.
    """
    client = get_chroma_client()
    ef     = get_embedding_function()
    collection = client.get_or_create_collection(
        name=JOBS_COLLECTION,
        embedding_function=ef
    )

    total = collection.count()
    if total == 0:
        print("  ⚠️  No jobs in vector store yet.")
        return []

    results = collection.query(
        query_texts=[build_resume_document(profile)],
        n_results=min(top_k, total)
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta       = results["metadatas"][0][i]
        distance   = results["distances"][0][i]
        similarity = round((1 - distance) * 100, 1)

        hits.append({
            "rank":            i + 1,
            "similarity_score": similarity,
            "title":           meta.get("title"),
            "company":         meta.get("company"),
            "location":        meta.get("location"),
            "source":          meta.get("source"),
            "llm_score":       meta.get("overall_score"),
            "recommendation":  meta.get("recommendation"),
            "apply_url":       meta.get("apply_url"),
            "missing_skills":  meta.get("missing_skills", "")
        })

    return hits


def print_semantic_results(hits: List[dict]):
    """Pretty-print direct ChromaDB semantic results."""
    rec_emoji = {
        "Strong Apply": "🟢", "Apply": "🟡",
        "Consider":     "🟠", "Skip":  "🔴"
    }

    print(f"\n{'═'*65}")
    print(f"  🧠 SEMANTIC MATCH RESULTS  (ChromaDB + Gemini Embeddings)")
    print(f"{'═'*65}")

    for h in hits:
        filled = "█" * int(h["similarity_score"] // 10)
        empty  = "░" * (10 - int(h["similarity_score"] // 10))
        re     = rec_emoji.get(h["recommendation"], "⚪")

        print(f"""
  [{h['rank']}] {h['title']} @ {h['company']}
      📍 {h['location']} | 🌐 {h['source']}
      Semantic : {filled}{empty} {h['similarity_score']}%
      LLM Score: {h['llm_score']}/100  {re} {h['recommendation']}
      ❌ Gaps  : {h['missing_skills'] or 'None'}
      🔗 {h['apply_url']}""")

    print(f"\n{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — LangChain RAG (proper retriever → LLM chain)
# ══════════════════════════════════════════════════════════════════════════════

def build_langchain_vectorstore(scored_jobs: List[ScoredJob]) -> Chroma:
    """
    Build a LangChain Chroma vectorstore from scored jobs.
    Uses LangChain Document objects — this is proper RAG indexing.

    Shape journey per chunk:
      text → tokenize → [1 × T × 512] → attention × N → mean pool → [1 × 768]
    768-dim vectors stored in ChromaDB.
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Convert each scored job to a LangChain Document
    docs = []
    for scored in scored_jobs:
        job = scored.job
        doc = Document(
            page_content=build_job_document(job),
            metadata={
                "title":           job.title,
                "company":         job.company,
                "location":        job.location,
                "source":          job.source,
                "overall_score":   scored.score.overall_score,
                "recommendation":  scored.score.recommendation,
                "apply_url":       job.apply_url,
                "missing_skills":  ", ".join(scored.score.missing_skills[:5])
            }
        )
        docs.append(doc)

    # Index into ChromaDB via LangChain
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=LANGCHAIN_COLLECTION
    )

    print(f"  ✅ LangChain vectorstore  →  {len(docs)} documents indexed")
    return vectorstore


def load_langchain_vectorstore() -> Chroma:
    """Load existing LangChain vectorstore from disk."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=LANGCHAIN_COLLECTION
    )


def build_rag_chain(vectorstore: Chroma):
    """
    Build the full LangChain RAG chain:

    Question
        ↓
    Retriever  (similarity search → top-4 job docs)
        ↓
    Prompt     (question + retrieved job context)
        ↓
    Gemini     (reads context, generates grounded answer)
        ↓
    Answer
    """

    # Step 1 — retriever: top-4 most similar job docs
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Step 2 — LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Step 3 — prompt template (context injected here)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a career advisor helping a job seeker.
Use ONLY the retrieved job postings below to answer the question.
Be specific — mention actual company names, roles, skills from the context.
If the answer is not in the context, say 'Not found in current job data'.

Retrieved Job Postings:
{context}
"""),
        ("human", "{question}")
    ])

    # Step 4 — format retrieved docs into readable context string
    def format_docs(docs: list) -> str:
        return "\n\n---\n\n".join([
            f"Job {i+1}: {d.metadata.get('title')} @ {d.metadata.get('company')}\n"
            f"Location: {d.metadata.get('location')}\n"
            f"Score: {d.metadata.get('overall_score')}/100 | "
            f"{d.metadata.get('recommendation')}\n"
            f"Missing Skills: {d.metadata.get('missing_skills', 'None')}\n"
            f"Apply: {d.metadata.get('apply_url')}\n\n"
            f"{d.page_content}"
            for i, d in enumerate(docs)
        ])

    # Step 5 — wire the RAG chain together
    # This is the LCEL (LangChain Expression Language) chain:
    # context = retrieve docs → format
    # question = pass through as-is
    # → inject both into prompt → LLM → parse output
    rag_chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Full Phase 5 pipeline
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from resume_parser import extract_resume_profile
    from job_searcher import search_jobs_for_profile
    from job_scorer import score_all_jobs
    from storage import init_db, save_all_scored_jobs

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/GarnepudiVenkateshResume.pdf"

    # ── Step 1: Parse resume ──────────────────────────────────────────────────
    print("\n📄 Step 1: Parsing resume...")
    profile = extract_resume_profile(pdf_path)
    print(f"  ✅ {profile.name} | {profile.current_role} | {profile.total_experience_years} yrs")

    # ── Step 2: Search jobs ───────────────────────────────────────────────────
    print("\n🌐 Step 2: Searching jobs via SerpAPI...")
    jobs = search_jobs_for_profile(profile, max_results_per_query=3)

    # ── Step 3: Score jobs ────────────────────────────────────────────────────
    print("\n🎯 Step 3: Scoring matches with Gemini...")
    scored_jobs = score_all_jobs(profile, jobs)

    # ── Step 4: Save to SQLite ────────────────────────────────────────────────
    print("\n💾 Step 4: Saving to SQLite...")
    init_db()
    result = save_all_scored_jobs(scored_jobs)
    print(f"  ✅ Inserted: {result['inserted']} | Duplicates: {result['duplicates']}")

    # ── Step 5a: Store in ChromaDB (direct) ──────────────────────────────────
    print("\n🧠 Step 5a: Storing in ChromaDB (direct)...")
    store_resume(profile)
    store_jobs_direct(scored_jobs)

    # ── Step 5b: Semantic search (direct ChromaDB) ────────────────────────────
    print("\n🔍 Step 5b: Semantic search — top 5 matches...")
    hits = find_similar_jobs_direct(profile, top_k=5)
    print_semantic_results(hits)

    # ── Step 5c: Build LangChain RAG vectorstore ──────────────────────────────
    print("\n🔗 Step 5c: Building LangChain RAG vectorstore...")
    vectorstore = build_langchain_vectorstore(scored_jobs)

    # ── Step 5d: Build RAG chain ──────────────────────────────────────────────
    print("\n⛓️  Step 5d: Building RAG chain (Retriever → Gemini)...")
    rag_chain = build_rag_chain(vectorstore)

    # ── Step 5e: Ask questions using RAG ─────────────────────────────────────
    questions = [
        "Which jobs best match my LangChain and RAG experience?",
        "What are the most common missing skills across all jobs?",
        "Which companies are hiring and what seniority do they require?"
    ]

    print(f"\n{'═'*65}")
    print("  🤖 RAG Q&A  —  Answers grounded in your job search data")
    print(f"{'═'*65}")

    for q in questions:
        print(f"\n  ❓ {q}")
        print(f"  {'─'*60}")
        answer = rag_chain.invoke(q)
        print(f"  💬 {answer}")
        print(f"  {'─'*60}")

    print(f"\n{'═'*65}\n")
