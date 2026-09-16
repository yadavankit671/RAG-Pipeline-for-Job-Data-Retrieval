"""Retrieve the most relevant job postings and synthesize grounded answers.

This module wraps ChromaDB retrieval and the Ollama model. It converts raw user
queries into semantic search calls, optionally applies metadata filters, and
returns both the generated answer and the source job IDs that supported it.
"""

from __future__ import annotations

import chromadb
import ollama

DB_PATH = r"D:\Assignment\RAG\Data\chroma_db"
COLLECTION_NAME = "job_postings"
OLLAMA_MODEL = "llama3.1" 
TOP_K_DEFAULT = 5
# =============================================================================

_collection = None  # lazy singleton -- avoid reopening the DB on every call


def get_collection():
    """Return a cached Chroma collection for all retrieval calls in this process."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def build_where(filters: dict | None) -> dict | None:
    """Translate a simple metadata filter dictionary into Chroma ``where`` syntax.

    Example input: {"job_level": "Senior Level", "job_location": "New York, NY"}
    """
    if not filters:
        return None
    conditions = [{k: v} for k, v in filters.items() if v]
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve(query: str, top_k: int = TOP_K_DEFAULT, filters: dict | None = None) -> list[dict]:
    """Run semantic retrieval and return the top matching jobs.

    Each result includes the job ID, document chunk text, metadata fields, and the
    cosine distance score. Lower distance means a more similar match.
    """
    collection = get_collection()
    where = build_where(filters)

    results = collection.query(query_texts=[query], n_results=top_k, where=where)

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []
    for job_id, doc, meta, dist in zip(ids, docs, metas, distances):
        record = dict(meta)
        record["job_id"] = job_id
        record["chunk_text"] = doc
        record["distance"] = dist
        matches.append(record)
    return matches


GENERATION_SYSTEM_PROMPT = """You are a helpful assistant answering questions about job \
postings. You will be given a user question and a set of retrieved job \
postings as context. Answer using ONLY the information in the provided \
postings -- do not use outside knowledge and do not invent details.

Rules:
- Cite the job(s) you draw on by their ID in square brackets, e.g. [LF0001].
- If the postings don't contain enough information to answer the question, \
  say so plainly rather than guessing.
- Be concise and directly answer what was asked; don't just list the postings.
"""


def format_context(retrieved: list[dict]) -> str:
    """Convert retrieved records into a compact prompt context block for the LLM."""
    blocks = []
    for r in retrieved:
        blocks.append(
            f"[{r['job_id']}]\n{r.get('chunk_text', '')}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, retrieved: list[dict]) -> str:
    """Generate a grounded answer using only the retrieved job postings as context."""
    if not retrieved:
        return "I couldn't find any job postings relevant to that question."

    context = format_context(retrieved)
    user_msg = f"Question: {query}\n\nRelevant job postings:\n\n{context}"

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.2},
    )
    return resp["message"]["content"]


def answer_question(query: str, top_k: int = TOP_K_DEFAULT, filters: dict | None = None) -> dict:
    """Retrieve relevant jobs and return a grounded response with citation metadata."""
    retrieved = retrieve(query, top_k=top_k, filters=filters)
    answer = generate_answer(query, retrieved)
    return {
        "answer": answer,
        "sources": [
            {
                "job_id": r["job_id"],
                "job_title": r.get("job_title"),
                "company_name": r.get("company_name"),
                "job_location": r.get("job_location"),
                "job_level": r.get("job_level"),
                "distance": r.get("distance"),
            }
            for r in retrieved
        ],
    }
