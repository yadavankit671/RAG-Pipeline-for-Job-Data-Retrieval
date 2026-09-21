
"""Build the final job-posting vector records used for semantic retrieval.

This module reads the cleaned metadata and description JSONL files, asks Groq to
summarize each job posting into a structured record, and upserts the resulting
searchable document together with metadata into ChromaDB. The script is designed
for resumable indexing: already-processed jobs are skipped automatically on rerun.
"""

from __future__ import annotations
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import chromadb
import ollama

METADATA_FILE = "D:\Assignment\RAG\Data\json\metadata.jsonl"
DESCRIPTIONS_FILE = "D:\Assignment\RAG\Data\json\description_clean.jsonl"
DB_PATH = "D:\Assignment\RAG\Data\chroma_db"
COLLECTION_NAME = "job_postings"


# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# GROQ_MODEL = "openai/gpt-oss-20b"
OLLAMA_MODEL ="qwen2.5:1.5b" # "llama3.2:3B"
CONCURRENCY = 1

LIMIT = None                       
AUDIT_FILE = "chunks.jsonl"        
TEST_QUERY = None
TEST_FILTER = None



SYSTEM_PROMPT = """You are extracting and summarizing structured information from a job \
posting for a retrieval system. You will be given the cleaned text of one job \
posting (headers are marked with "## ", list items with "- "). Read the whole \
posting and respond with ONLY a JSON object matching the required schema -- \
no other text before or after it.

Rules:
- Do NOT copy long verbatim spans; write concise original summaries (1-4 \
  sentences per field, or short bullet-style sentences packed into one string).
- Merge "required" and "preferred" qualifications into one `requirements` \
  summary; you may note which items are preferred if relevant.
- `about_company` should summarize what the company/business does and any \
  culture/mission language, NOT the role itself.
- `compensation` should capture pay range, pay type (hourly/salary), bonus, \
  equity, or benefits-adjacent pay info if stated; otherwise use null.
- If a field genuinely has no content in the posting, use null for that \
  field rather than inventing content.
- Keep summaries factual and neutral; do not add opinions or marketing tone \
  beyond what's in the source.
"""

# JSON Schema
CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "about_company": {
            "type": ["string", "null"],
            "description": "1-4 sentence summary of the company/business (not the role).",
        },
        "responsibilities": {
            "type": ["string", "null"],
            "description": "Summary of what the role actually does day-to-day.",
        },
        "requirements": {
            "type": ["string", "null"],
            "description": "Summary of required + preferred qualifications/skills/experience.",
        },
        "benefits": {
            "type": ["string", "null"],
            "description": "Summary of benefits/perks offered (health, 401k, PTO, etc.).",
        },
        "compensation": {
            "type": ["string", "null"],
            "description": "Pay range/type/bonus info if stated, else null.",
        },
    },
    "required": ["about_company", "responsibilities", "requirements", "benefits", "compensation"],
}


def load_jsonl(path: Path) -> dict:
    """Load a JSON Lines file into a dictionary keyed by the job ID.

    Args:
        path: Path to the JSONL file.

    Returns:
        A mapping of job ID strings to their JSON object payload.
    """
    records = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[str(obj["ID"]).strip()] = obj
    return records


def get_collection(db_path: str, collection_name: str):
    """Open or create the persistent Chroma collection for this project.

    The collection stores one document per job. Each document contains the text
    used for embedding and semantic search, while the metadata holds the filtered
    fields used by the API and UI.
    """
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def existing_ids_in_db(collection) -> set:
    """Return the IDs of jobs already processed in Chroma.

    The collection may contain placeholder rows inserted by the metadata step, so
    this function only treats a job as complete when the summary fields expected
    from the Groq chunker are present in metadata.
    """
    try:
        # Ask Chroma to return the metadata along with the IDs
        existing = collection.get(include=["metadatas"])
        done_ids = set()
        
        if not existing or not existing.get("ids"):
            return done_ids
            
        for job_id, meta in zip(existing["ids"], existing["metadatas"]):
            if meta and ("responsibilities" in meta or "requirements" in meta):
                done_ids.add(job_id)
                
        return done_ids
    except Exception as e:
        print(f"Warning: Could not fetch existing IDs for resume check: {e}")
        return set()


def sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool -- no None. Drop
    or stringify anything else so upsert never fails on a missing field."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean



def summarize_description(description_text: str, job_title: str, company_name: str,
                        max_retries: int = 3) -> dict:
    """Ask Ollama to summarize a single job posting into structured fields.

    The returned dictionary matches the schema expected by the retrieval index and
    contains company overview, responsibilities, requirements, benefits, and
    compensation text. If the posting is empty, it returns a null-filled record.
    """
    if not description_text.strip():
        return {
            "about_company": None,
            "responsibilities": None,
            "requirements": None,
            "benefits": None,
            "compensation": None,
        }

    user_msg = (
        f"Job title: {job_title}\nCompany: {company_name}\n\n"
        f"--- Posting text ---\n{description_text}\n\n"
        f"CRITICAL: Output your response strictly as a JSON object matching this exact schema:\n"
        f"{json.dumps(CHUNK_SCHEMA, indent=2)}"
    )

    delay = 4.0
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                format = CHUNK_SCHEMA
            )
            
            content = resp["message"]["content"]
            data = json.loads(content)

            for key in ("about_company", "responsibilities", "requirements", "benefits", "compensation"):
                data.setdefault(key, None)
                
            time.sleep(2.5)
            
            return data
            
        except Exception as e: 
            last_err = e
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed after {max_retries} attempts: {last_err}"
                ) from last_err
            
            time.sleep(delay)
            delay *= 2
            
    raise RuntimeError("unreachable")


def build_chunk_text(meta: dict, summary: dict) -> str:
    """Build the text string that will be embedded and semantically searched.

    The document text is intentionally human-readable and combines metadata with
    the summary fields so retrieval has enough context to find relevant jobs.
    """
    parts = [
        f"Job Title: {meta.get('Job Title', '')}",
        f"Company: {meta.get('Company Name', '')}",
        f"Location: {meta.get('Job Location', '')}",
        f"Level: {meta.get('Job Level', '')}",
        f"Category: {meta.get('Job Category', '')}",
    ]
    field_labels = [
        ("about_company", "About the Company"),
        ("responsibilities", "Responsibilities"),
        ("requirements", "Requirements"),
        ("benefits", "Benefits"),
        ("compensation", "Compensation"),
    ]
    for key, label in field_labels:
        value = summary.get(key)
        if value:
            parts.append(f"\n{label}: {value}")
    return "\n".join(parts).strip()


def process_one(job_id, meta, desc_record) -> dict:
    """Summarize a single job and package it for database upsert.

    Args:
        job_id: Source job identifier.
        meta: Metadata row for the job.
        desc_record: Cleaned description payload loaded from the JSONL export.

    Returns:
        A dictionary containing both the searchable document text and the metadata
        used in the Chroma collection.
    """
    description_text = desc_record.get("job_description_clean", "") if desc_record else ""
    job_title = meta.get("Job Title", "")
    company_name = meta.get("Company Name", "")

    summary = summarize_description(description_text, job_title, company_name)

    chunk = {
        "job_id": job_id,
        "job_title": job_title,
        "company_name": company_name,
        "job_category": meta.get("Job Category"),
        "job_level": meta.get("Job Level"),
        "job_location": meta.get("Job Location"),
        "publication_date": meta.get("Publication Date"),
        "about_company": summary.get("about_company"),
        "responsibilities": summary.get("responsibilities"),
        "requirements": summary.get("requirements"),
        "benefits": summary.get("benefits"),
        "compensation": summary.get("compensation"),
    }
    chunk["chunk_text"] = build_chunk_text(meta, summary)
    return chunk


def ingest():
    meta_path = Path(METADATA_FILE)
    desc_path = Path(DESCRIPTIONS_FILE)

    if not meta_path.exists():
        sys.exit(f"Metadata file not found: {meta_path}")
    if not desc_path.exists():
        sys.exit(f"Cleaned descriptions file not found: {desc_path}. Run description_cleaner.py first.")
    if ollama is None:
        sys.exit("Missing dependency. Run: pip install ollama")

    metadata = load_jsonl(meta_path)
    descriptions = load_jsonl(desc_path)

    collection = get_collection(DB_PATH, COLLECTION_NAME) 
    done_ids = existing_ids_in_db(collection)  

    job_ids = [jid for jid in metadata.keys() if jid not in done_ids]
    if LIMIT:
        job_ids = job_ids[:LIMIT]

    if not job_ids:
        print("Nothing to do -- every job in metadata.jsonl is already in the DB.")
        return

    print(f"Summarizing {len(job_ids)} jobs with '{OLLAMA_MODEL}' ({len(done_ids)} already in DB, skipping)...")

    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(process_one, jid, metadata[jid], descriptions.get(jid)): jid
            for jid in job_ids
        }
        n_done, n_err = 0, 0
        for future in as_completed(futures):
            jid = futures[future]
            try:
                results.append(future.result())
                n_done += 1
                print(f"[{jid}] processed successfully.")
            except Exception as e:  
                n_err += 1
                print(f"[ERROR] {jid}: {e}", file=sys.stderr)
            if (n_done + n_err) % 25 == 0:
                print(f"  ...{n_done + n_err}/{len(job_ids)} (errors: {n_err})")

    if not results:
        print("No records summarized successfully; nothing written to the DB.")
        return

    collection.upsert(
        ids=[r["job_id"] for r in results],
        documents=[r["chunk_text"] for r in results],
        metadatas=[
            sanitize_metadata({k: v for k, v in r.items() if k not in ("job_id", "chunk_text")})
            for r in results
        ],
    )
    print(f"Upserted {len(results)} records into DB at '{DB_PATH}' (collection '{COLLECTION_NAME}').")

    if AUDIT_FILE:
        audit_path = Path(AUDIT_FILE)
        mode = "a" if audit_path.exists() else "w"
        with audit_path.open(mode, encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Audit copy written to {audit_path} (not used at query time).")

    print(f"Done. {n_done} succeeded, {n_err} failed.")
    if n_err:
        print("Re-run the script again to retry only the failed/missing IDs (resume is automatic).")


def run_test_query():
    """Prove out retrieval against the single DB: semantic search over
    chunk_text, optionally combined with a metadata filter, in one call."""
    collection = get_collection(DB_PATH, COLLECTION_NAME)

    where = None
    if TEST_FILTER:
        key, _, value = TEST_FILTER.partition("=")
        where = {key.strip(): value.strip()}

    results = collection.query(query_texts=[TEST_QUERY], n_results=5, where=where)

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not ids:
        print("No results.")
        return

    for job_id, doc, meta, dist in zip(ids, docs, metas, distances):
        print(f"\n[{job_id}]  (distance={dist:.4f})")
        print(f"  {meta.get('job_title')} @ {meta.get('company_name')} -- {meta.get('job_location')}")
        print(f"  {doc[:200].strip()}...")


def main():
    if TEST_QUERY:
        run_test_query()
    else:
        ingest()


if __name__ == "__main__":
    main()