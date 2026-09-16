"""Create the metadata index used to seed the ChromaDB collection.

This script reads the cleaned metadata workbook, keeps the job-identifying fields,
serializes them into JSONL, and upserts the minimal searchable record for each
job before the detailed description summaries are added later in the pipeline.
"""

import json
import os
import pandas as pd
import chromadb

INPUT_XLSX = r'D:\Assignment\RAG\Data\cleaned\metadata.xlsx'
OUTPUT_JSONL = r'D:\Assignment\RAG\Data\json\metadata.jsonl'


DB_PATH = "D:\Assignment\RAG\Data\chroma_db"
COLLECTION_NAME = "job_postings"

def get_collection(db_path: str, collection_name: str):
    """Return the persistent Chroma collection used for metadata seeding."""
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool -- no None/NaN.
    Drop or stringify anything else so upsert never fails on a missing or
    NaN field (pandas often leaves NaN for blank Excel cells)."""
    clean = {}
    for k, v in meta.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def build_minimal_doc(data: dict) -> str:
    """Placeholder searchable text for a row before the full description
    summary exists. Gets replaced by the real chunk_text once
    description_chunker.py processes this job_id."""
    return (
        f"Job Title: {data.get('Job Title', '')}\n"
        f"Company: {data.get('Company Name', '')}\n"
        f"Location: {data.get('Job Location', '')}\n"
        f"Level: {data.get('Job Level', '')}\n"
        f"Category: {data.get('Job Category', '')}"
    )


def convert_to_single_jsonl(df, output_filepath):
    """Write a JSONL file and upsert the placeholder metadata rows into Chroma.

    The placeholder documents are intentionally minimal so the indexing pipeline can
    resume safely before Groq-generated summaries are added.
    """

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    collection = get_collection(DB_PATH, COLLECTION_NAME)

    ids, docs, metadatas = [], [], []

    with open(output_filepath, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            data = {
                "ID": row['ID'],
                "Job Category": row['Job Category'],
                "Job Title": row['Job Title'],
                "Company Name": row['Company Name'],
                "Publication Date": row['Publication Date'],
                "Job Location": row['Job Location'],
                "Job Level": row['Job Level']
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


            job_id = str(data["ID"]).strip()
            ids.append(job_id)
            docs.append(build_minimal_doc(data))
            metadatas.append(sanitize_metadata({
                "job_title": data["Job Title"],
                "company_name": data["Company Name"],
                "job_category": data["Job Category"],
                "job_level": data["Job Level"],
                "job_location": data["Job Location"],
                "publication_date": data["Publication Date"],
            }))

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    print(f"Upserted {len(ids)} metadata records into DB at '{DB_PATH}' (collection '{COLLECTION_NAME}').")


df = pd.read_excel(INPUT_XLSX)

convert_to_single_jsonl(df, OUTPUT_JSONL)
print("Metadata JSONL file created successfully!")