# Project Overview

## Dataset

[Open the source spreadsheet](https://docs.google.com/spreadsheets/d/1Sv88JfzxJlLrwpQDbwZpK7Jl_iwGzcbnvZqsEbdfMtE/edit?usp=sharing)

| Metric | Value |
| --- | ---: |
| Total jobs | 1,000 |
| Unique companies | 145 |
| Unique job categories | 7 |
| Columns | 9 |
| Rows | 1,000 |

### Dataset Fields

| Field | Description |
| --- | --- |
| `ID` | Unique identifier for each job listing, formatted as `LF####`. |
| `Job Category` | Job category classification. |
| `Job Title` | Job title or position name. |
| `Company Name` | Name of the hiring company. |
| `Publication Date` | Date the job was posted, in ISO 8601 format. |
| `Job Location` | Geographic location where the job is based. |
| `Job Level` | Required experience level, such as Mid Level, Senior Level, or Internship. |
| `Tags` | Additional tags associated with the job. |
| `Job Description` | Full job description, including responsibilities, requirements, and benefits. The content is HTML formatted. |

## Requirements

### 1. RAG Pipeline Components

- **Preprocessing:** Clean and chunk job descriptions.
- **Embeddings:** Generate embeddings using a service or model such as Gemini, Cohere, or Hugging Face.
- **Vector store:** Store embeddings in a vector database.
- **Retriever:** Fetch the top-*N* relevant chunks for each query.
- **LLM integration:** Combine the query with retrieved context to generate enriched responses.

### 2. API Endpoint

- **Tech stack:** Python, with FastAPI preferred.
- **Endpoint:** `POST /api/query`

## Expectations

1. Build a fully functional RAG pipeline that returns relevant job listings for user queries.
2. Keep the code modular, well organized, and clearly documented.
3. Design prompts that produce accurate and concise LLM responses.

## Optional Enhancements

1. **Hybrid search:** Combine vector and keyword search for improved precision.
2. **Reranker model:** Use a cross-encoder or another reranking approach to improve result ordering.

## Documentation

Include a Google Docs report covering:

1. High-level architecture, engineering decisions, and the reasoning behind each decision.
2. Setup and installation instructions.
3. Example requests and expected responses.
4. Assumptions made during development.
5. Drawbacks and potential future enhancements.

## Submission

Upload the complete codebase and all related files, including the documentation report, to a GitHub repository. Share the repository link upon completion.