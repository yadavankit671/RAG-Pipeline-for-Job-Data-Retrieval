"""FastAPI application for the job-postings RAG demo.

The app exposes a query endpoint for retrieving and answering job-related
questions, and serves a lightweight HTML interface for browser-based testing.
"""

from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_engine import answer_question, TOP_K_DEFAULT

app = FastAPI(title="Job Postings RAG")


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K_DEFAULT
    job_level: Optional[str] = None
    job_location: Optional[str] = None
    job_category: Optional[str] = None


class Source(BaseModel):
    job_id: str
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    job_location: Optional[str] = None
    job_level: Optional[str] = None
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Answer a question using semantic retrieval plus optional metadata filters."""
    filters = {
        "job_level": req.job_level,
        "job_location": req.job_location,
        "job_category": req.job_category,
    }
    result = answer_question(req.question, top_k=req.top_k, filters=filters)
    return result


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Job Postings RAG</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  textarea, input, button, select { font: inherit; }
  textarea { width: 100%; min-height: 70px; padding: 8px; box-sizing: border-box; }
  .filters { display: flex; gap: 8px; margin: 8px 0; flex-wrap: wrap; }
  .filters input { flex: 1; min-width: 140px; padding: 6px; }
  button { padding: 8px 16px; margin-top: 8px; cursor: pointer; }
  #answer { white-space: pre-wrap; margin-top: 20px; padding: 12px; background: #f5f5f5; border-radius: 6px; min-height: 1.2em; }
  .source { border-top: 1px solid #ddd; padding: 8px 0; font-size: 0.9rem; }
  .source b { color: #333; }
  .muted { color: #777; font-size: 0.85rem; }
  #loading { display: none; color: #777; margin-top: 10px; }
</style>
</head>
<body>
  <h1>Job Postings RAG</h1>
  <textarea id="question" placeholder="Ask about the job postings, e.g. 'senior data analyst roles in healthcare'"></textarea>

  <div class="filters">
    <input id="job_level" placeholder="Job Level (optional, e.g. Senior Level)">
    <input id="job_location" placeholder="Job Location (optional)">
    <input id="job_category" placeholder="Job Category (optional)">
  </div>

  <button onclick="ask()">Ask</button>
  <div id="loading">Thinking...</div>

  <div id="answer"></div>
  <div id="sources"></div>

<script>
async function ask() {
  const question = document.getElementById('question').value.trim();
  if (!question) return;

  const body = {
    question,
    job_level: document.getElementById('job_level').value.trim() || null,
    job_location: document.getElementById('job_location').value.trim() || null,
    job_category: document.getElementById('job_category').value.trim() || null,
  };

  document.getElementById('answer').textContent = '';
  document.getElementById('sources').innerHTML = '';
  document.getElementById('loading').style.display = 'block';

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error('Request failed: ' + res.status);
    const data = await res.json();

    document.getElementById('answer').textContent = data.answer;

    const sourcesDiv = document.getElementById('sources');
    data.sources.forEach(s => {
      const div = document.createElement('div');
      div.className = 'source';
      div.innerHTML = `<b>[${s.job_id}]</b> ${s.job_title || ''} @ ${s.company_name || ''}
        <div class="muted">${s.job_location || ''} · ${s.job_level || ''} · distance: ${s.distance?.toFixed(3) ?? 'n/a'}</div>`;
      sourcesDiv.appendChild(div);
    });
  } catch (err) {
    document.getElementById('answer').textContent = 'Error: ' + err.message;
  } finally {
    document.getElementById('loading').style.display = 'none';
  }
}

document.getElementById('question').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the single-page browser interface for the RAG demo."""
    return INDEX_HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)