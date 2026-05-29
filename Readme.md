# Autonomous Multi-Agent Research Analyst

A production-grade autonomous research system built with LangGraph multi-agent orchestration, RAG (FAISS + Sentence Transformers), real-time web search, and persistent memory. Given a research query, the system plans, searches, verifies, writes, critiques, and evaluates a structured report — fully autonomously.

---

## What It Does

**Input**
```
Analyze impact of LLMs on Indian fintech startups with data and trends
```

**Output**
- Structured 6-section research report with citations
- Verified sources with credibility scores
- Contradiction detection across sources
- Factual, citation, and hallucination evaluation metrics
- Follow-up Q&A answered from persistent memory
- Downloadable PDF report

---

## Pipeline Architecture

```
User Query
    |
    v
Planner Agent        -- breaks query into subtasks, sets depth and domain
    |
    v
Researcher Agent     -- Tavily search + HTML scraper + PDF parser per subtask
    |
    v
Verifier Agent       -- deduplication, domain credibility scoring, contradiction detection
    |
    v
Writer Agent         -- RAG-grounded 6-section report with cited sources (FAISS retrieval)
    |
    v
Critic Agent         -- quality score, missing insights, logical gaps, verdict
    |
    v
Evaluator Agent      -- factual score, citation score, hallucination score, latency
    |
    v
Memory + Report      -- LONG_TERM FAISS persistence, follow-up Q&A, PDF export
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | llama-3.1-8b-instant via Groq (free tier) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Vector DB | FAISS (Flat L2) |
| Web Search | Tavily API (free tier, 1000 req/month) |
| Web Scraping | requests + BeautifulSoup4 |
| PDF Extraction | PyMuPDF |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF Export | reportlab |
| Python | 3.10.x |
| Package Manager | uv |

---

## Project Structure

```
research_agent/
|
|-- backend/
|   |-- main.py                  # FastAPI app entry point, CORS, router
|   |-- agents/
|   |   |-- state.py             # ResearchState TypedDict, initial_state()
|   |   |-- llm_client.py        # Groq API wrapper, exponential backoff
|   |   |-- planner.py           # subtask decomposition
|   |   |-- researcher.py        # tool orchestration, RAG embedding
|   |   |-- verifier.py          # dedup, credibility scoring, contradictions
|   |   |-- writer.py            # RAG retrieval, report generation
|   |   |-- critic.py            # report quality evaluation
|   |   |-- evaluator.py         # factual, citation, hallucination metrics
|   |   |-- memory_agent.py      # follow-up Q&A wrapper
|   |   |-- graph.py             # LangGraph pipeline, LONG_TERM persistence
|   |-- tools/
|   |   |-- search_tool.py       # Tavily search
|   |   |-- scraper.py           # static HTML fetch and clean
|   |   |-- pdf_parser.py        # PyMuPDF chunked extraction
|   |-- rag/
|   |   |-- embedder.py          # sentence-transformers embedding
|   |   |-- retriever.py         # FAISS index add, search, save, load
|   |-- memory/
|   |   |-- vector_store.py      # SHORT_TERM + LONG_TERM VectorStore
|   |-- api/
|       |-- routes.py            # POST /research, POST /followup, GET /health
|
|-- frontend/
|   |-- app.py                   # Streamlit UI (5 pages)
|
|-- tests/
|   |-- test_tools.py            # 15 tests
|   |-- test_rag.py              # 19 tests
|   |-- test_agents.py           # 25 tests
|
|-- requirements.txt
|-- .env                         # not committed — see Environment Variables
|-- .gitignore
|-- README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Liveness probe |
| POST | `/api/research` | Run full pipeline, returns report + sources + evaluation + critique |
| POST | `/api/followup` | Answer follow-up question from LONG_TERM memory |

**POST /api/research — request body**
```json
{
  "query": "Impact of LLMs on Indian fintech startups",
  "depth": "quick",
  "domain": "finance"
}
```

`depth` accepts `quick` (2 results per subtask) or `detailed` (5 results per subtask).
`domain` accepts `general`, `finance`, `tech`, `healthcare`.

---

## Report Output Format

Every report follows this strict 6-section structure:

```
1. Overview
2. Key Insights
3. Data & Statistics
4. Contradictions
5. Sources
6. Conclusion
```

---

## Evaluation Metrics

| Metric | Source | Range | Description |
|---|---|---|---|
| `factual_score` | LLM (Groq) | 0 – 10 | Claim accuracy against used sources |
| `citation_score` | URL matching | 0.0 – 1.0 | Fraction of used source URLs present in report |
| `hallucination_score` | LLM (Groq) | 0.0 – 1.0 | Direct contradictions only; 0.0 is best |
| `latency` | time.time() | seconds | Full pipeline wall-clock time |
| `quality_score` | Critic agent | 0 – 10 | Overall report quality |
| `verdict` | Critic agent | acceptable / needs_revision | quality_score >= 6 = acceptable |

---

## Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/ShashankSingh1001/researchforge-ai.git
cd research-analyst
```

**2. Create and activate a virtual environment**
```bash
uv venv
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
uv pip install -r requirements.txt
```

**4. Create a `.env` file in the project root**
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
BACKEND_URL=http://localhost:8000
```

Get your keys here:
- Groq: https://console.groq.com
- Tavily: https://app.tavily.com

**5. Run the backend**
```bash
uvicorn backend.main:app --reload
```

**6. Run the frontend** (in a separate terminal)
```bash
streamlit run frontend/app.py
```

**7. Run all tests**
```bash
pytest tests/ -v
```

All 59 tests run fully offline with mocked LLM and search calls.

---

## Streamlit UI Pages

| Page | Description |
|---|---|
| Overview | Pipeline architecture, module map, output schema, quick-start guide |
| Research | Query input, depth/domain config, 7-step agent progress stepper, report display, PDF export |
| Sources | Verified sources with credibility score pills, snippets |
| Evaluation | Metric cards (factual, citation, hallucination, latency), critic analysis, raw JSON debug |
| Follow-up Q&A | Suggested questions from critic, free-form Q&A answered from LONG_TERM memory |

---

## Deployment

**Backend — Render**

1. Push repo to GitHub
2. Create a new Web Service on Render pointing to the repo
3. Set environment variables: `GROQ_API_KEY`, `TAVILY_API_KEY`
4. Start command:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 10000
```

**Frontend — Streamlit Cloud**

1. Connect repo at https://share.streamlit.io
2. Set main file path: `frontend/app.py`
3. Set environment variable: `BACKEND_URL=https://your-render-service.onrender.com`

---

## Memory Design

The system maintains two stores backed by FAISS:

- `SHORT_TERM` — in-memory, session-scoped. Created by the Researcher and consumed by the Writer for RAG retrieval within a single pipeline run.
- `LONG_TERM` — persisted to disk at `backend/memory/store/`. Populated after every pipeline run with the query, report, and verified source content. Used by the Follow-up Q&A system across sessions.

---

## Resume Entry

> Built a production-grade autonomous research agent using LangGraph with multi-agent orchestration (planner, researcher, verifier, writer, critic), integrating RAG (FAISS + Sentence Transformers), real-time web search, and memory. Enabled structured report generation with citations, contradiction detection, and evaluation metrics, deployed via FastAPI and Streamlit.

---

## License

MIT