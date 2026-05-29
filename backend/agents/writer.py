from typing import List, Dict, Any, Optional
from backend.agents.state import ResearchState
from backend.agents.llm_client import generate, LLMError
from backend.memory.vector_store import VectorStore, VectorStoreError, SHORT_TERM

# max characters of content to inject per source into the prompt
MAX_CONTENT_PER_SOURCE = 500

# number of top sources to retrieve from vector store by similarity
MAX_SOURCES_IN_PROMPT = 5


def _retrieve_relevant_chunks(query: str, store: VectorStore) -> List[Dict[str, Any]]:
    # queries SHORT_TERM store for most semantically relevant chunks to the query
    try:
        return store.query(query, top_k=MAX_SOURCES_IN_PROMPT, store_type=SHORT_TERM)
    except VectorStoreError:
        return []


def _parse_source_string(source_str: str) -> tuple[str, str]:
    # parses url and title from pipe-delimited source string stored during embedding
    parts = source_str.split("|")
    url = parts[0] if len(parts) > 0 else ""
    title = parts[1] if len(parts) > 1 else ""
    return url, title


def _format_rag_sources(chunks: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    # formats RAG chunks into prompt block; returns formatted text and used source dicts
    lines = []
    used = []
    for i, chunk in enumerate(chunks, 1):
        source_str = chunk.get("source", "")
        url, title = _parse_source_string(source_str)
        text = chunk.get("text", "")[:MAX_CONTENT_PER_SOURCE]
        lines.append(f"Source {i}: {title}\nURL: {url}\n{text}\n")
        used.append({"url": url, "title": title, "content": text})
    return "\n".join(lines), used


def _format_fallback_sources(results: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    # formats verified results as fallback; returns formatted text and used source dicts
    lines = []
    used = []
    for i, r in enumerate(results[:MAX_SOURCES_IN_PROMPT], 1):
        content = r.get("content", "")[:MAX_CONTENT_PER_SOURCE]
        title = r.get("title", "")
        url = r.get("url", "")
        lines.append(f"Source {i}: {title}\nURL: {url}\n{content}\n")
        used.append({"url": url, "title": title, "content": content})
    return "\n".join(lines), used


def _format_contradictions(results: List[Dict[str, Any]]) -> str:
    # extracts contradiction list from first verified result that has them
    for r in results:
        contradictions = r.get("contradictions", [])
        if contradictions:
            return "\n".join(f"- {c}" for c in contradictions)
    return "No contradictions detected across sources."


def _build_prompt(query: str, domain: str, depth: str, sources_block: str, contradictions_block: str) -> str:
    # constructs structured report generation prompt
    detail_instruction = (
        "Be concise, 2-3 sentences per section."
        if depth == "quick"
        else "Be thorough, 4-6 sentences per section with specific data points."
    )

    return f"""You are a professional research analyst specializing in {domain}.

Write a structured research report for the following query:
Query: {query}

Use only the provided sources. {detail_instruction}
IMPORTANT: In section 5, you MUST use the exact URLs provided above. Do NOT invent or modify URLs.

Sources:
{sources_block}

Contradictions:
{contradictions_block}

Write the report in exactly this format:

## 1. Overview
[overview here]

## 2. Key Insights
[key insights here]

## 3. Data and Statistics
[data and statistics here]

## 4. Contradictions
[contradictions here]

## 5. Sources
[list each source as: - Title | URL, using exact URLs from the Sources block above]

## 6. Conclusion
[conclusion here]

Report:"""


def run(state: ResearchState) -> ResearchState:
    # entry point for the writer node in the LangGraph graph
    verified = state["verified_results"]

    if not verified:
        state["errors"].append("Writer error: no verified results to write report from")
        state["report"] = ""
        return state

    # retrieve semantically relevant chunks from SHORT_TERM store if available
    store: Optional[VectorStore] = state.get("session_store")
    rag_chunks = _retrieve_relevant_chunks(state["query"], store) if store else []

    # use RAG chunks if available, fall back to verified results
    if rag_chunks:
        sources_block, used_sources = _format_rag_sources(rag_chunks)
    else:
        sources_block, used_sources = _format_fallback_sources(verified)

    # store used sources in state for citation scoring
    state["used_sources"] = used_sources

    contradictions_block = _format_contradictions(verified)

    prompt = _build_prompt(
        query=state["query"],
        domain=state["domain"],
        depth=state["depth"],
        sources_block=sources_block,
        contradictions_block=contradictions_block,
    )

    try:
        report = generate(prompt, max_new_tokens=1024)
        state["report"] = report
    except LLMError as e:
        state["errors"].append(f"Writer error: {e}")
        state["report"] = ""

    return state