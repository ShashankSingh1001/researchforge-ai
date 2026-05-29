from typing import List, Dict, Any
from backend.agents.state import ResearchState
from backend.tools.search_tool import search, SearchError
from backend.tools.scraper import scrape, ScrapeError
from backend.tools.pdf_parser import parse_pdf, PDFParseError
from backend.memory.vector_store import VectorStore, VectorStoreError, SHORT_TERM

# number of search results to fetch per subtask based on depth
DEPTH_RESULT_COUNT = {"quick": 2, "detailed": 5}

# max characters to keep per page to avoid bloated state
MAX_CONTENT_LENGTH = 3000


def _is_pdf(url: str) -> bool:
    # checks URL suffix to decide if pdf_parser should be used
    return url.lower().endswith(".pdf")


def _is_remote_url(url: str) -> bool:
    # checks if URL is a remote HTTP/HTTPS address
    return url.lower().startswith("http://") or url.lower().startswith("https://")


def _fetch_url(url: str) -> str:
    # falls back to scraper for URLs where Tavily content is empty; remote PDFs use scraper
    if _is_pdf(url) and not _is_remote_url(url):
        chunks = parse_pdf(url)
        return " ".join(chunks)[:MAX_CONTENT_LENGTH]
    return scrape(url)[:MAX_CONTENT_LENGTH]


def _research_subtask(subtask: str, max_results: int) -> tuple[List[Dict[str, Any]], List[str]]:
    # searches for a single subtask; uses Tavily content directly, scraper as fallback
    results = []
    errors = []

    try:
        search_results = search(subtask, max_results=max_results)
    except SearchError as e:
        errors.append(f"Search failed for '{subtask}': {e}")
        return results, errors

    for item in search_results:
        url = item.get("url", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")

        # use Tavily-provided content directly — avoids scraping and 403 errors
        content = item.get("content", "").strip()

        if not url:
            continue

        # only fall back to scraper if Tavily returned no content
        if not content:
            try:
                content = _fetch_url(url)
            except (ScrapeError, PDFParseError) as e:
                errors.append(f"Fetch failed for '{url}': {e}")
                content = snippet

        content = content[:MAX_CONTENT_LENGTH]

        results.append({
            "subtask": subtask,
            "url": url,
            "title": title,
            "content": content,
            "source_type": "pdf" if _is_pdf(url) else "web",
        })

    return results, errors


def _embed_results_into_store(results: List[Dict[str, Any]], store: VectorStore, errors: List[str]) -> None:
    # embeds each result's content into SHORT_TERM vector store for RAG retrieval
    for result in results:
        content = result.get("content", "")
        if not content:
            continue
        source = f"{result.get('url', '')}|{result.get('title', '')}|{result.get('subtask', '')}"
        try:
            store.store(content, source=source, store_type=SHORT_TERM)
        except VectorStoreError as e:
            errors.append(f"Embedding failed for '{result.get('url', '')}': {e}")


def run(state: ResearchState) -> ResearchState:
    # entry point for the researcher node in the LangGraph graph
    subtasks = state["subtasks"]
    depth = state["depth"]
    max_results = DEPTH_RESULT_COUNT.get(depth, DEPTH_RESULT_COUNT["quick"])

    all_results = []
    all_errors = []

    # initialize a fresh SHORT_TERM store for this session
    store = VectorStore()

    for subtask in subtasks:
        results, errors = _research_subtask(subtask, max_results)
        all_results.extend(results)
        all_errors.extend(errors)

    # embed all gathered results into the session vector store
    _embed_results_into_store(all_results, store, all_errors)

    state["raw_results"] = all_results
    state["errors"].extend(all_errors)
    state["session_store"] = store

    return state