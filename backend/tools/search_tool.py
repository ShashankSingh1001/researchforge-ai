import os
from tavily import TavilyClient
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()


class SearchError(Exception):
    # raised when Tavily search fails
    pass


def _get_client() -> TavilyClient:
    # initializes Tavily client with API key from environment
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise SearchError("TAVILY_API_KEY environment variable is not set")
    return TavilyClient(api_key=api_key)


def search(query: str, max_results: int = 10) -> list[dict]:
    # returns list of {title, url, snippet, content} dicts for the given query
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    try:
        client = _get_client()
        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=False,
        )
    except Exception as e:
        raise SearchError(f"Tavily search failed: {e}") from e

    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", "").strip(),
            "url": item.get("url", "").strip(),
            "snippet": item.get("content", "").strip(),
            "content": item.get("content", "").strip(),
        })

    return results