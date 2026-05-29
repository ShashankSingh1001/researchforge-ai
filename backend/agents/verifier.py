from urllib.parse import urlparse
from typing import List, Dict, Any
from backend.agents.state import ResearchState
from backend.agents.llm_client import generate, LLMError

# domain tier scoring for source credibility
DOMAIN_SCORES = {
    ".gov": 1.0,
    ".edu": 0.9,
    ".org": 0.7,
    ".com": 0.5,
}
DEFAULT_DOMAIN_SCORE = 0.4

# minimum content length to consider a result valid
MIN_CONTENT_LENGTH = 100

# number of top results to send to LLM for contradiction detection
CONTRADICTION_SAMPLE_SIZE = 4


def _score_domain(url: str) -> float:
    # assigns credibility score based on top-level domain
    try:
        hostname = urlparse(url).hostname or ""
        for tld, score in DOMAIN_SCORES.items():
            if hostname.endswith(tld):
                return score
    except Exception:
        pass
    return DEFAULT_DOMAIN_SCORE


def _deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # removes duplicate results by URL, keeps first occurrence
    seen = set()
    unique = []
    for result in results:
        url = result.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(result)
    return unique


def _filter_short_content(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # removes results with content too short to be useful
    return [r for r in results if len(r.get("content", "")) >= MIN_CONTENT_LENGTH]


def _score_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # adds credibility_score field to each result and sorts by score descending
    for result in results:
        result["credibility_score"] = _score_domain(result.get("url", ""))
    return sorted(results, key=lambda x: x["credibility_score"], reverse=True)


def _build_contradiction_prompt(results: List[Dict[str, Any]]) -> str:
    # builds prompt asking LLM to identify contradictions across top sources
    sources = ""
    for i, r in enumerate(results[:CONTRADICTION_SAMPLE_SIZE], 1):
        snippet = r.get("content", "")[:300]
        sources += f"\nSource {i} ({r.get('url', '')}):\n{snippet}\n"

    return f"""You are a research verifier. Analyze these sources and identify any contradictions.

{sources}

Rules:
- Return ONLY a valid JSON object, no explanation, no markdown
- Format: {{"contradictions": ["contradiction1", "contradiction2"], "has_contradictions": true/false}}
- If no contradictions found, return {{"contradictions": [], "has_contradictions": false}}

JSON:"""


def _detect_contradictions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # uses LLM to find contradictions across top results, returns structured finding
    if len(results) < 2:
        return {"contradictions": [], "has_contradictions": False}

    prompt = _build_contradiction_prompt(results)

    try:
        response = generate(prompt, max_new_tokens=256)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        import json
        return json.loads(cleaned)
    except (LLMError, Exception):
        # non-fatal: skip contradiction detection if LLM fails
        return {"contradictions": [], "has_contradictions": False}


def run(state: ResearchState) -> ResearchState:
    # entry point for the verifier node in the LangGraph graph
    raw = state["raw_results"]

    # apply filtering and scoring pipeline
    deduped = _deduplicate(raw)
    filtered = _filter_short_content(deduped)
    scored = _score_results(filtered)

    # detect contradictions across top scored sources
    contradiction_result = _detect_contradictions(scored)

    # attach contradiction flag to each result for writer reference
    has_contradictions = contradiction_result.get("has_contradictions", False)
    contradictions = contradiction_result.get("contradictions", [])

    for result in scored:
        result["has_contradictions"] = has_contradictions
        result["contradictions"] = contradictions

    state["verified_results"] = scored
    return state