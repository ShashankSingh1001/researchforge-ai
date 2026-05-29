import json
import re
from typing import List, Dict, Any
from backend.agents.state import ResearchState
from backend.agents.llm_client import generate, LLMError

# max chars of report sent to LLM for scoring
MAX_REPORT_FOR_EVAL = 1500

# max chars of source content injected per source for LLM evaluation
MAX_SOURCE_FOR_EVAL = 600

# number of top sources used for LLM-based evaluation
MAX_SOURCES_FOR_EVAL = 6


def _extract_sources_section(report: str) -> str:
    # extracts the Sources section from the report for citation checking
    marker = "## 5. Sources"
    if marker in report:
        return report.split(marker)[1].split("## 6.")[0]
    return report


def _compute_citation_score(report: str, used_sources: List[Dict[str, Any]]) -> float:
    # checks what fraction of actually-used source URLs appear in the report
    if not used_sources or not report:
        return 0.0
    sources_section = _extract_sources_section(report)
    matched = 0
    for r in used_sources:
        url = r.get("url", "")
        if not url:
            continue
        if url in sources_section or url in report:
            matched += 1
            continue
        # partial credit for domain name match when full URL is absent
        from urllib.parse import urlparse
        domain = urlparse(url).hostname or ""
        if domain and domain in sources_section:
            matched += 1
    return round(matched / len(used_sources), 2)


def _build_factual_prompt(report: str, sources: List[Dict[str, Any]]) -> str:
    # builds LLM prompt to rate report factual accuracy against used sources
    sources_block = ""
    for i, s in enumerate(sources[:MAX_SOURCES_FOR_EVAL], 1):
        content = s.get("content", "")[:MAX_SOURCE_FOR_EVAL]
        sources_block += f"Source {i} ({s.get('url', '')}):\n{content}\n\n"

    return f"""You are a fact-checking assistant. Rate the factual accuracy of the following report against the provided sources.

Sources:
{sources_block}

Report:
{report[:MAX_REPORT_FOR_EVAL]}

Return ONLY a valid JSON object, no explanation, no markdown.
Format: {{"factual_score": <integer 0-10>, "reasoning": "<one sentence>"}}

Rules:
- factual_score 10: all claims fully supported by sources
- factual_score 5: some claims supported, some unverified
- factual_score 0: claims contradict or are absent from sources

JSON:"""


def _build_hallucination_prompt(report: str, sources: List[Dict[str, Any]]) -> str:
    # builds LLM prompt to detect claims that directly contradict sources
    sources_block = ""
    for i, s in enumerate(sources[:MAX_SOURCES_FOR_EVAL], 1):
        content = s.get("content", "")[:MAX_SOURCE_FOR_EVAL]
        sources_block += f"Source {i} ({s.get('url', '')}):\n{content}\n\n"

    return f"""You are a hallucination detection assistant. Identify claims in the report that directly contradict the provided sources.

Sources:
{sources_block}

Report:
{report[:MAX_REPORT_FOR_EVAL]}

Return ONLY a valid JSON object, no explanation, no markdown.
Format: {{"hallucination_score": <float 0.0-1.0>, "hallucinated_claims": ["claim1", "claim2"]}}

Rules:
- Only flag claims that DIRECTLY CONTRADICT the sources, not reasonable inferences or general knowledge
- Do NOT flag claims just because they are not explicitly mentioned in sources
- hallucination_score 0.0: no direct contradictions found
- hallucination_score 0.5: some claims contradict sources
- hallucination_score 1.0: most claims contradict sources

JSON:"""


def _parse_llm_json(response: str, fallback: dict) -> dict:
    # strips markdown fences and parses JSON; falls back to brace matching then regex
    cleaned = response.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            if part.strip().startswith("{"):
                cleaned = part.strip()
                break
    # find first complete JSON object using brace matching
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # last resort: extract numeric score with regex
        result = dict(fallback)
        score_match = re.search(
            r"""[\"']?(?:factual_score|hallucination_score|quality_score)[\"']?\s*:\s*([0-9.]+)""",
            cleaned
        )
        if score_match:
            key = list(fallback.keys())[0]
            try:
                val = score_match.group(1)
                result[key] = float(val) if "." in val else int(val)
            except ValueError:
                pass
        return result


def _compute_factual_score(report: str, used_sources: List[Dict[str, Any]], errors: List[str]) -> tuple[int, str]:
    # calls LLM to rate factual accuracy against used sources
    if not report or not used_sources:
        return 0, "No report or sources available for evaluation."
    prompt = _build_factual_prompt(report, used_sources)
    try:
        response = generate(prompt, max_new_tokens=256)
        result = _parse_llm_json(response, {"factual_score": 0, "reasoning": "Parse failed."})
        return result.get("factual_score", 0), result.get("reasoning", "")
    except LLMError as e:
        errors.append(f"Evaluator factual_score error: {e}")
        return 0, "LLM unavailable."


def _compute_hallucination_score(report: str, used_sources: List[Dict[str, Any]], errors: List[str]) -> tuple[float, List[str]]:
    # calls LLM to detect claims that directly contradict used sources
    if not report or not used_sources:
        return 0.0, []
    prompt = _build_hallucination_prompt(report, used_sources)
    try:
        response = generate(prompt, max_new_tokens=256)
        result = _parse_llm_json(response, {"hallucination_score": 0.0, "hallucinated_claims": []})
        return result.get("hallucination_score", 0.0), result.get("hallucinated_claims", [])
    except LLMError as e:
        errors.append(f"Evaluator hallucination_score error: {e}")
        return 0.0, []


def run(state: ResearchState) -> ResearchState:
    # entry point for the evaluator node in the LangGraph graph
    report = state["report"]
    used_sources = state.get("used_sources", [])
    errors = []

    # citation score uses only writer-injected sources, not all verified results
    citation_score = _compute_citation_score(report, used_sources)

    # factual and hallucination scores use the same used sources for consistency
    factual_score, factual_reasoning = _compute_factual_score(report, used_sources, errors)
    hallucination_score, hallucinated_claims = _compute_hallucination_score(report, used_sources, errors)

    # latency is measured in graph.py and injected before this node runs
    latency = state.get("evaluation", {}).get("latency", 0.0)

    state["evaluation"] = {
        "latency": latency,
        "citation_score": citation_score,
        "factual_score": factual_score,
        "factual_reasoning": factual_reasoning,
        "hallucination_score": hallucination_score,
        "hallucinated_claims": hallucinated_claims,
    }

    state["errors"].extend(errors)
    return state