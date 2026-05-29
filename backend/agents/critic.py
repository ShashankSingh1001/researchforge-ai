import json
from backend.agents.state import ResearchState
from backend.agents.llm_client import generate, LLMError

# max characters of report to send to critic to stay within token limits
MAX_REPORT_LENGTH = 2000


def _build_prompt(query: str, report: str, source_count: int) -> str:
    # constructs evaluation prompt asking LLM to critique the report
    truncated_report = report[:MAX_REPORT_LENGTH]

    return f"""You are a senior research critic. Evaluate the following research report.

Original Query: {query}
Sources Used: {source_count}

Report:
{truncated_report}

Evaluate the report on these dimensions and return ONLY a valid JSON object, no explanation, no markdown.

Format:
{{
    "quality_score": <integer 0-10>,
    "missing_insights": ["insight1", "insight2"],
    "weak_citations": ["citation issue 1", "citation issue 2"],
    "logical_gaps": ["gap1", "gap2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "verdict": "acceptable" or "needs_revision"
}}

Rules:
- quality_score: 0 is worst, 10 is best
- missing_insights: topics the report should have covered but did not
- weak_citations: sources that are low credibility or unverified claims
- logical_gaps: conclusions not supported by the provided evidence
- suggestions: specific actionable improvements
- verdict: "acceptable" if quality_score >= 6, else "needs_revision"

JSON:"""


def _parse_response(response: str) -> dict:
    # strips markdown fences and parses JSON critique from LLM response
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # return a default critique if LLM returns malformed JSON
        return {
            "quality_score": 0,
            "missing_insights": [],
            "weak_citations": [],
            "logical_gaps": [],
            "suggestions": ["Critique parsing failed, manual review recommended."],
            "verdict": "needs_revision",
        }


def run(state: ResearchState) -> ResearchState:
    # entry point for the critic node in the LangGraph graph
    report = state["report"]

    if not report:
        state["errors"].append("Critic error: no report to evaluate")
        state["critique"] = {}
        return state

    source_count = len(state["verified_results"])
    prompt = _build_prompt(state["query"], report, source_count)

    try:
        response = generate(prompt, max_new_tokens=512)
        critique = _parse_response(response)
    except LLMError as e:
        state["errors"].append(f"Critic error: {e}")
        critique = {}

    state["critique"] = critique
    return state