import json
from backend.agents.state import ResearchState
from backend.agents.llm_client import generate, LLMError

# maximum subtasks allowed to keep research focused
MAX_SUBTASKS = 6


def _build_prompt(query: str, depth: str, domain: str) -> str:
    # constructs a strict JSON-output prompt for the planner
    return f"""You are a research planning assistant. Break the following query into subtasks.

Query: {query}
Depth: {depth}
Domain: {domain}

Rules:
- Return ONLY a valid JSON object, no explanation, no markdown
- Format: {{"subtasks": ["subtask1", "subtask2", ...], "depth": "{depth}", "domain": "{domain}"}}
- Maximum {MAX_SUBTASKS} subtasks
- Each subtask must be a specific, actionable research question
- For depth "quick": 2-3 subtasks
- For depth "detailed": 4-6 subtasks

JSON:"""


def _parse_response(response: str) -> dict:
    # strips markdown fences if LLM wraps output in code blocks
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner LLM returned invalid JSON: {e}\nRaw: {response}")

    if "subtasks" not in data or not isinstance(data["subtasks"], list):
        raise ValueError(f"Planner response missing 'subtasks' list: {data}")

    # enforce maximum subtask limit
    data["subtasks"] = data["subtasks"][:MAX_SUBTASKS]
    return data


def run(state: ResearchState) -> ResearchState:
    # entry point for the planner node in the LangGraph graph
    query = state["query"]
    depth = state["depth"]
    domain = state["domain"]

    prompt = _build_prompt(query, depth, domain)

    try:
        response = generate(prompt, max_new_tokens=512)
        parsed = _parse_response(response)
        state["subtasks"] = parsed["subtasks"]
    except (LLMError, ValueError) as e:
        # log error non-fatally and set a single fallback subtask
        state["errors"].append(f"Planner error: {e}")
        state["subtasks"] = [query]

    return state