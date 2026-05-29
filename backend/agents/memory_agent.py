from backend.agents.graph import run_followup
from backend.agents.state import ResearchState


class MemoryAgentError(Exception):
    # raised when follow-up Q&A cannot be completed
    pass


def answer_followup(question: str, state: ResearchState) -> str:
    # answers a follow-up question using LONG_TERM memory from previous pipeline run
    if not question or not question.strip():
        raise MemoryAgentError("Follow-up question must be a non-empty string")

    original_query = state.get("query", "")
    answer = run_followup(question=question, original_query=original_query)
    return answer


def get_suggested_followups(state: ResearchState) -> list[str]:
    # generates suggested follow-up questions based on critic's missing insights
    critique = state.get("critique", {})
    missing_insights = critique.get("missing_insights", [])
    suggestions = critique.get("suggestions", [])

    followups = []

    # convert missing insights into follow-up questions
    for insight in missing_insights[:3]:
        followups.append(f"Can you elaborate on: {insight}?")

    # convert suggestions into actionable follow-up questions
    for suggestion in suggestions[:2]:
        followups.append(f"How would you {suggestion.lower()}?")

    return followups