from typing import TypedDict, List, Dict, Any, Optional


# single state object passed through every node in the LangGraph graph
class ResearchState(TypedDict):
    query: str
    depth: str                          # "quick" or "detailed"
    domain: str                         # "finance", "tech", "healthcare", "general"
    subtasks: List[str]                 # planner output
    raw_results: List[Dict[str, Any]]   # gathered docs from researcher
    verified_results: List[Dict[str, Any]]  # filtered/scored docs from verifier
    report: str                         # final structured report from writer
    critique: Dict[str, Any]            # critic evaluation of the report
    evaluation: Dict[str, Any]          # metrics: latency, citation_score, factual_score, hallucination_score
    used_sources: List[Dict[str, Any]]  # sources actually injected into writer prompt
    followup_answer: str                # answer to a follow-up question via memory
    errors: List[str]                   # non-fatal errors accumulated across agents
    session_store: Optional[Any]        # SHORT_TERM VectorStore instance passed between researcher and writer


# default factory to initialize a blank state
def initial_state(query: str, depth: str = "quick", domain: str = "general") -> ResearchState:
    return ResearchState(
        query=query,
        depth=depth,
        domain=domain,
        subtasks=[],
        raw_results=[],
        verified_results=[],
        report="",
        critique={},
        evaluation={},
        used_sources=[],
        followup_answer="",
        errors=[],
        session_store=None,
    )