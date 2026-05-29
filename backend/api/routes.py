import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.agents.graph import run_pipeline, run_followup
from backend.agents.memory_agent import get_suggested_followups

logger = logging.getLogger(__name__)

router = APIRouter()

# --- request / response models ---

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Research query to investigate")
    depth: str = Field("quick", description="Research depth: quick or detailed")
    domain: str = Field("general", description="Domain context: finance, tech, healthcare, general")

class ResearchResponse(BaseModel):
    query: str
    report: str
    sources: list[dict]
    evaluation: dict
    critique: dict
    suggested_followups: list[str]
    errors: list[str]

class FollowupRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Follow-up question")
    original_query: str = Field(..., min_length=5, description="Original research query for memory context")

class FollowupResponse(BaseModel):
    question: str
    answer: str
    errors: list[str]

class HealthResponse(BaseModel):
    status: str
    version: str

# --- endpoint definitions ---

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check():
    # simple liveness probe
    return HealthResponse(status="ok", version="1.0.0")

@router.post("/research", response_model=ResearchResponse, tags=["research"])
def run_research(request: ResearchRequest):
    # validate depth value before hitting the pipeline
    if request.depth not in ("quick", "detailed"):
        raise HTTPException(status_code=422, detail="depth must be quick or detailed")

    try:
        state = run_pipeline(
            query=request.query,
            depth=request.depth,
            domain=request.domain,
        )
    except Exception as exc:
        logger.exception("Pipeline failure for query: %s", request.query)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    # extract trimmed fields the frontend needs
    sources = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "credibility_score": r.get("credibility_score", 0),
            "snippet": r.get("content", "")[:300],
        }
        for r in state.get("verified_results", [])
    ]

    suggested = get_suggested_followups(state)

    return ResearchResponse(
        query=state.get("query", request.query),
        report=state.get("report", ""),
        sources=sources,
        evaluation=state.get("evaluation", {}),
        critique=state.get("critique", {}),
        suggested_followups=suggested,
        errors=state.get("errors", []),
    )

@router.post("/followup", response_model=FollowupResponse, tags=["research"])
def run_followup_endpoint(request: FollowupRequest):
    # answer a follow-up question using persisted LONG_TERM memory
    try:
        answer = run_followup(
            question=request.question,
            original_query=request.original_query,
        )
    except Exception as exc:
        logger.exception("Followup failure for question: %s", request.question)
        raise HTTPException(status_code=500, detail=f"Followup error: {str(exc)}")

    return FollowupResponse(
        question=request.question,
        answer=answer,
        errors=[],
    )