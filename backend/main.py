import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

# configure root logger once at app startup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Autonomous Research Analyst API",
    description="Multi-agent research pipeline with RAG, verification, and memory.",
    version="1.0.0",
)

# allow all origins; suitable for dev and portfolio deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register all API routes under /api prefix
app.include_router(router, prefix="/api")