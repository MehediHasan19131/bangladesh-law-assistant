"""FastAPI service: JSON API + a small web UI.

    uvicorn lawrag.api:app --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .index import HybridIndex
from .llm import get_llm
from .rag import Assistant

STATIC = Path(__file__).resolve().parents[2] / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = HybridIndex.load(Path(os.environ.get("INDEX_DIR", "index")))
    state["assistant"] = Assistant(index, get_llm())
    yield
    state.clear()


app = FastAPI(title="Bangladesh Law Assistant", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(6, ge=1, le=12)
    mode: str = Field("dense", pattern="^(dense|hybrid|bm25)$")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    assistant: Assistant = state["assistant"]
    return {
        "status": "ok",
        "chunks": len(assistant.index.chunks),
        "embedder": assistant.index.embedder_name,
        "llm": f"{assistant.llm.name}/{getattr(assistant.llm, 'model', '')}",
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    try:
        return state["assistant"].ask(req.question.strip(), k=req.k, mode=req.mode).to_dict()
    except Exception as exc:  # surface LLM/provider failures as a clean 502 instead of a stack trace
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {exc}") from exc


@app.get("/api/search")
def search(q: str = Query(min_length=2), k: int = Query(6, ge=1, le=20), mode: str = "dense") -> list[dict]:
    if mode not in ("hybrid", "bm25", "dense"):
        raise HTTPException(status_code=422, detail="mode must be hybrid, bm25 or dense")
    hits = state["assistant"].index.search(q, k=k, mode=mode)
    return [{"id": c.id, "label": c.label, "title": c.title, "score": round(s, 4), "text": c.text} for c, s in hits]
