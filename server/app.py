"""
FastAPI server exposing the NestTutorEnv via HTTP.
Provides OpenEnv-compliant endpoints: POST /reset, POST /step, GET /state.
Additional endpoints: GET /tasks, GET /health.

This module exists to satisfy OpenEnv "multi-mode deployment" validation,
which expects `server/app.py` and a `server` entry point in `pyproject.toml`.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.environment import NestTutorEnv

app = FastAPI(
    title="nest-tutor-eval-env",
    description=(
        "OpenEnv environment for evaluating AI tutor response quality. "
        "Built on the NEST.ai EdTech platform domain."
    ),
    version="1.0.0",
)

# In-memory session store (fine for single HF Space container)
_sessions: dict[str, NestTutorEnv] = {}


class ResetRequest(BaseModel):
    session_id: Optional[str] = "default"
    task_name: Optional[str] = "factual_accuracy"


class StepRequest(BaseModel):
    session_id: Optional[str] = "default"
    action: dict


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    task_name = req.task_name or "factual_accuracy"
    try:
        env = NestTutorEnv(task_name=task_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    session_id = req.session_id or "default"
    _sessions[session_id] = env
    return env.reset()


@app.post("/step")
def step(req: StepRequest):
    sid = req.session_id or "default"
    env = _sessions.get(sid)
    if env is None:
        raise HTTPException(status_code=400, detail=f"No active session '{sid}'. Call POST /reset first.")
    try:
        return env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(session_id: str = "default"):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail=f"No active session '{session_id}'. Call POST /reset first.")
    return env.state()


@app.get("/tasks")
def list_tasks():
    from app.tasks import TASKS

    return {
        name: {
            "difficulty": t.difficulty,
            "rubric": t.evaluation_rubric,
            "descriptions": t.rubric_descriptions,
            "max_steps": 5,
        }
        for name, t in TASKS.items()
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": "nest-tutor-eval-env", "version": "1.0.0"}


@app.get("/")
def root():
    return {
        "name": "nest-tutor-eval-env",
        "description": "OpenEnv environment for AI tutor evaluation — NEST.ai",
        "docs": "/docs",
        "tasks": "/tasks",
        "health": "/health",
    }


def main() -> None:
    """CLI entrypoint used by `python -m server.app` or `server` script."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()

