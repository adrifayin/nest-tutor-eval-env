"""
FastAPI server exposing the NestTutorEnv via HTTP.
Provides OpenEnv-compliant endpoints: POST /reset, POST /step, GET /state.
Additional endpoints: GET /tasks, GET /health.

Deployed on Hugging Face Spaces — port 7860.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.environment import NestTutorEnv

app = FastAPI(
    title="nest-tutor-eval-env",
    description=(
        "OpenEnv environment for evaluating AI tutor response quality. "
        "Built on the NEST.ai EdTech platform domain. "
        "Agents evaluate AI Co-Tutor responses across factual accuracy, "
        "pedagogical quality, and personalisation dimensions."
    ),
    version="1.0.0",
)

# In-memory session store (fine for single HF Space container)
_sessions: dict[str, NestTutorEnv] = {}


# ── Request schemas ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    session_id: Optional[str] = "default"
    task_name: Optional[str] = "factual_accuracy"


class StepRequest(BaseModel):
    session_id: Optional[str] = "default"
    action: dict


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """
    Reset the environment for a given task.
    Creates (or replaces) a session and returns the initial observation.
    """
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
    """
    Submit one evaluation action and receive the next observation + reward.
    Requires an active session (call /reset first).
    """
    sid = req.session_id or "default"
    env = _sessions.get(sid)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session '{sid}'. Call POST /reset first.",
        )
    try:
        return env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(session_id: str = "default"):
    """Return the full current state of an active session."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session '{session_id}'. Call POST /reset first.",
        )
    return env.state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with their rubric and difficulty."""
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
    """Health check — required by Docker HEALTHCHECK and HF Spaces ping."""
    return {"status": "ok", "env": "nest-tutor-eval-env", "version": "1.0.0"}


@app.get("/")
def root():
    """Root redirect to API docs."""
    return {
        "name": "nest-tutor-eval-env",
        "description": "OpenEnv environment for AI tutor evaluation — NEST.ai",
        "docs": "/docs",
        "tasks": "/tasks",
        "health": "/health",
    }
