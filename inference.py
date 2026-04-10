"""
NEST.ai Tutor Evaluation — OpenEnv Baseline Inference Script
============================================================
Baseline agent that evaluates AI tutor responses using an LLM.
Runs all 3 tasks (easy → medium → hard) sequentially via HTTP.

Mandatory stdout format (machine-parsed by judges):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL  — LLM API base URL (default: HF inference router)
  MODEL_NAME    — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      — API key (required)
  ENV_URL       — Environment server URL (default: http://localhost:7860)
"""
import asyncio
import json
import os
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str = os.getenv("HF_TOKEN")  # required — no default

# Optional — only needed when using from_docker_image()
LOCAL_IMAGE_NAME: str = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS: int = 5
SUCCESS_THRESHOLD: float = 0.5
BENCHMARK: str = "nest-tutor-eval-env"
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 600

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert educational reviewer evaluating AI tutor responses.

    You will receive:
    - A student's profile (level, subject, learning style, known mistakes)
    - The student's question
    - An AI Co-Tutor's response to that question
    - A rubric with specific criteria to score

    Your task: evaluate the tutor's response against every rubric criterion.

    RESPOND ONLY with a valid JSON object in exactly this format:
    {
        "rubric_scores": {
            "<criterion_name>": <float 0.0 to 1.0>,
            ...
        },
        "overall_quality": <float 0.0 to 1.0>,
        "improvement_suggestion": "<specific suggestion or null>",
        "flag_for_human_review": false
    }

    Scoring guide:
    - 0.0–0.2: Very poor / absent
    - 0.3–0.5: Below expectations
    - 0.6–0.7: Acceptable
    - 0.8–0.9: Good
    - 1.0:     Excellent

    Be critical. A response can be factually correct but still score low
    on depth or personalisation if it doesn't match the student's level.
    Consider the student's learning style and known mistakes carefully.

    No preamble. No markdown fences. Just the JSON.
""").strip()


# ── Stdout loggers (mandatory format — do NOT modify) ─────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line — one per episode."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit [STEP] line — one per env.step() call."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitise action: no newlines allowed on a single log line
    action_str = str(action).replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    """Emit [END] line — always emitted, even on exception."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM evaluation call ───────────────────────────────────────────────────

def get_evaluation(
    client: OpenAI,
    observation: dict,
    history: List[str],
) -> dict:
    """
    Call the LLM to produce a NestAction evaluation dict.
    Falls back to neutral mid-range scores on any failure.
    """
    session = observation.get("session", {})
    rubric = observation.get("evaluation_rubric", [])
    descriptions = observation.get("rubric_descriptions", {})
    step_num = observation.get("step_number", 0)

    rubric_block = "\n".join(
        f"  - {item}: {descriptions.get(item, '')}" for item in rubric
    )
    history_block = "\n".join(history[-3:]) if history else "None"

    student = session.get("student_profile", {})
    user_msg = textwrap.dedent(f"""
        === STUDENT PROFILE ===
        Level: {student.get('level', 'unknown')}
        Subject: {student.get('subject', 'unknown')}
        Learning style: {student.get('learning_style', 'unknown')}
        Known mistakes: {', '.join(student.get('recent_mistakes', []))}

        === STUDENT QUESTION ===
        {session.get('student_question', '')}

        === AI TUTOR RESPONSE ===
        {session.get('ai_tutor_response', '')}

        === EXPECTED LEARNING OBJECTIVE ===
        {session.get('expected_learning_objective', '')}

        === RUBRIC (score each 0.0 to 1.0) ===
        {rubric_block}

        === PREVIOUS ATTEMPTS ===
        {history_block}

        Step {step_num + 1}: Provide your evaluation as JSON.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if model wraps the JSON
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        return json.loads(text.strip())
    except Exception as e:
        print(f"[DEBUG] Model call failed: {e}", flush=True)
        # Fallback: mid-range scores across all rubric items
        return {
            "rubric_scores": {item: 0.5 for item in rubric},
            "overall_quality": 0.5,
            "improvement_suggestion": None,
            "flag_for_human_review": False,
        }


# ── Single task runner ────────────────────────────────────────────────────

async def run_task(task_name: str, client: OpenAI) -> None:
    """Run one complete episode for a given task."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(timeout=120.0) as http:
            # Reset
            resp = await http.post(
                f"{ENV_URL}/reset",
                json={"task_name": task_name, "session_id": task_name},
            )
            resp.raise_for_status()
            result = resp.json()
            obs = result["observation"]
            done = result.get("done", False)
            history: List[str] = []

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                action = get_evaluation(client, obs, history)

                step_resp = await http.post(
                    f"{ENV_URL}/step",
                    json={"session_id": task_name, "action": action},
                )
                step_resp.raise_for_status()
                step_result = step_resp.json()

                reward = float(step_result.get("reward", 0.0))
                done = step_result.get("done", False)
                info = step_result.get("info", {})
                obs = step_result.get("observation", obs)

                error_msg: Optional[str] = None

                rewards.append(reward)
                steps_taken = step
                history.append(f"Step {step}: reward={reward:.3f}")

                action_summary = str(action.get("rubric_scores", {}))[:150]
                log_step(
                    step=step,
                    action=action_summary,
                    reward=reward,
                    done=done,
                    error=error_msg,
                )

                if done:
                    break

        score = max(rewards) if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main: run all 3 tasks ─────────────────────────────────────────────────

async def main() -> None:
    if not HF_TOKEN:
        print("[DEBUG] WARNING: HF_TOKEN not set — LLM calls will fail.", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = ["factual_accuracy", "pedagogical_quality", "personalisation_review"]

    for task_name in tasks:
        await run_task(task_name, client)
        await asyncio.sleep(2)  # brief pause between tasks


if __name__ == "__main__":
    asyncio.run(main())
