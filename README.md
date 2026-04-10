---
title: nest-tutor-eval-env
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: mit
short_description: OpenEnv environment for evaluating AI tutor response quality
tags:
  - openenv
  - education
  - tutoring
  - edtech
  - nlp
  - reinforcement-learning
---

# nest-tutor-eval-env

> An OpenEnv environment for evaluating AI tutor response quality,
> built on the NEST.ai EdTech platform domain.

[![OpenEnv](https://img.shields.io/badge/openenv-1.0.0-blue)](https://github.com/openenv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/python-3.11-green)](https://python.org)

---

## Motivation

[NEST.ai](https://github.com/adhilrifayin) is a production-grade EdTech platform with an AI Co-Tutor that generates personalised responses to student questions in real time. As the system scales, human review of tutor responses becomes the bottleneck: reviewers can't evaluate every response at speed, binary correct/incorrect metrics miss pedagogical nuance, and personalisation quality is nearly impossible to measure without domain expertise.

**nest-tutor-eval-env** solves this. An AI agent is shown a real tutoring session — a student profile, their question, and the AI Co-Tutor's response — and must evaluate that response against a structured rubric. This is exactly the task a human reviewer performs inside NEST.ai. Training agents on this environment creates scalable, automated quality assurance for EdTech AI systems.

---

## Environment Overview

The agent receives a `NestObservation` containing:
- A **student profile** (skill level, subject, learning style, known mistakes)
- The **student's question**
- The **AI Co-Tutor's response**
- A **rubric** of evaluation criteria with descriptions

The agent submits a `NestAction` — rubric scores, an overall quality estimate, an improvement suggestion, and optionally a flag for human review.

The environment grades the action against ground-truth rubric scores, computes a shaped reward, and returns a new observation. Episodes run up to 5 steps, encouraging iterative refinement.

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `factual_accuracy` | Easy | Catch a clear factual error: the tutor claims `len()` only works on strings |
| `pedagogical_quality` | Medium | A factually correct but pedagogically weak response: no example, ignores student's learning style, misses root cause |
| `personalisation_review` | Hard | Factually correct AND clear — but a beginner analogy for an advanced ML student expecting Q/K/V depth |

### Why this difficulty progression?

- **Easy**: The error is explicit and domain-checkable — any agent with Python knowledge should catch it.
- **Medium**: Requires reasoning about *what makes good teaching*, not just factual correctness.
- **Hard**: Requires distinguishing "correct" from "appropriate" — the hardest judgement a reviewer makes.

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `session.student_question` | string | The student's question |
| `session.ai_tutor_response` | string | The AI Co-Tutor's response to evaluate |
| `session.expected_learning_objective` | string | What the student should learn |
| `session.student_profile.level` | enum | beginner \| intermediate \| advanced |
| `session.student_profile.subject` | string | Subject area |
| `session.student_profile.learning_style` | enum | visual \| text-heavy \| example-first |
| `session.student_profile.recent_mistakes` | list[str] | Known weak areas |
| `evaluation_rubric` | list[str] | Criterion names to score |
| `rubric_descriptions` | dict[str, str] | Human-readable criterion descriptions |
| `review_history` | list[str] | Summaries of previous steps this episode |
| `step_number` | int | Current step index |
| `task_name` | str | Active task name |

---

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `rubric_scores` | dict[str, float] | Score per criterion (0.0–1.0) |
| `overall_quality` | float [0.0, 1.0] | Holistic quality score |
| `improvement_suggestion` | str \| null | Specific actionable suggestion |
| `flag_for_human_review` | bool | Flag dangerous/broken responses only |

---

## Reward Function

The reward is **shaped across steps** with four components:

```
reward = base + improvement - step_penalty - flag_penalty
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| `base` | 70% of raw score | Rewards rubric alignment at every step |
| `improvement` | 20% of delta over best | Encourages iterative refinement |
| `step_penalty` | 0.01 × step | Discourages padding — be efficient |
| `flag_penalty` | 0.10 if unnecessary flag | Models real production cost of over-escalation |

**Why flag_penalty?** In a real EdTech platform, routing responses to human review costs time and money. An agent that flags every response "just in case" is useless. This penalty trains agents to escalate only when genuinely warranted — a novel reward mechanic for this domain.

Each task grader also applies domain-specific **bonuses**:
- Easy: bonus for scoring `factual_accuracy` low (catching the error)
- Medium: bonus for suggesting "base case" in the improvement field
- Hard: large bonus for correctly identifying depth mismatch while acknowledging technical accuracy

---

## Baseline Scores

Scores achieved with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router:

| Task | Difficulty | Score |
|------|-----------|-------|
| `factual_accuracy` | Easy | ~0.75 |
| `pedagogical_quality` | Medium | ~0.65 |
| `personalisation_review` | Hard | ~0.50 |

*Run `inference.py` to reproduce.*

---

## Setup

### Prerequisites
- Docker
- Python 3.11+
- HuggingFace API token with Inference Router access

### Run with Docker

```bash
# Build
docker build -t nest-tutor-eval-env .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=your_hf_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  nest-tutor-eval-env
```

### Run locally

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### API Endpoints

```bash
# Health check
curl http://localhost:7860/health

# Reset (start episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "factual_accuracy"}'

# Step (submit evaluation)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "rubric_scores": {"factual_accuracy": 0.1, "clarity": 0.8, "example_given": 0.1},
      "overall_quality": 0.4,
      "improvement_suggestion": "The response incorrectly states len() only works on strings.",
      "flag_for_human_review": false
    }
  }'

# State
curl http://localhost:7860/state

# List tasks
curl http://localhost:7860/tasks
```

### Run Inference Script

```bash
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=http://localhost:7860

python inference.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — (required) | HuggingFace / API key |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |

---

## OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

---

## About NEST.ai

NEST.ai is Adhil's flagship full-stack GenAI EdTech platform. Built with FastAPI, React, PostgreSQL, and multiple LLM APIs (OpenAI, Anthropic), it features an AI Co-Tutor, voice-to-visual pipeline, adaptive quizzes, personalised learning path optimisation, and a 13-entity relational data model covering Admin, Teacher, Student, Content, and Community modules. This OpenEnv environment is the quality evaluation and training layer that NEST.ai's AI Co-Tutor uses in production to benchmark and continuously improve response quality — making it a genuine real-world application, not a toy problem.

---

*Built by Adhil Rifayin K S — BSc AI, University of Calicut · IEDC Student Lead, Kerala Startup Mission*
