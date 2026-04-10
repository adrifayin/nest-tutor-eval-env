"""
Core OpenEnv environment: NestTutorEnv.
Implements reset(), step(), and state() per OpenEnv spec.
"""
from typing import List

from app.models import NestObservation, NestAction, NestReward
from app.tasks import TASKS, Task


class NestTutorEnv:
    """
    OpenEnv-compliant environment for evaluating AI tutor response quality.

    Models the evaluation workflow inside NEST.ai's AI Co-Tutor system.
    An agent receives a tutoring session (student profile + question + AI response)
    and must evaluate that response against a structured rubric.

    The reward is shaped across steps to encourage iterative improvement:
      - Base reward: 70% of the raw grader score
      - Improvement bonus: 20% for improving over the previous best score
      - Step efficiency penalty: small per-step penalty to discourage padding
      - False-flag penalty: penalty for unnecessarily flagging for human review
    """

    def __init__(self, task_name: str = "factual_accuracy"):
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task: {task_name!r}. Valid options: {list(TASKS.keys())}"
            )
        self._task_name = task_name
        self._task: Task = TASKS[task_name]
        self._step_count: int = 0
        self._max_steps: int = 5
        self._done: bool = False
        self._review_history: List[str] = []
        self._best_score: float = 0.0
        self._last_reward: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """
        Reset the environment and return the initial observation.
        Every call returns a clean state — no carry-over from a previous episode.
        """
        self._step_count = 0
        self._done = False
        self._review_history = []
        self._best_score = 0.0
        self._last_reward = 0.0

        obs = NestObservation(
            session=self._task.session,
            evaluation_rubric=self._task.evaluation_rubric,
            rubric_descriptions=self._task.rubric_descriptions,
            review_history=[],
            step_number=0,
            task_name=self._task_name,
        )
        return {
            "observation": obs.model_dump(),
            "done": False,
            "info": {
                "task": self._task_name,
                "difficulty": self._task.difficulty,
                "max_steps": self._max_steps,
            },
        }

    def step(self, action: dict) -> dict:
        """
        Process one evaluation action.

        Args:
            action: dict matching NestAction schema.

        Returns:
            dict with keys: observation, reward (float), done (bool), info (dict).
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )

        self._step_count += 1

        # Parse action — penalise but don't crash on bad format
        try:
            parsed = NestAction(**action)
        except Exception as e:
            reward_obj = NestReward(
                score=0.0,
                rubric_alignment=0.0,
                false_flag_penalty=0.0,
                feedback=f"Invalid action format: {e}",
                ground_truth=self._task.ground_truth,
            )
            self._done = True
            return self._build_result(reward_obj, self._done)

        # Grade
        raw_score = self._task.grader(parsed)

        # Shaped reward
        shaped = self._compute_shaped_reward(raw_score, self._step_count, parsed)
        self._best_score = max(self._best_score, raw_score)
        self._last_reward = shaped

        # Reward components for transparency
        false_flag_penalty = (
            0.10
            if parsed.flag_for_human_review and not self._task.human_review_warranted
            else 0.0
        )

        feedback = (
            f"Step {self._step_count}: raw={raw_score:.3f}, shaped={shaped:.3f}. "
            + (
                "Excellent evaluation!" if raw_score > 0.85
                else "Good progress — keep refining your rubric scores." if raw_score > 0.65
                else "Need improvement — review the rubric descriptions carefully."
            )
        )

        reward_obj = NestReward(
            score=shaped,
            rubric_alignment=raw_score,
            false_flag_penalty=false_flag_penalty,
            feedback=feedback,
            ground_truth=self._task.ground_truth,
        )

        # Episode ends when agent reaches near-perfect score or max steps reached
        self._done = raw_score >= 0.92 or self._step_count >= self._max_steps

        summary = f"Step {self._step_count}: raw_score={raw_score:.3f}, shaped={shaped:.3f}"
        self._review_history.append(summary)

        return self._build_result(reward_obj, self._done)

    def state(self) -> dict:
        """Return the full current environment state."""
        return {
            "task_name": self._task_name,
            "difficulty": self._task.difficulty,
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "done": self._done,
            "best_score": self._best_score,
            "last_reward": self._last_reward,
            "review_history": self._review_history,
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _compute_shaped_reward(
        self, raw_score: float, step: int, action: NestAction
    ) -> float:
        """
        Four-component shaped reward:
          1. Base (70%): raw grader score scaled down — ensures partial credit
          2. Improvement (20%): bonus for improving over previous best — encourages iteration
          3. Step efficiency (-): small per-step penalty — discourages padding
          4. False-flag (-): penalty for unnecessarily flagging — models real production cost
        """
        base = raw_score * 0.70
        improvement = max(0.0, raw_score - self._best_score) * 0.20
        step_penalty = 0.01 * step
        flag_penalty = (
            0.10
            if action.flag_for_human_review and not self._task.human_review_warranted
            else 0.0
        )

        shaped = base + improvement - step_penalty - flag_penalty
        return round(min(max(shaped, 0.0), 1.0), 4)

    def _build_result(self, reward: NestReward, done: bool) -> dict:
        """Assemble the step result dict."""
        obs = NestObservation(
            session=self._task.session,
            evaluation_rubric=self._task.evaluation_rubric,
            rubric_descriptions=self._task.rubric_descriptions,
            review_history=self._review_history[-3:],
            step_number=self._step_count,
            task_name=self._task_name,
        )
        return {
            "observation": obs.model_dump(),
            "reward": reward.score,
            "done": done,
            "info": reward.model_dump(),
        }
