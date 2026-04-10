"""
Pydantic models for nest-tutor-eval-env.
Defines the typed Observation, Action, and Reward models required by OpenEnv.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict


class StudentProfile(BaseModel):
    """Profile of the student receiving tutoring."""
    level: Literal["beginner", "intermediate", "advanced"]
    subject: str
    recent_mistakes: List[str] = Field(default=[])
    learning_style: Literal["visual", "text-heavy", "example-first"]


class TutorSession(BaseModel):
    """A complete AI tutoring session to be evaluated."""
    student_question: str
    ai_tutor_response: str
    expected_learning_objective: str
    student_profile: StudentProfile


class NestObservation(BaseModel):
    """
    OpenEnv Observation — what the agent sees each step.
    Contains the tutoring session, rubric criteria, prior review history,
    and current step number.
    """
    session: TutorSession
    evaluation_rubric: List[str] = Field(
        description="Rubric criteria the agent must score"
    )
    rubric_descriptions: Dict[str, str] = Field(
        default={},
        description="Human-readable description of each rubric criterion",
    )
    review_history: List[str] = Field(
        default=[],
        description="Summary of previous evaluation attempts this episode",
    )
    step_number: int = 0
    task_name: str = ""


class NestAction(BaseModel):
    """
    OpenEnv Action — what the agent submits each step.
    A structured evaluation of the AI tutor response.
    """
    rubric_scores: Dict[str, float] = Field(
        description="Score each rubric criterion from 0.0 (very poor) to 1.0 (excellent)"
    )
    overall_quality: float = Field(
        ge=0.0, le=1.0,
        description="Holistic quality score for the tutor response",
    )
    improvement_suggestion: Optional[str] = Field(
        default=None,
        description="Specific suggestion to improve the tutor response",
    )
    flag_for_human_review: bool = Field(
        default=False,
        description="Set True only if the response is dangerous or fundamentally broken",
    )


class NestReward(BaseModel):
    """
    OpenEnv Reward — what the environment returns after each step.
    Shaped reward with sub-components for interpretability.
    """
    score: float = Field(ge=0.0, le=1.0, description="Shaped reward for this step")
    rubric_alignment: float = Field(description="Raw grader score (alignment to ground truth)")
    false_flag_penalty: float = Field(description="Penalty applied for unnecessary human review flag")
    feedback: str = Field(description="Human-readable feedback for this step")
    ground_truth: Dict[str, float] = Field(
        default={},
        description="Ground truth rubric scores (revealed after step for transparency)",
    )
