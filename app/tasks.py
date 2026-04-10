"""
Task definitions, sessions, rubrics, ground-truth scores, and graders.
Three tasks of increasing difficulty:
  1. factual_accuracy   (easy)   — catch a factual error
  2. pedagogical_quality (medium) — evaluate teaching effectiveness
  3. personalisation_review (hard) — detect depth mismatch
"""
from dataclasses import dataclass, field
from typing import List, Dict, Callable

from app.models import StudentProfile, TutorSession, NestAction


@dataclass
class Task:
    """Container for all task metadata and grader."""
    name: str
    difficulty: str
    session: TutorSession
    evaluation_rubric: List[str]
    rubric_descriptions: Dict[str, str]
    ground_truth: Dict[str, float]
    grader: Callable[[NestAction], float]
    human_review_warranted: bool = False


# ─────────────────────────────────────────────────────────────
# TASK 1: factual_accuracy (EASY)
# The tutor says len() only works on strings — that's wrong.
# Agent must catch this clear factual error and score accordingly.
# ─────────────────────────────────────────────────────────────

EASY_SESSION = TutorSession(
    student_question="What does the len() function do in Python?",
    ai_tutor_response=(
        "Great question! The len() function returns the number of characters "
        "in a string, or the number of items in a list. Note that len() only "
        "works on strings — you cannot use it on integers or floats."
    ),
    expected_learning_objective=(
        "Student understands len() works on any sequence type, not just strings"
    ),
    student_profile=StudentProfile(
        level="beginner",
        subject="Python basics",
        recent_mistakes=["confused list indexing"],
        learning_style="example-first",
    ),
)

EASY_RUBRIC = ["factual_accuracy", "clarity", "example_given"]

EASY_DESCRIPTIONS = {
    "factual_accuracy": "Is every factual claim in the response correct?",
    "clarity":          "Is the explanation easy to understand for a beginner?",
    "example_given":    "Does the tutor include a concrete code example?",
}

EASY_GROUND_TRUTH = {
    "factual_accuracy": 0.15,   # low: the response has a clear factual error
    "clarity":          0.75,   # reasonable clarity otherwise
    "example_given":    0.05,   # almost no example given
}


def grade_easy(action: NestAction) -> float:
    """
    Grader for factual_accuracy task (EASY).
    Rewards rubric alignment, bonuses for catching the error and missing example.
    Returns float in [0.0, 1.0].
    """
    scores = action.rubric_scores
    gt = EASY_GROUND_TRUTH
    rubric_items = EASY_RUBRIC
    score = 0.0

    # Rubric alignment: reward closeness to ground truth per item
    for key in rubric_items:
        agent_score = scores.get(key, 0.5)
        diff = abs(agent_score - gt[key])
        score += (1.0 - diff) * (1.0 / len(rubric_items))

    # Bonus: explicitly catch the factual error (low factual_accuracy score)
    if scores.get("factual_accuracy", 1.0) < 0.35:
        score += 0.15

    # Bonus: notices no example was given
    if scores.get("example_given", 1.0) < 0.2:
        score += 0.10

    # Penalty: flagging for human review is unnecessary here
    if action.flag_for_human_review:
        score -= 0.10

    return round(min(max(score, 0.0), 1.0), 4)


# ─────────────────────────────────────────────────────────────
# TASK 2: pedagogical_quality (MEDIUM)
# Tutor's answer is factually correct but ignores the student's
# learning style, doesn't give an example, and doesn't address
# the student's known confusion (missing base case in recursion).
# Agent must distinguish "correct" from "good teaching".
# ─────────────────────────────────────────────────────────────

MEDIUM_SESSION = TutorSession(
    student_question="I don't understand why my recursive function runs forever.",
    ai_tutor_response=(
        "Recursion is when a function calls itself. If your function runs "
        "forever, it means there's an issue with your code. Make sure your "
        "function has the right logic and try again."
    ),
    expected_learning_objective=(
        "Student understands that a base case is the termination condition "
        "for recursion, and that missing it causes infinite recursion"
    ),
    student_profile=StudentProfile(
        level="intermediate",
        subject="Data structures",
        recent_mistakes=["missing base case", "infinite loop in while"],
        learning_style="example-first",
    ),
)

MEDIUM_RUBRIC = [
    "addresses_root_cause",
    "uses_example",
    "connects_to_student_history",
    "clarity",
]

MEDIUM_DESCRIPTIONS = {
    "addresses_root_cause":        "Does the response identify the base case as the likely cause?",
    "uses_example":                "Is a concrete code example provided?",
    "connects_to_student_history": "Does the response reference the student's known mistake areas?",
    "clarity":                     "Is the explanation clear and well-structured?",
}

MEDIUM_GROUND_TRUTH = {
    "addresses_root_cause":        0.10,   # vague "right logic" ≠ explaining base case
    "uses_example":                0.00,   # no example at all
    "connects_to_student_history": 0.00,   # completely ignores student profile
    "clarity":                     0.55,
}


def grade_medium(action: NestAction) -> float:
    """
    Grader for pedagogical_quality task (MEDIUM).
    Rewards rubric alignment, bonuses for catching missing example
    and mentioning base case in improvement suggestion.
    Returns float in [0.0, 1.0].
    """
    scores = action.rubric_scores
    gt = MEDIUM_GROUND_TRUTH
    rubric_items = MEDIUM_RUBRIC
    score = 0.0

    for key in rubric_items:
        agent_score = scores.get(key, 0.5)
        diff = abs(agent_score - gt[key])
        score += (1.0 - diff) * (1.0 / len(rubric_items))

    # Bonus: catches that no example was given for an example-first learner
    if scores.get("uses_example", 1.0) < 0.15:
        score += 0.12

    # Bonus: improvement suggestion mentions base case
    if action.improvement_suggestion:
        if "base case" in action.improvement_suggestion.lower():
            score += 0.10

    # Bonus: mentions the student's history contextually
    if action.improvement_suggestion:
        sugg = action.improvement_suggestion.lower()
        if "student" in sugg or "history" in sugg or "previous" in sugg:
            score += 0.05

    return round(min(max(score, 0.0), 1.0), 4)


# ─────────────────────────────────────────────────────────────
# TASK 3: personalisation_review (HARD)
# The tutor's response is factually correct AND clear — but it's
# a beginner-level analogy for an advanced ML student who asked
# about attention mechanisms. The agent must detect depth mismatch.
# Hard because: factually OK + pedagogically simple ≠ appropriate.
# ─────────────────────────────────────────────────────────────

HARD_SESSION = TutorSession(
    student_question="Can you explain how attention mechanisms work in transformers?",
    ai_tutor_response=(
        "Attention is a way for the model to focus on important words. "
        "Imagine you're reading a sentence — you naturally pay more attention "
        "to some words than others. Transformers do the same thing! "
        "The model looks at all words and decides which ones matter most. "
        "This helps it understand context better."
    ),
    expected_learning_objective=(
        "Student understands Q/K/V matrices, scaled dot-product attention, "
        "softmax normalisation, and multi-head attention — and can distinguish "
        "self-attention from cross-attention"
    ),
    student_profile=StudentProfile(
        level="advanced",
        subject="Machine learning",
        recent_mistakes=["confused self-attention vs cross-attention"],
        learning_style="text-heavy",
    ),
)

HARD_RUBRIC = [
    "depth_appropriate_to_level",
    "addresses_known_confusion",
    "technical_accuracy",
    "learning_objective_coverage",
    "style_match",
]

HARD_DESCRIPTIONS = {
    "depth_appropriate_to_level":  "Is the depth of explanation appropriate for an advanced ML student?",
    "addresses_known_confusion":   "Does the response address self-attention vs cross-attention confusion?",
    "technical_accuracy":          "Is the technical content correct (even if shallow)?",
    "learning_objective_coverage": "Are Q/K/V, softmax, and multi-head attention mentioned?",
    "style_match":                 "Does the style match a text-heavy learner (not just analogies)?",
}

HARD_GROUND_TRUTH = {
    "depth_appropriate_to_level":  0.05,  # beginner analogy for an advanced student
    "addresses_known_confusion":   0.00,  # self vs cross-attention not mentioned
    "technical_accuracy":          0.65,  # not wrong, just surface-level
    "learning_objective_coverage": 0.05,  # Q/K/V, softmax, multi-head — none mentioned
    "style_match":                 0.25,  # analogy-first for a text-heavy learner
}


def grade_hard(action: NestAction) -> float:
    """
    Grader for personalisation_review task (HARD).
    Key nuance: response is factually OK but depth-mismatched.
    Agent must distinguish correct-but-shallow from appropriate.
    Returns float in [0.0, 1.0].
    """
    scores = action.rubric_scores
    gt = HARD_GROUND_TRUTH
    rubric_items = HARD_RUBRIC
    score = 0.0

    for key in rubric_items:
        agent_score = scores.get(key, 0.5)
        diff = abs(agent_score - gt[key])
        score += (1.0 - diff) * (1.0 / len(rubric_items))

    # Key bonus: agent correctly identifies depth mismatch DESPITE factual correctness
    depth = scores.get("depth_appropriate_to_level", 1.0)
    accuracy = scores.get("technical_accuracy", 0.0)
    if depth < 0.20 and accuracy > 0.40:
        score += 0.20  # correctly detected the nuance

    # Bonus: notices Q/K/V / learning objectives not covered
    if scores.get("learning_objective_coverage", 1.0) < 0.20:
        score += 0.10

    # Should NOT flag for human review — just needs a rewrite, not escalation
    if not action.flag_for_human_review:
        score += 0.05

    return round(min(max(score, 0.0), 1.0), 4)


# ─────────────────────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────────────────────

TASKS: dict[str, Task] = {
    "factual_accuracy": Task(
        name="factual_accuracy",
        difficulty="easy",
        session=EASY_SESSION,
        evaluation_rubric=EASY_RUBRIC,
        rubric_descriptions=EASY_DESCRIPTIONS,
        ground_truth=EASY_GROUND_TRUTH,
        grader=grade_easy,
        human_review_warranted=False,
    ),
    "pedagogical_quality": Task(
        name="pedagogical_quality",
        difficulty="medium",
        session=MEDIUM_SESSION,
        evaluation_rubric=MEDIUM_RUBRIC,
        rubric_descriptions=MEDIUM_DESCRIPTIONS,
        ground_truth=MEDIUM_GROUND_TRUTH,
        grader=grade_medium,
        human_review_warranted=False,
    ),
    "personalisation_review": Task(
        name="personalisation_review",
        difficulty="hard",
        session=HARD_SESSION,
        evaluation_rubric=HARD_RUBRIC,
        rubric_descriptions=HARD_DESCRIPTIONS,
        ground_truth=HARD_GROUND_TRUTH,
        grader=grade_hard,
        human_review_warranted=False,
    ),
}
