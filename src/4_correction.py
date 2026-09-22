
# src/4_correction.py

from typing import Any, Dict, List


# -----------------------------
# Thresholds
# -----------------------------

RELEVANCE_THRESHOLD = 0.60
COVERAGE_THRESHOLD = 0.60
EVIDENCE_THRESHOLD = 0.60


# -----------------------------
# Helpers
# -----------------------------

def _get_score(
    result: Dict[str, Any],
    key: str
) -> float:
    """Safely extract a valid numeric score."""

    value = result.get(key)

    if isinstance(value, bool):
        return 0.0

    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not 0.0 <= score <= 1.0:
        return 0.0

    return score


def _get_label(result: Dict[str, Any]) -> str:
    """Safely extract the evaluator label."""

    return str(
        result.get("label", "Ambiguous")
    ).strip().lower()


def _is_valid_result(result: Any) -> bool:
    """Exclude malformed results and evaluator failures."""

    if not isinstance(result, dict):
        return False

    if _get_label(result) not in {
        "correct",
        "ambiguous",
        "incorrect"
    }:
        return False

    required_scores = [
        "relevance",
        "coverage",
        "evidence_quality"
    ]

    for key in required_scores:
        value = result.get(key)

        if isinstance(value, bool):
            return False

        if not isinstance(value, (int, float)):
            return False

        if not 0.0 <= value <= 1.0:
            return False

    return True


# -----------------------------
# Evaluate evidence quality
# -----------------------------

def evaluate_evidence_quality(
    evaluation_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Summarize evaluator results and determine
    whether retrieved evidence needs correction.
    """

    if not evaluation_results:
        return {
            "status": "insufficient_evidence",
            "needs_correction": True,
            "reason": "No evaluation results were provided.",
            "average_relevance": 0.0,
            "average_coverage": 0.0,
            "average_evidence_quality": 0.0,
            "correct_count": 0,
            "ambiguous_count": 0,
            "incorrect_count": 0,
            "evaluation_error_count": 0,
            "total_evaluated": 0,
        }

    valid_results = [
        result
        for result in evaluation_results
        if _is_valid_result(result)
    ]

    evaluation_error_count = (
        len(evaluation_results) - len(valid_results)
    )

    if not valid_results:
        return {
            "status": "evaluation_failed",
            "needs_correction": True,
            "reason": "No valid evaluator results were available.",
            "average_relevance": 0.0,
            "average_coverage": 0.0,
            "average_evidence_quality": 0.0,
            "correct_count": 0,
            "ambiguous_count": 0,
            "incorrect_count": 0,
            "evaluation_error_count": evaluation_error_count,
            "total_evaluated": 0,
        }

    relevance_scores = [
        _get_score(result, "relevance")
        for result in valid_results
    ]

    coverage_scores = [
        _get_score(result, "coverage")
        for result in valid_results
    ]

    evidence_scores = [
        _get_score(result, "evidence_quality")
        for result in valid_results
    ]

    average_relevance = (
        sum(relevance_scores) / len(valid_results)
    )

    average_coverage = (
        sum(coverage_scores) / len(valid_results)
    )

    average_evidence_quality = (
        sum(evidence_scores) / len(valid_results)
    )

    correct_count = sum(
        1 for result in valid_results
        if _get_label(result) == "correct"
    )

    ambiguous_count = sum(
        1 for result in valid_results
        if _get_label(result) == "ambiguous"
    )

    incorrect_count = sum(
        1 for result in valid_results
        if _get_label(result) == "incorrect"
    )

    # Check whether at least one document
    # independently meets the evidence thresholds.
    accepted_results = [
        result
        for result in valid_results
        if (
            _get_label(result) == "correct"
            and _get_score(result, "relevance")
                >= RELEVANCE_THRESHOLD
            and _get_score(result, "coverage")
                >= COVERAGE_THRESHOLD
            and _get_score(result, "evidence_quality")
                >= EVIDENCE_THRESHOLD
        )
    ]

    needs_correction = len(accepted_results) == 0

    if needs_correction:
        status = "needs_correction"
        reason = (
            "No individual document met all configured "
            "evidence thresholds."
        )
    else:
        status = "evidence_accepted"
        reason = (
            "At least one document met all configured "
            "evidence thresholds."
        )

    return {
        "status": status,
        "needs_correction": needs_correction,
        "reason": reason,
        "average_relevance": round(average_relevance, 3),
        "average_coverage": round(average_coverage, 3),
        "average_evidence_quality": round(
            average_evidence_quality, 3
        ),
        "correct_count": correct_count,
        "ambiguous_count": ambiguous_count,
        "incorrect_count": incorrect_count,
        "evaluation_error_count": evaluation_error_count,
        "total_evaluated": len(valid_results),
    }


# -----------------------------
# Correct retrieval
# -----------------------------

def correct_retrieval(
    evaluation_results: List[Dict[str, Any]],
    retrieved_documents: List[Any],
) -> Dict[str, Any]:
    """
    Filter retrieved documents using evaluator labels.

    This function does not perform another retrieval
    or web search.
    """

    quality = evaluate_evidence_quality(
        evaluation_results
    )

    if not retrieved_documents:
        return {
            **quality,
            "corrected_documents": [],
            "correction_action": "no_documents",
        }

    corrected_documents = []

    # Keep evaluation results aligned with their original
    # retrieved documents. Do not filter before pairing.
    for index, document in enumerate(retrieved_documents):

        if index >= len(evaluation_results):
            continue

        result = evaluation_results[index]

        if not _is_valid_result(result):
            continue

        label = _get_label(result)

        relevance = _get_score(result, "relevance")
        coverage = _get_score(result, "coverage")
        evidence_quality = _get_score(
            result,
            "evidence_quality"
        )

        if (
            label == "correct"
            and relevance >= RELEVANCE_THRESHOLD
            and coverage >= COVERAGE_THRESHOLD
            and evidence_quality >= EVIDENCE_THRESHOLD
        ):
            corrected_documents.append(document)

    if corrected_documents:
        correction_action = "filtered_documents"
    else:
        correction_action = "no_reliable_documents"

    return {
        **quality,
        "corrected_documents": corrected_documents,
        "correction_action": correction_action,
    }


# -----------------------------
# Standalone test
# -----------------------------

if __name__ == "__main__":

    sample_evaluations = [
        {
            "relevance": 0.9,
            "coverage": 0.8,
            "evidence_quality": 0.9,
            "label": "Correct",
        },
        {
            "relevance": 0.7,
            "coverage": 0.7,
            "evidence_quality": 0.8,
            "label": "Correct",
        },
        {
            "relevance": None,
            "coverage": None,
            "evidence_quality": None,
            "label": "EvaluationError",
        },
    ]

    result = evaluate_evidence_quality(
        sample_evaluations
    )

    print("\nCorrection Evaluation")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")