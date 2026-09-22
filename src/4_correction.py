# src/4_correction.py

from typing import Any, Dict, List


# Thresholds
RELEVANCE_THRESHOLD = 0.60
COVERAGE_THRESHOLD = 0.60
EVIDENCE_THRESHOLD = 0.60


def _get_score(result: Dict[str, Any], key: str) -> float:
    """
    Safely extract a numeric score from an evaluator result.
    """
    try:
        return float(result.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _get_label(result: Dict[str, Any]) -> str:
    """
    Safely extract the evaluator's correctness label.
    """
    return str(result.get("label", "Ambiguous")).strip().lower()


def evaluate_evidence_quality(
    evaluation_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Summarize evaluator results and determine whether
    retrieved evidence needs correction.

    Expected result format:
    [
        {
            "relevance": 0.8,
            "coverage": 0.7,
            "evidence_quality": 0.9,
            "label": "Correct"
        }
    ]
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
        }

    valid_results = [
        result
        for result in evaluation_results
        if isinstance(result, dict)
        and not result.get("error")
    ]

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
        sum(relevance_scores) / len(relevance_scores)
    )

    average_coverage = (
        sum(coverage_scores) / len(coverage_scores)
    )

    average_evidence_quality = (
        sum(evidence_scores) / len(evidence_scores)
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

    needs_correction = (
        average_relevance < RELEVANCE_THRESHOLD
        or average_coverage < COVERAGE_THRESHOLD
        or average_evidence_quality < EVIDENCE_THRESHOLD
        or correct_count == 0
    )

    if needs_correction:
        status = "needs_correction"
        reason = (
            "Retrieved evidence did not meet the configured "
            "quality thresholds."
        )
    else:
        status = "evidence_accepted"
        reason = (
            "Retrieved evidence met the configured "
            "quality thresholds."
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
        "total_evaluated": len(valid_results),
    }


def correct_retrieval(
    evaluation_results: List[Dict[str, Any]],
    retrieved_documents: List[Any],
) -> Dict[str, Any]:
    """
    Filter retrieved documents using evaluator labels
    and return a correction decision.

    This function does not perform another retrieval
    or web search. It prepares the result for the next
    C-RAG pipeline stage.
    """

    quality = evaluate_evidence_quality(evaluation_results)

    if not retrieved_documents:
        return {
            **quality,
            "corrected_documents": [],
            "correction_action": "no_documents",
        }

    # Match evaluator outputs to retrieved documents by index.
    corrected_documents = []

    valid_results = [
        result
        for result in evaluation_results
        if isinstance(result, dict)
        and not result.get("error")
    ]

    for index, document in enumerate(retrieved_documents):
        if index >= len(valid_results):
            continue

        result = valid_results[index]
        label = _get_label(result)

        if label == "correct":
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


if __name__ == "__main__":
    # Simple standalone test
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
    ]

    result = evaluate_evidence_quality(sample_evaluations)

    print("\nCorrection Evaluation")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")