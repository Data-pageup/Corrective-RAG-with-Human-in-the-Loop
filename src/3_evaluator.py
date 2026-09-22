
import json
import sys
from pathlib import Path
import importlib

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Import retriever
# -----------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

retriever = importlib.import_module("2_retriever")


# -----------------------------
# Configuration
# -----------------------------

LLM_MODEL = "llama3.2:1b"

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    format="json"
)


# -----------------------------
# Evaluation prompt
# -----------------------------

evaluation_prompt = ChatPromptTemplate.from_template(
    """
You are a document relevance evaluator.

Evaluate only the actual document provided below.

QUESTION:
{question}

DOCUMENT:
{document}

Assess these three dimensions:

1. relevance:
Does the document discuss the specific subject
asked about?

2. coverage:
How much of the requested information does
the document actually provide?

3. evidence_quality:
Does the document contain specific facts,
explanations, or steps supporting an answer?

Scoring:
0.0 = None
0.5 = Partial
1.0 = Strong

Use intermediate values when appropriate.

Do not assume relevance just because the
document discusses a related topic.

Do not invent facts.
Do not copy scores from examples.
Base every score and explanation on the
actual document.

Return a valid JSON object with exactly these fields:
relevance, coverage, evidence_quality, reason.

The three scores must be numbers between 0.0 and 1.0.
The reason must explain what information the
document actually contains.
"""
)


# -----------------------------
# Helpers
# -----------------------------

def get_score(value, field_name):
    """Validate and return a numeric score between 0 and 1."""

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be a number, not a boolean"
        )

    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")

    score = float(value)

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1"
        )

    return score


def evaluation_error(document, error):
    """Return a consistent result when evaluation fails."""

    return {
        "label": "EvaluationError",
        "relevance": None,
        "coverage": None,
        "evidence_quality": None,
        "evidence_score": None,
        "reason": str(error),
        "source": document.metadata.get("source"),
        "page": document.metadata.get("page"),
        "document": document.page_content
    }


# -----------------------------
# Evaluate one document
# -----------------------------

def evaluate_document(question, document):

    try:
        prompt = evaluation_prompt.invoke({
            "question": question,
            "document": document.page_content
        })

        response = llm.invoke(prompt)
        raw_response = response.content.strip()

        print("\n--- RAW OLLAMA RESPONSE ---")
        print(raw_response)
        print("---------------------------")

        # Handle Markdown code fences
        if raw_response.startswith("```"):
            raw_response = raw_response.split(
                "\n", 1
            )[1].rsplit("```", 1)[0].strip()

        result = json.loads(raw_response)

        # Validate response type first
        if not isinstance(result, dict):
            raise ValueError(
                "Ollama response must be a JSON object"
            )

        # Validate exact JSON fields
        expected_fields = {
            "relevance",
            "coverage",
            "evidence_quality",
            "reason"
        }

        if set(result.keys()) != expected_fields:
            raise ValueError(
                "Response must contain exactly: "
                "relevance, coverage, evidence_quality, reason"
            )

        # Validate scores
        relevance = get_score(
            result.get("relevance"),
            "relevance"
        )

        coverage = get_score(
            result.get("coverage"),
            "coverage"
        )

        quality = get_score(
            result.get("evidence_quality"),
            "evidence_quality"
        )

        reason = result.get("reason")

        if not isinstance(reason, str):
            raise ValueError("reason must be a string")

        # Calculate combined evidence score
        evidence_score = (
            0.3 * relevance
            + 0.4 * coverage
            + 0.3 * quality
        )

        # Classify the document
        if relevance < 0.4 or coverage < 0.4:
            label = "Incorrect"

        elif evidence_score >= 0.70 and coverage >= 0.60:
            label = "Correct"

        else:
            label = "Ambiguous"

        return {
            "label": label,
            "relevance": relevance,
            "coverage": coverage,
            "evidence_quality": quality,
            "evidence_score": round(evidence_score, 3),
            "reason": reason,
            "source": document.metadata.get("source"),
            "page": document.metadata.get("page"),
            "document": document.page_content
        }

    except Exception as error:
        print(f"\nEvaluator error: {error}")
        return evaluation_error(document, error)


# -----------------------------
# Evaluate all documents
# -----------------------------

def evaluate_documents(question, documents):

    results = []

    for index, document in enumerate(documents, start=1):

        print(
            f"\nEvaluating document {index}/{len(documents)}"
        )

        result = evaluate_document(question, document)
        results.append(result)

        print(f"Label: {result['label']}")
        print(f"Relevance: {result['relevance']}")
        print(f"Coverage: {result['coverage']}")
        print(f"Evidence quality: {result['evidence_quality']}")
        print(f"Evidence score: {result['evidence_score']}")
        print(f"Reason: {result['reason']}")

        print("\n--- RETRIEVED DOCUMENT TEXT ---")
        print(document.page_content)

    return results


# -----------------------------
# Overall evaluation
# -----------------------------

def summarize_evaluation(results):

    counts = {
        "Correct": 0,
        "Ambiguous": 0,
        "Incorrect": 0,
        "EvaluationError": 0
    }

    for result in results:
        label = result.get("label", "EvaluationError")

        if label not in counts:
            label = "EvaluationError"

        counts[label] += 1

    # Exclude evaluation failures from aggregation
    valid_results = [
        result
        for result in results
        if result.get("label") in {
            "Correct",
            "Ambiguous",
            "Incorrect"
        }
        and result.get("coverage") is not None
        and result.get("evidence_score") is not None
    ]

    if not valid_results:
        decision = (
            "EvaluationError"
            if results
            else "Incorrect"
        )

        coverage = None
        evidence_score = None

    else:
        # Select one document using coverage first,
        # then evidence score as the tie-breaker.
        best_result = max(
            valid_results,
            key=lambda result: (
                result["coverage"],
                result["evidence_score"]
            )
        )

        coverage = best_result["coverage"]
        evidence_score = best_result["evidence_score"]
        decision = best_result["label"]

    return {
        "decision": decision,
        "counts": counts,
        "max_coverage": coverage,
        "max_evidence_score": evidence_score,
        "results": results
    }


# -----------------------------
# Main pipeline
# -----------------------------

def main():

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    # Load and split documents once
    source_documents = retriever.load_documents()
    chunks = retriever.split_documents(source_documents)

    # Initialize embeddings and ChromaDB
    embeddings = retriever.get_embeddings()

    vector_store = retriever.create_vector_store(
        chunks,
        embeddings
    )

    # Build BM25 keyword index
    bm25_index = retriever.create_bm25_index(chunks)

    # Hybrid retrieval: BM25 + vector search + RRF
    documents = retriever.retrieve_documents(
        vector_store=vector_store,
        query=question,
        chunks=chunks,
        bm25_index=bm25_index
    )

    print(f"\nRetrieved {len(documents)} documents.")

    # Evaluate retrieved documents
    results = evaluate_documents(
        question,
        documents
    )

    # Summarize evaluation
    summary = summarize_evaluation(results)

    print("\n--- Overall Evaluation ---")
    print(f"Decision: {summary['decision']}")
    print(f"Counts: {summary['counts']}")
    print(f"Maximum coverage: {summary['max_coverage']}")
    print(
        f"Maximum evidence score: "
        f"{summary['max_evidence_score']}"
    )


if __name__ == "__main__":
    main()