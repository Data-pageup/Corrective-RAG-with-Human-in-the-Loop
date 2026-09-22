# src/7_pipeline.py

import importlib.util
from pathlib import Path


# --------------------------------------------------
# Load numbered modules
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent


def load_module(module_name, file_name):
    """Load a Python module from the src directory."""

    module_path = SRC_DIR / file_name

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {file_name}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


retriever = load_module(
    "retriever",
    "2_retriever.py"
)

evaluator = load_module(
    "evaluator",
    "3_evaluator.py"
)

correction = load_module(
    "correction",
    "4_correction.py"
)

web_search = load_module(
    "web_search",
    "5_web_search.py"
)

generator = load_module(
    "generator",
    "6_generator.py"
)


# --------------------------------------------------
# Human-in-the-loop consent
# --------------------------------------------------

def request_web_search():
    """
    Ask the user for permission before public web search.

    The user supplies a separate query so private
    document contents are not automatically sent.
    """

    print("\n" + "-" * 60)
    print("HUMAN-IN-THE-LOOP: WEB SEARCH CONSENT")
    print("-" * 60)

    print(
        "\nThe private documents did not provide "
        "enough accepted evidence."
    )

    print(
        "\nYou can choose whether to search public websites."
    )

    print(
        "Only the separate search query you enter below "
        "will be sent to the web search service."
    )

    print(
        "Do not include confidential information, "
        "private document text, or sensitive details."
    )

    choice = input(
        "\nDo you approve a public web search? (yes/no): "
    ).strip().lower()

    if choice not in ("yes", "y"):
        print("\nWeb search declined.")
        return None

    print(
        "\nEnter a generic, public-safe search query."
    )

    search_query = input(
        "Public search query: "
    ).strip()

    if not search_query:
        print("\nNo search query provided. Skipping web search.")
        return None

    return search_query


# --------------------------------------------------
# C-RAG Pipeline
# --------------------------------------------------

def run_pipeline(question: str):

    print("\n" + "=" * 60)
    print("CORRECTIVE RAG PIPELINE")
    print("=" * 60)

    # ----------------------------------------------
    # 1. Load documents and vector store
    # ----------------------------------------------

    print("\n[1] Loading documents and vector store...")

    documents = retriever.load_documents()

    chunks = retriever.split_documents(documents)

    embeddings = retriever.get_embeddings()

    vector_store = retriever.create_vector_store(
        chunks,
        embeddings
    )

    bm25_index = retriever.create_bm25_index(chunks)

    # ----------------------------------------------
    # 2. Retrieve documents
    # ----------------------------------------------

    print("\n[2] Retrieving relevant documents...")

    retrieved_documents = retriever.retrieve_documents(
        vector_store=vector_store,
        query=question,
        chunks=chunks,
        bm25_index=bm25_index
    )

    print(
        f"Retrieved documents: "
        f"{len(retrieved_documents)}"
    )

    # ----------------------------------------------
    # 3. Evaluate retrieved evidence
    # ----------------------------------------------

    print("\n[3] Evaluating retrieved evidence...")

    evaluation_results = evaluator.evaluate_documents(
        question=question,
        documents=retrieved_documents
    )

    # ----------------------------------------------
    # 4. Correct / filter evidence
    # ----------------------------------------------

    print("\n[4] Correcting retrieved evidence...")

    correction_result = correction.correct_retrieval(
        evaluation_results=evaluation_results,
        retrieved_documents=retrieved_documents
    )

    corrected_documents = correction_result.get(
        "corrected_documents",
        []
    )

    needs_correction = correction_result.get(
        "needs_correction",
        True
    )

    print(
        "Correction status:",
        correction_result.get("status", "unknown")
    )

    # ----------------------------------------------
    # 5. Human approval + optional web search
    # ----------------------------------------------

    print("\n[5] Checking whether web search is needed...")

    web_results = []

    local_evidence_sufficient = (
        not needs_correction
        and bool(corrected_documents)
    )

    if local_evidence_sufficient:

        print("\nLocal evidence is sufficient.")
        print("Using accepted private documents only.")

    else:

        print("\nLocal evidence may be insufficient.")

        search_query = request_web_search()

        if search_query:

            print("\nSearching public websites...")

            try:
                web_results = web_search.web_search(
                    query=search_query,
                    max_results=5
                )

                print(
                    f"Public web results: {len(web_results)}"
                )

            except Exception as error:
                print("\nWeb search failed:", error)
                web_results = []

        else:
            print(
                "\nContinuing without public web search."
            )

    # ----------------------------------------------
    # 6. Generate final answer
    # ----------------------------------------------

    print("\n[6] Generating final answer...")

    answer = generator.generate_answer(
        question=question,
        documents=corrected_documents,
        web_results=web_results
    )

    # ----------------------------------------------
    # 7. Display results
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)

    # ----------------------------------------------
    # Return pipeline results
    # ----------------------------------------------

    return {
        "question": question,
        "retrieved_documents": retrieved_documents,
        "evaluation_results": evaluation_results,
        "correction_result": correction_result,
        "web_results": web_results,
        "answer": answer
    }