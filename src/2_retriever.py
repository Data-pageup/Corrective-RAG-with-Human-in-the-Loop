
import json
from pathlib import Path
import re

import chromadb
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "processed" / "extracted_pages.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma_db"


# -----------------------------
# Configuration
# -----------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 7

COLLECTION_NAME = "research_documents"

# Reciprocal Rank Fusion constant
RRF_K = 60


# -----------------------------
# 1. Load extracted PDF pages
# -----------------------------

def load_documents():
    """Load page-level text from the Phase 1 JSONL file."""

    documents = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            page = json.loads(line)

            documents.append(
                Document(
                    page_content=page["text"],
                    metadata={
                        "source": page["source"],
                        "page": page["page"]
                    }
                )
            )

    return documents


# -----------------------------
# 2. Split documents into chunks
# -----------------------------

def split_documents(documents):
    """Split pages into overlapping text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# -----------------------------
# 3. Initialize embeddings
# -----------------------------

def get_embeddings():
    """Load the local Hugging Face embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# -----------------------------
# 4. Create / load ChromaDB
# -----------------------------


def create_vector_store(chunks, embeddings):
    # Recreate the collection so it always matches the
    # freshly generated chunks used by BM25.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted existing ChromaDB collection.")
    except Exception:
        pass

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    vector_store.add_documents(chunks)

    print(f"Stored {len(chunks)} fresh chunks in ChromaDB.")

    return vector_store


# -----------------------------
# 5. BM25 keyword search
# -----------------------------

def tokenize_text(text):
    """Convert text into lowercase word tokens."""

    return re.findall(r"\w+", text.lower())


def create_bm25_index(chunks):
    """Create a BM25 index from the current document chunks."""

    tokenized_chunks = [
        tokenize_text(chunk.page_content)
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def bm25_search(bm25_index, chunks, query, top_k=TOP_K):
    """Retrieve chunks using BM25 keyword matching."""

    query_tokens = tokenize_text(query)

    scores = bm25_index.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices:
        if scores[index] <= 0:
            continue

        results.append(chunks[index])

        if len(results) >= top_k:
            break

    return results


# -----------------------------
# 6. Reciprocal Rank Fusion
# -----------------------------


def reciprocal_rank_fusion(bm25_results, vector_results, top_k=TOP_K):
    scores = {}
    documents = {}

    result_lists = [bm25_results, vector_results]

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):

            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content
            )

            scores[key] = scores.get(key, 0) + (
                1 / (RRF_K + rank)
            )

            documents[key] = doc

    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True
    )

    return [
        documents[key]
        for key in ranked_keys[:top_k]
    ]


# -----------------------------
# 7. Hybrid retrieval
# -----------------------------

def retrieve_documents(
    vector_store,
    query,
    chunks,
    bm25_index,
    top_k=TOP_K
):
    """Retrieve and fuse BM25 and vector search results."""

    # Keyword-based retrieval
    bm25_results = bm25_search(
        bm25_index,
        chunks,
        query,
        top_k=top_k
    )

    # Semantic vector retrieval
    vector_results = vector_store.similarity_search(
        query,
        k=top_k
    )

    # Combine both rankings
    hybrid_results = reciprocal_rank_fusion(
        bm25_results,
        vector_results,
        top_k=top_k
    )

    return hybrid_results


# -----------------------------
# 8. Run a retrieval test
# -----------------------------

def main():
    query = input("\nEnter your question: ")

    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = get_embeddings()

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    # Build the BM25 index from the loaded chunks
    bm25_index = create_bm25_index(chunks)

    results = retrieve_documents(
        vector_store,
        query,
        chunks,
        bm25_index
    )

    print("\n--- Hybrid Search Results ---")

    for i, doc in enumerate(results, start=1):
        print(f"\nDocument {i}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Content:\n{doc.page_content}")


if __name__ == "__main__":
    main()