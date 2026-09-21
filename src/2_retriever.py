# Read the extracted JSONL.
# Split pages into overlapping chunks.
# Convert each chunk into an embedding.
# Store the embeddings and metadata in ChromaDB

import json 
from pathlib import Path 

import chromadb
from sentence_transformers import SentenceTransformer
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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4

COLLECTION_NAME = "research_documents"


# -----------------------------
# 1. Load extracted PDF pages
# -----------------------------

def load_documents():
    """Load page-level text from the Phase 1 JSONL file."""

    documents = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            import json

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
    """Store document chunks in persistent ChromaDB."""

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    vector_store.add_documents(chunks)

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

    return vector_store


# -----------------------------
# 5. Retrieve relevant documents
# -----------------------------

def retrieve_documents(vector_store, query, top_k=TOP_K):
    """Retrieve the most relevant chunks for a query."""

    results = vector_store.similarity_search(
        query,
        k=top_k
    )

    return results


# -----------------------------
# 6. Run a retrieval test
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

    

    results = retrieve_documents(
        vector_store,
        query
    )

    print("\n--- Retrieved Documents ---")

    for i, doc in enumerate(results, start=1):

        print(f"\nDocument {i}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Content:\n{doc.page_content}")


if __name__ == "__main__":
    main()