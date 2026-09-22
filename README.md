
# Corrective RAG with Human-in-the-Loop (C-RAG)

A privacy-aware Retrieval-Augmented Generation (RAG) system that retrieves information from private documents, evaluates the retrieved evidence, identifies missing information, and optionally uses public web search with human approval.

The goal is to generate answers grounded in available evidence while clearly identifying what is missing instead of guessing.

---

## Overview

Traditional RAG systems retrieve relevant document chunks and pass them directly to a language model. However, retrieved information may be irrelevant, incomplete, or insufficient to answer the user's question.

This project implements a **Corrective RAG (C-RAG)** workflow that evaluates retrieved evidence before generating an answer.

When private documents do not contain sufficient information, the system can ask the user for permission to perform an external web search.

![Uploading image.png…]()


### Key Features

- Private document ingestion and preprocessing
- PDF text extraction
- Document chunking
- Semantic vector search
- BM25 keyword-based retrieval
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)
- LLM-based evidence evaluation
- Evidence correction and decision routing
- Human-in-the-Loop (HITL) approval for external search
- Optional public web search using Tavily
- Grounded answer generation
- Identification of missing information
- Separation of private-document and public-web findings

---

## Architecture

![Corrective RAG with Human-in-the-Loop Architecture](docs/crag_architecture.png)

The architecture follows a private-knowledge-first approach. Public web search is optional and should only proceed after the user approves it.

### High-Level Workflow

```text
User Query
    |
    v
Query Understanding
    |
    v
Private Document Retrieval
    |
    +----------------------+
    |                      |
    v                      v
Vector Search          BM25 Search
    |                      |
    +----------+-----------+
               |
               v
       Rank Fusion (RRF)
               |
               v
       Evidence Evaluation
               |
               v
       Evidence Correction
               |
               v
       Decision Router
               |
       +-------+--------+
       |                |
       v                v
 Sufficient Evidence  Missing Evidence
       |                |
       |                v
       |          HITL Approval
       |                |
       |          +-----+-----+
       |          |           |
       |          v           v
       |        Approve      Decline
       |          |           |
       |          v           v
       |      Web Search   Private-Only
       |          |
       |          v
       |      Web Evidence
       |      Evaluation
       |          |
       +----------+
               |
               v
       Grounded Generation
               |
               v
       Final Answer
       + Citations
       + Missing Information
```

---

## System Components

### 1. Document Ingestion

The ingestion module extracts text and metadata from source documents.

Supported input formats depend on the configured ingestion pipeline.

Current workflow:
- Load PDF documents.
- Extract text from pages.
- Preserve page-level information.
- Store extracted content for downstream processing.

### 2. Document Chunking

Documents are split into smaller text chunks to improve retrieval.

Chunking helps the retriever locate relevant sections instead of processing entire documents for every query.

### 3. Hybrid Retrieval

The retrieval module combines two retrieval approaches.

| Method | Purpose |
|---|---|
| Vector Search | Retrieves semantically similar content |
| BM25 | Retrieves content based on keyword relevance |
| Reciprocal Rank Fusion | Combines rankings from both retrieval methods |

The resulting candidates are used as evidence for evaluation.

### 4. Evidence Evaluation

Retrieved documents are evaluated using an LLM.

The evaluator considers:

- **Relevance:** Does the retrieved evidence address the question?
- **Coverage:** How much of the requested information is provided?
- **Evidence Quality:** Does the document provide specific supporting information?

The evaluator produces scores, a label, and a reason for its assessment.

### 5. Evidence Correction

The correction module uses the evaluation results to determine whether retrieved evidence can be accepted.

Possible outcomes include:

- Evidence accepted
- Needs correction
- Evaluation failed
- Insufficient evidence

The decision depends on the configured evaluation thresholds and correction logic.

### 6. Human-in-the-Loop (HITL)

HITL introduces a user approval step before optional external web search.

When private evidence is insufficient, the system can ask:

> Some information is missing from your private documents. Do you want to search public sources?

The user can choose to:

- **Approve:** Continue with an appropriately sanitized public-search query.
- **Decline:** Generate an answer using private evidence only and identify missing information.

HITL is intended to keep the user in control of whether information is sent to an external search service.

### 7. Optional Web Search

If approved, the system can use Tavily to retrieve relevant public information.

The intended workflow is:

1. Prepare a public-safe query.
2. Search external sources.
3. Collect relevant web results.
4. Evaluate web evidence.
5. Keep public findings distinguishable from private-document findings.

**Implementation note:** Web evidence evaluation and full conflict-handling should only be considered implemented when those checks are connected and tested in the pipeline.

### 8. Grounded Answer Generation

The generator uses the available evidence to construct a response.

The response aims to:

- Answer using supported information.
- Distinguish private and public findings.
- Identify missing information.
- Avoid unsupported assumptions.
- Include source references where available.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM Evaluation | Ollama |
| Local LLM | Llama 3.2 |
| Embeddings | Sentence Transformers |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| External Search | Tavily |
| Generation | Groq API |
| Environment Configuration | `.env` |

---

## Project Structure

```text
C-RAG/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── extracted_pages.jsonl
│
├── src/
│   ├── 1_ingestion.py
│   ├── 2_retriever.py
│   ├── 3_evaluator.py
│   ├── 4_correction.py
│   ├── 5_web_search.py
│   ├── 6_generator.py
│   └── 7_pipeline.py
│
├── main.py
├── .env
├── .gitignore
└── README.md
```

*Update the structure above if your repository uses different filenames or directories.*

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Data-pageup/C-RAG.git
cd C-RAG
```

Replace the repository URL with your actual GitHub repository URL if it differs.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies used by your implementation.

### 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_available_groq_model
TAVILY_API_KEY=your_tavily_api_key
```

Use model IDs available to your Groq account.

**Never commit API keys, private documents, or sensitive environment variables to GitHub.**

### 5. Start Ollama

Ensure Ollama is installed and the configured local model is available.

For example:

```bash
ollama pull llama3.2:1b
```

---

## Running the Project

Run the main entry point from the project root:

```bash
python main.py
```

Enter a question about the documents supplied to the system.

The pipeline retrieves relevant evidence, evaluates it, and generates a response based on the available information.

If the question requires information that is not present in the private documents, the workflow can request permission for optional external search.

---

## Example

Using a synthetic employee handbook as a demonstration dataset:

### User Question

```text
How many annual leave days are provided?
```

### Private Document Evidence

```text
Employees receive 20 working days of annual leave
per calendar year.
```

### Expected Grounded Answer

```text
Employees receive 20 working days of annual leave
per calendar year.
```

The system should use the retrieved statement as supporting evidence rather than inventing a different entitlement.

### Missing Information Example

```text
Can unused annual leave be carried forward?
```

If the handbook does not specify the carry-forward policy, the system should identify that information as missing rather than assume a policy.

---

## Privacy Considerations

This project follows a private-knowledge-first design.

- Private documents are used for local retrieval.
- External web search is optional.
- User approval is intended to gate external search.
- Queries sent to external services should be sanitized.
- Private document contents should not be sent to external search services without explicit authorization.
- API keys and confidential files must remain outside public repositories.

Privacy depends on the actual implementation and configuration of each component.

---

## Current Limitations

- LLM-based evidence scores are not guaranteed to be correct.
- Retrieval can return irrelevant or incomplete chunks.
- A high evaluation score does not independently prove factual correctness.
- External search results may contain inaccurate or conflicting information.
- Source citations must be checked against the actual source content.
- HITL behavior depends on the configured interface and pipeline.
- Web evidence evaluation and conflict resolution require dedicated implementation and testing.
- The project has not been presented here as a production-validated system.

---

## Future Improvements

- [ ] Implement a persistent HITL pause/resume workflow.
- [ ] Add a Streamlit or FastAPI user interface.
- [ ] Add a dedicated web evidence evaluator.
- [ ] Implement evidence-level claim verification.
- [ ] Add reranking for improved retrieval quality.
- [ ] Add conflict detection between private and public sources.
- [ ] Add automated evaluation datasets and regression tests.
- [ ] Track retrieval and generation metrics.
- [ ] Add observability and audit logging.
- [ ] Containerize the application using Docker.
- [ ] Deploy the application with appropriate access controls.

---

## Project Goal

The goal of this project is to explore how corrective retrieval, evidence evaluation, and human approval can improve transparency and control in Retrieval-Augmented Generation systems.

Rather than treating every retrieved passage as reliable, the pipeline attempts to assess whether the available evidence is sufficient and communicate when information is missing.

---

## Author

**AG**

GitHub: https://github.com/Data-pageup

Portfolio: https://data-pageup.github.io/AmirthaganeshR
