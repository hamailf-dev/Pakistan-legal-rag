# Pakistan Legal RAG Assistant

A production-quality Retrieval-Augmented Generation (RAG) assistant for answering questions from Pakistani legal documents using grounded answers and source citations.

## Project Objective

The goal of this project is to build a legal RAG system that retrieves relevant information from Pakistani legal documents and generates answers grounded only in the retrieved legal context.

The system is designed to reduce hallucinations by requiring generated answers to be based on retrieved legal text.

## Legal Corpus

The initial legal corpus contains:

* **Pakistan Penal Code, 1860**

## RAG Pipeline

The complete pipeline is:

```text
Legal PDF
   ↓
Document Cleaning
   ↓
Text Chunking
   ↓
BGE-M3 Embeddings
   ↓
Qdrant Vector Database
   ↓
Semantic Retrieval
   ↓
Qwen LLM Generation
   ↓
Grounded Answer + Source Citation
   ↓
Streamlit UI
```

## Technologies Used

* **Python** — Core development language
* **BGE-M3** — Text embedding model for semantic representations
* **Qdrant** — Vector database for storing and searching embeddings
* **Qwen** — Large Language Model for grounded answer generation
* **Streamlit** — User interface
* **PyTorch / Transformers** — Model and inference support

## Key Features

* PDF document ingestion
* Legal text cleaning and preprocessing
* Semantic document chunking
* BGE-M3 embeddings
* Qdrant vector database
* Semantic similarity retrieval
* Qwen-based answer generation
* Source and page citations
* Retrieved legal text display
* Out-of-domain question handling
* Grounded responses to reduce hallucination

## Evaluation

The system was evaluated using 8 test questions covering legal section retrieval, grounded generation, source citation, and out-of-domain handling.

| Evaluation Area           | Result   |
| ------------------------- | -------- |
| Total Tests               | 8        |
| Tests Passed              | 8        |
| Overall Pass Rate         | **100%** |
| Correct Section Retrieval | ✅        |
| Grounded Answers          | ✅        |
| Source/Page Citation      | ✅        |
| Out-of-Domain Handling    | ✅        |
| Hallucination Observed    | None     |

### Evaluation Questions

The test set included questions about:

* Section 302 — Punishment of qatl-i-amd
* Section 379 — Punishment for theft
* Section 392 — Punishment for robbery
* Section 420 — Cheating
* Direct section-based questions
* An unrelated question to test safe refusal

The system correctly retrieved the relevant legal sections and refused to provide information when the question was outside the provided legal corpus.

## Project Structure

```text
pakistan-legal-rag/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── qdrant/
│
├── evaluation/
│   ├── test_questions.json
│   └── evaluation_results.md
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── ingestion/
│   ├── embeddings/
│   ├── retrieval/
│   └── generation/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/hamailf-dev/pakistan-legal-rag.git
cd pakistan-legal-rag
```

### 2. Create and activate the virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in the browser and provide an interface for asking questions about the Pakistan Penal Code.

## Example

**Question:**

> What is the punishment for theft?

**Retrieved Result:**

> Section 379 — Punishment for theft.

The application provides the relevant legal text together with the source document and page number.

## Current Status

The core RAG pipeline and evaluation are complete.

* ✅ Document ingestion
* ✅ Text cleaning
* ✅ Text chunking
* ✅ BGE-M3 embeddings
* ✅ Qdrant vector database
* ✅ Semantic retrieval
* ✅ Qwen answer generation
* ✅ Streamlit UI
* ✅ Source citations
* ✅ Evaluation — 8/8 tests passed

## Future Improvements

* Expand the legal corpus with additional Pakistani laws
* Improve retrieval and reranking
* Add more comprehensive automated evaluation
* Add multilingual legal question support
* Improve deployment and production infrastructure
