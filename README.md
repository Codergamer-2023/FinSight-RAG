# 📊 FinSight-RAG

**FinSight-RAG** is an AI-powered financial document research assistant that allows users to upload financial PDFs, retrieve relevant information using hybrid search, rerank the most relevant passages, and generate grounded answers using an LLM.

It is designed as an end-to-end Retrieval-Augmented Generation (RAG) system with a production-style architecture using **FastAPI, Streamlit, Qdrant Cloud, Cohere, Groq, and Docker**.

## 🚀 Live Demo

**Frontend:**  
`https://finsight-rag-frontend.onrender.com`

**Backend API:**  
`https://finsight-rag-b2vj.onrender.com`

**API Health Check:**  
`https://finsight-rag-b2vj.onrender.com/health`

> The application is deployed using Render's free tier. Free-tier services may sleep when inactive and can take some time to wake up.

---

## ✨ Features

- 📄 Upload financial PDF documents dynamically
- 🔎 Hybrid retrieval using:
  - Semantic vector search
  - BM25 keyword search
- 🎯 Cohere-based reranking of retrieved chunks
- 🤖 Grounded answer generation using Groq
- 📚 Source and page attribution for generated answers
- ☁️ Qdrant Cloud vector database
- ⚡ FastAPI backend
- 🖥️ Streamlit frontend
- 🐳 Dockerized deployment
- 🌐 Public deployment on Render
- 🔐 Environment-based API key configuration
- 🛡️ Grounded responses that avoid inventing information not supported by the retrieved documents

---

## 🏗️ Architecture

```text
                         USER
                           │
                           ▼
                ┌────────────────────┐
                │ Streamlit Frontend │
                │      Render        │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   FastAPI Backend  │
                │      Render        │
                └─────────┬──────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      Document Upload              User Query
             │                         │
             ▼                         ▼
        PDF Extraction          Hybrid Retrieval
             │                  ┌──────┴──────┐
             ▼                  │             │
          Chunking         Semantic Search   BM25
             │                  │             │
             ▼                  └──────┬──────┘
       Cohere Embeddings                │
             │                         ▼
             ▼                 Candidate Chunks
        Qdrant Cloud                  │
             │                         ▼
             │                  Cohere Reranker
             │                         │
             └─────────────────────────┤
                                       ▼
                                  Relevant Context
                                       │
                                       ▼
                                   Groq LLM
                                       │
                                       ▼
                               Grounded Answer
                                       │
                                       ▼
                               Sources + Pages
```

---

## 🔄 RAG Pipeline

### 1. Document Ingestion

When a user uploads a PDF:

```text
PDF
 ↓
PyMuPDF
 ↓
Document Extraction
 ↓
Text Chunking
 ↓
Cohere Embeddings
 ↓
Qdrant Cloud
```

Each chunk stores metadata such as:

- Document name
- Page number
- Extracted text

### 2. Hybrid Retrieval

When a user asks a question, FinSight combines two retrieval approaches.

#### Semantic Retrieval

The question is converted into an embedding using **Cohere Embed** and searched against the vectors stored in Qdrant.

This helps retrieve passages based on semantic meaning rather than exact word matches.

#### Keyword Retrieval

The same question is processed using **BM25 keyword retrieval**.

This helps when important financial terminology, names, figures, or phrases match the source document closely.

### 3. Hybrid Candidate Selection

The semantic and keyword results are combined to produce a broader candidate set.

```text
Semantic Results
       +
BM25 Results
       ↓
Hybrid Candidate Set
```

### 4. Reranking

The candidate chunks are sent to **Cohere Rerank**.

The reranker scores the relevance of each candidate against the user's question and selects the strongest passages.

```text
Hybrid Candidates
       ↓
Cohere Rerank
       ↓
Top Relevant Chunks
```

### 5. Grounded Generation

The selected chunks are passed to the Groq-hosted LLM together with the user's question.

The generation prompt instructs the model to:

- Use only the supplied context
- Avoid outside knowledge
- Avoid inventing facts or numbers
- State when the requested information is unavailable

```text
Question + Retrieved Context
            ↓
         Groq LLM
            ↓
     Grounded Answer
```

### 6. Sources

The application returns document and page information associated with the retrieved chunks so that users can trace the answer back to the source material.

---

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| Language | Python |
| PDF Processing | PyMuPDF |
| Text Splitting | LangChain Text Splitters |
| Vector Database | Qdrant Cloud |
| Embeddings | Cohere Embed |
| Reranking | Cohere Rerank |
| Keyword Retrieval | BM25 |
| LLM | Groq |
| API Communication | REST |
| Containerization | Docker |
| Deployment | Render |
| Testing | Pytest |
| Version Control | Git + GitHub |

---

## 📁 Project Structure

```text
FinSight-RAG/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── schemas.py
│   │
│   ├── ingestion/
│   │   ├── load_pdf.py
│   │   ├── chunk_documents.py
│   │   ├── embed_documents.py
│   │   └── vector_store.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── keyword_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   ├── schemas.py
│   │   └── tests
│   │
│   ├── llm/
│   │   ├── cohere_client.py
│   │   └── generator.py
│   │
│   └── Dockerfile
│
├── frontend/
│   ├── app.py
│   └── Dockerfile
│
├── requirements.txt
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── pytest.ini
```

---

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "FinSight API"
}
```

### Upload Document

```http
POST /api/v1/upload
```

Accepts a PDF file and processes it through the ingestion pipeline.

### Query

```http
POST /api/v1/query
```

Example request:

```json
{
  "question": "What was the company's total revenue?"
}
```

The response contains the generated answer and source information.

---

## 🐳 Running with Docker

### Build the backend

```bash
docker build -f backend/Dockerfile -t finsight-backend .
```

### Build the frontend

```bash
docker build -f frontend/Dockerfile -t finsight-frontend .
```

### Run with Docker Compose

```bash
docker compose up --build
```

The services are exposed locally at:

```text
Frontend: http://localhost:8501
Backend:  http://localhost:8000
```

---

## 💻 Local Development

### 1. Clone the repository

```bash
git clone https://github.com/Codergamer-2023/FinSight-RAG.git
cd FinSight-RAG
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=openai/gpt-oss-20b

COHERE_API_KEY=<your-cohere-api-key>
COHERE_EMBED_MODEL=embed-v4.0
COHERE_EMBED_DIMENSION=1024
COHERE_RERANK_MODEL=rerank-v3.5

QDRANT_URL=<your-qdrant-url>
QDRANT_API_KEY=<your-qdrant-api-key>

COLLECTION_NAME=finsight_documents
RETRIEVAL_TOP_K=10
RERANK_TOP_K=5
```

**Never commit `.env` or API keys to GitHub.**

### 5. Start the backend

```bash
uvicorn backend.app.main:app --reload
```

### 6. Start the frontend

In another terminal:

```bash
streamlit run frontend/app.py
```

---

## 🧪 Testing

Run the automated test suite with:

```bash
pytest -q
```

The project includes tests covering retrieval, reranking, keyword retrieval, hybrid retrieval, and LLM-related components.

---

## 🔐 Security

Secrets are provided through environment variables rather than being hardcoded in source code.

The repository ignores:

```text
.env
.env.*
data/raw/
data/uploads/
data/qdrant/
```

API keys should never be committed to the repository.

---

## 📌 Design Decisions

### Why Hybrid Retrieval?

Pure semantic search is powerful for understanding meaning, but keyword retrieval can be especially useful for financial documents where exact terms, company names, metrics, and figures matter.

Combining semantic search with BM25 gives FinSight two complementary retrieval signals before reranking.

### Why Cohere?

Embedding and reranking are handled through hosted APIs rather than loading large transformer models inside the application container.

This keeps the deployed backend lightweight and avoids the memory overhead associated with hosting embedding and reranking models locally.

### Why Qdrant?

Qdrant provides vector search capabilities and allows the production application to use a hosted vector database rather than depending on the local filesystem of the Render container.

### Why Docker?

Docker provides a reproducible runtime environment for the application and its dependencies.

Render builds the Docker image automatically whenever changes are pushed to the configured GitHub branch.

---

## 🚀 Deployment Architecture

The production deployment consists of:

```text
GitHub
   │
   ▼
Render
   │
   ├── Streamlit Frontend
   │
   └── FastAPI Backend
          │
          ├── Qdrant Cloud
          ├── Cohere
          └── Groq
```

A typical development workflow is:

```text
Code Change
    ↓
Local Testing
    ↓
Git Commit
    ↓
Git Push
    ↓
Render detects new commit
    ↓
Docker image rebuilt
    ↓
Application redeployed
    ↓
Production testing
```

---

## ⚠️ Current Limitations

- Render free-tier services may sleep when inactive.
- External API rate limits may affect document ingestion and generation.
- Uploaded document files are handled by the application filesystem, while vectors are stored in Qdrant Cloud.
- The current system does not provide document-scoped retrieval controls.
- Conversation/chat history is not currently persisted as a server-side feature.

---

## 🔮 Future Improvements

Potential future improvements include:

- Persistent chat history
- Conversation/session management
- Document-scoped retrieval
- Document deletion and management
- Better ingestion progress reporting
- Background document processing
- Improved rate-limit handling and retry strategies
- Evaluation datasets and retrieval metrics
- Authentication and user accounts
- Production monitoring and observability

---

## 🎯 Project Goal

FinSight-RAG was built to demonstrate how a production-oriented RAG system can combine:

**document ingestion + embeddings + vector search + keyword retrieval + reranking + grounded LLM generation + API services + cloud infrastructure + containerized deployment**

rather than relying on a simple LLM chatbot.

---

## 👨‍💻 Author

**Manish Saini**

Built as a portfolio project demonstrating practical experience with:

- Retrieval-Augmented Generation
- LLM applications
- Information retrieval
- Vector databases
- API development
- Cloud deployment
- Docker
- Python