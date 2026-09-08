import logging
import os
from fastapi import FastAPI, HTTPException, File, UploadFile

from backend.app.schemas import QueryRequest, QueryResponse, Source

from backend.retrieval.retriever import Retriever
from backend.retrieval.keyword_retriever import KeywordRetriever
from backend.retrieval.hybrid_retriever import HybridRetriever

from backend.retrieval.reranker import Reranker

from backend.llm.generator import Generator

from pathlib import Path

from backend.ingestion.load_pdf import load_pdf
from backend.ingestion.chunk_documents import chunk_documents
from backend.ingestion.embed_documents import embed_documents
from backend.ingestion.vector_store import add_to_vector_store
from backend.retrieval.schemas import RetrievedChunk

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
app = FastAPI(
    title = "FinSight API",
    version = "0.1.0",
)

retriever = Retriever()
keyword_retriever = KeywordRetriever(
    retriever.client
)
hybrid_retriever = HybridRetriever(
    retriever,
    keyword_retriever,
)
generator = Generator()
reranker = Reranker()

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

@app.get("/health")
async def health_check() -> dict[str, str]:
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "service": "FinSight API",
    }
@app.post("/api/v1/upload")
async def upload_document(
    file: UploadFile = File(...),
) -> dict[str, object]:

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A file is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    upload_directory = Path("data/uploads")
    upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = upload_directory / Path(file.filename).name

    try:
        file_content = await file.read()

        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        file_path.write_bytes(file_content)

        documents = load_pdf(file_path)
        chunks = chunk_documents(documents)
        embeddings = embed_documents(chunks)

        client = add_to_vector_store(
            chunks,
            embeddings,
            retriever.client,
        )
        keyword_chunks = [
            RetrievedChunk(
                document=chunk.metadata["document"],
                page=chunk.metadata["page"],
                text=chunk.page_content,
                score=0.0,
            )
            for chunk in chunks
        ]

        keyword_retriever.add_chunks(
            keyword_chunks
        )
        collection = client.get_collection(
            "finsight_documents"
        )

        logger.info(
            "Document uploaded: %s | chunks=%d",
            file.filename,
            len(chunks),
        )

        return {
            "message": "Document processed successfully.",
            "document": file.filename,
            "pages": len(documents),
            "chunks": len(chunks),
            "total_vectors": collection.points_count,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Document processing failed: %s",
            file.filename,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the uploaded document.",
        )

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    logger.info("Query received")
    try:
        chunks = hybrid_retriever.retrieve(
            request.question,
            top_k=RETRIEVAL_TOP_K,
        )
        if not chunks:
            logger.info("No relevant documents found")

            return QueryResponse(
                answer=(
                    "The information is not available in "
                    "the provided documents."
                ),
                sources=[],
            )
        chunks = reranker.rerank(
            request.question,
            chunks,
            top_k=RERANK_TOP_K,
        )
        if not chunks:
            logger.info("No relevant chunks after reranking")

            return QueryResponse(
                answer=(
                    "The information is not available in "
                    "the provided documents."
                ),
                sources=[],
            )
        answer = generator.generate(
            request.question,
            chunks,
        )
        sources = []
        seen_sources = set()

        for chunk in chunks:
            source_key = (chunk.document, chunk.page)

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            sources.append(
                Source(
                    document=chunk.document,
                    page=chunk.page,
                )
            )
        return QueryResponse(
            answer=answer,
            sources=sources,
        )
    except Exception:
        logger.exception("Query processing failed")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the query.",
        )
