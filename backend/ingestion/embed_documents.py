from backend.llm.cohere_client import CohereClient


def embed_documents(
    chunks,
) -> list[list[float]]:
    if not chunks:
        return []

    client = CohereClient()

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    return client.embed_documents(texts)