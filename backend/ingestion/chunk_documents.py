from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents : list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
    )
    return splitter.split_documents(documents)

if __name__ == "__main__":
    from backend.ingestion.load_pdf import load_pdf

    documents = load_pdf()
    chunks = chunk_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Metadata: {chunk.metadata}")
        print(f"Characters: {len(chunk.page_content)}")
        print(chunk.page_content[:500])
