from pathlib import Path
import pymupdf

from langchain_core.documents import Document

PDF_FILE = Path(
    "data/raw/nvidia/nvidia_2026_10k.pdf"
)

def load_pdf() -> list[Document]:
    pdf = pymupdf.open(PDF_FILE)
    documents =[]

    for page_number in range(len(pdf)):
        page = pdf[page_number]
        text = page.get_text()

        documents.append(
            Document(
                page_content = text,
                metadata = {
                    "document": PDF_FILE.name,
                    "page": page_number + 1,
                },
            )
        )

    pdf.close()
    return documents

if __name__ == "__main__":
    documents = load_pdf()

    print(f"Documents loaded: {len(documents)}")
    print(f"First page: {documents[0].metadata}")
    print(f"First page characters: {len(documents[0].page_content)}")
    print(f"Last page: {documents[-1].metadata}")
