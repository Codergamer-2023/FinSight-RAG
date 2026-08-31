import os

from dotenv import load_dotenv
from groq import Groq

from backend.retrieval.schemas import RetrievedChunk

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"


class Generator:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:

        context = "\n\n".join(
            f"Document: {chunk.document}\n"
            f"Page: {chunk.page}\n"
            f"Text: {chunk.text}"
            for chunk in chunks
        )

        prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the context does not contain enough information to answer
the question, say that the information is not available in
the provided documents.

Do not use outside knowledge.
Do not invent facts or numbers.

Context:
{context}

Question:
{question}

Answer:
""".strip()

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content
