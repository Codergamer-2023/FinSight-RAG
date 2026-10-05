import os

import cohere
from dotenv import load_dotenv


load_dotenv()


class CohereClient:
    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not configured."
            )

        self.client = cohere.ClientV2(
            api_key=api_key
        )

        self.embed_model = os.getenv(
            "COHERE_EMBED_MODEL",
            "embed-v4.0",
        )

        self.rerank_model = os.getenv(
            "COHERE_RERANK_MODEL",
            "rerank-v3.5",
        )

        self.embed_dimension = int(
            os.getenv(
                "COHERE_EMBED_DIMENSION",
                "1024",
            )
        )
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        all_embeddings = []

        batch_size = 90

        for start in range(
            0,
            len(texts),
            batch_size,
        ):
            batch = texts[
                start:start + batch_size
            ]

            response = self.client.embed(
                model=self.embed_model,
                input_type="search_document",
                texts=batch,
                output_dimension=self.embed_dimension,
                embedding_types=["float"],
            )

            all_embeddings.extend(
                response.embeddings.float
            )   

            print(
                f"Embedded "
                f"{min(start + batch_size, len(texts))}"
                f"/{len(texts)} chunks"
            )

        return all_embeddings
    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        response = self.client.embed(
            model=self.embed_model,
            input_type="search_query",
            texts=[text],
            output_dimension=self.embed_dimension,
            embedding_types=["float"],
        )

        return response.embeddings.float[0]

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ):
        response = self.client.rerank(
            model=self.rerank_model,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        return response.results