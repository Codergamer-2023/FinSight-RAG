from types import SimpleNamespace

from backend.llm.generator import Generator
from backend.retrieval.schemas import RetrievedChunk


class FakeCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="NVIDIA's revenue growth was driven by strong demand for its products."
                    )
                )
            ]
        )


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


def test_generator_returns_llm_answer():
    generator = Generator()

    generator.client = FakeClient()

    question = "What drove NVIDIA's revenue growth in fiscal 2026?"

    chunks = [
        RetrievedChunk(
            document="nvidia_2026_10k.pdf",
            page=69,
            text=(
                "NVIDIA reported strong revenue growth driven "
                "by demand for its products."
            ),
            score=0.9,
            rerank_score=5.2,
        )
    ]

    answer = generator.generate(
        question,
        chunks,
    )

    assert isinstance(answer, str)
    assert answer.strip()
    assert "revenue growth" in answer.lower()
