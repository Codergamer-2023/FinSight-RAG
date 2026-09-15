import pytest

from backend.retrieval.retriever import Retriever


@pytest.fixture(scope="session")
def retriever():
    return Retriever()
