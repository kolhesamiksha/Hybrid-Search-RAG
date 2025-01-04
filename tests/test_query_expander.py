import os
from unittest.mock import patch, MagicMock
import pytest
from hybrid_rag.src.advance_rag.query_expander import CustomQueryExpander
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Mock environment variables based on .env.example
MOCK_ENV_VARS = {
    "LLM_MODEL_NAME": "llama-3.1-70b-versatile",
    "DENSE_EMBEDDING_MODEL": "jinaai/jina-embeddings-v2-base-en",
    "SPARSE_EMBEDDING_MODEL": "Qdrant/bm42-all-minilm-l6-v2-attentions",
    "QUESTION": "What is Generative AI?",
}

@pytest.fixture(scope="module", autouse=True)
def mock_env():
    with patch.dict(os.environ, MOCK_ENV_VARS):
        yield


@pytest.fixture
def query_expander():
    with patch("hybrid_rag.src.models.query_expansion_model.models.QueryExpansionModel") as mock_query_model_cls:
        mock_query_model_cls.return_value.get_expanded_queries.return_value = ["expanded_query1", "expanded_query2"]
        return CustomQueryExpander(model_name=os.getenv("DENSE_EMBEDDING_MODEL"))

def test_initialization(query_expander):
    assert query_expander.model_name == os.getenv("DENSE_EMBEDDING_MODEL")
    assert query_expander.query_model is not None

def test_expand_query(query_expander):
    query = os.getenv("QUESTION")
    top_k = 3  # Set top_k for the test case
    
    expanded_queries = query_expander.expand_query(query, top_k)
    
    assert expanded_queries == ["expanded_query1", "expanded_query2"]
    query_expander.logger.info.assert_called_with(f"Expanding query: {query} with top_k: {top_k}")

def test_expand_query_exception_handling(query_expander):
    query = "Invalid input"
    top_k = 3

    # Simulate an exception during query expansion
    query_expander.query_model.get_expanded_queries.side_effect = Exception("Model error")

    with pytest.raises(Exception, match="Model error"):
        query_expander.expand_query(query, top_k)
    query_expander.logger.error.assert_called_with("Failed to expand query. Reason: Model error")

def test_env_variable_usage():
    assert os.getenv("DENSE_EMBEDDING_MODEL") == "jinaai/jina-embeddings-v2-base-en"
    assert os.getenv("QUESTION") == "What is Generative AI?"

