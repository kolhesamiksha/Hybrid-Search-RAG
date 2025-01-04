import pytest
from unittest.mock import patch, MagicMock
from hybrid_rag.src.models.retriever_model.models import EmbeddingModels

@pytest.fixture
def mock_logger():
    """Fixture for a mocked logger."""
    return MagicMock()

@pytest.fixture
def embed_model(mock_logger):
    """Fixture for initializing the EmbeddingModels class."""
    return EmbeddingModels(embed_model="test-model")

@patch("hybrid_rag.src.hybrid_search.SparseFastEmbedEmbeddings")
def test_sparse_embedding_model_success(mock_sparse_embed, embed_model, mock_logger):
    # Mock SparseFastEmbedEmbeddings instance
    mock_instance = MagicMock()
    mock_sparse_embed.return_value = mock_instance
    mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    texts = ["Test text 1", "Test text 2"]
    embeddings = embed_model.sparse_embedding_model(texts)
    
    assert embeddings == [[0.1, 0.2, 0.3]]
    mock_sparse_embed.assert_called_once_with(model_name="test-model")
    mock_instance.embed_documents.assert_called_once_with([texts])
    mock_logger.info.assert_called_once_with("Successfully Converted the text into Sparse Vectors Using model: test-model")

@patch("hybrid_rag.src.hybrid_search.SparseFastEmbedEmbeddings", side_effect=Exception("Embedding Error"))
def test_sparse_embedding_model_failure(mock_sparse_embed, embed_model, mock_logger):
    texts = ["Test text 1", "Test text 2"]
    
    with pytest.raises(Exception, match="Embedding Error"):
        embed_model.sparse_embedding_model(texts)
    
    mock_logger.error.assert_called_once_with(
        "Failed to Initialised LLM model Reason: Embedding Error -> TRACEBACK : "
    )

@patch("hybrid_rag.src.hybrid_search.FastEmbedEmbeddings")
def test_dense_embedding_model_success(mock_dense_embed, embed_model, mock_logger):
    # Mock FastEmbedEmbeddings instance
    mock_instance = MagicMock()
    mock_dense_embed.return_value = mock_instance
    mock_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]

    texts = ["Test text 1", "Test text 2"]
    embeddings = embed_model.dense_embedding_model(texts)
    
    assert embeddings == [[0.1, 0.2, 0.3]]
    mock_dense_embed.assert_called_once_with(model_name="jinaai/jina-embeddings-v2-base-en")
    mock_instance.embed_documents.assert_called_once_with([texts])
    mock_logger.info.assert_called_once_with("Successfully Converted the text into Dense Vectors Using Model : test-model")

@patch("hybrid_rag.src.hybrid_search.FastEmbedEmbeddings")
def test_retrieval_embedding_model_success(mock_dense_embed, embed_model, mock_logger):
    # Mock FastEmbedEmbeddings instance
    mock_instance = MagicMock()
    mock_dense_embed.return_value = mock_instance

    embed_model_instance = embed_model.retrieval_embedding_model()
    
    assert embed_model_instance == mock_instance
    mock_dense_embed.assert_called_once_with(model_name="test-model")
    mock_logger.info.assert_called_once_with("Successfully Initialised FastEmbed retriever model: test-model")

@patch("hybrid_rag.src.hybrid_search.FastEmbedEmbeddings", side_effect=Exception("Embedding Error"))
def test_retrieval_embedding_model_failure(mock_dense_embed, embed_model, mock_logger):
    embed_model_instance = embed_model.retrieval_embedding_model()
    
    assert embed_model_instance is None
    mock_logger.error.assert_called_once_with(
        "Failed to Initialised LLM model Reason: Embedding Error -> TRACEBACK : "
    )
