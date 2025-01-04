import pytest
from unittest.mock import MagicMock, patch
from hybrid_rag.src.advance_rag.hybrid_search import MilvusHybridSearch
from hybrid_rag.src.models.retriever_model.models import EmbeddingModels
from langchain_core.documents import Document
from pymilvus import SearchResult

# Fixtures for reusable inputs
@pytest.fixture
def mock_vector_db():
    mock_instance = MagicMock()
    mock_instance.load_collection.return_value = MagicMock()
    return mock_instance

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def hybrid_search_instance(mock_vector_db, mock_logger):
    return MilvusHybridSearch(
        collection_name="test_collection",
        sparse_embedding_model="sparse_model",
        dense_embedding_model="dense_model",
        sparse_search_params={"param1": "value1"},
        dense_search_params={"param2": "value2"},
        vectorDbInstance=mock_vector_db,
        logger=mock_logger,
    )

def test_initialization(hybrid_search_instance, mock_vector_db):
    assert hybrid_search_instance.collection_name == "test_collection"
    assert hybrid_search_instance.sparse_search_params == {"param1": "value1"}
    assert hybrid_search_instance.dense_search_params == {"param2": "value2"}
    mock_vector_db.load_collection.assert_called_once_with("test_collection")

def test_collection_name_validation(hybrid_search_instance):
    with pytest.raises(ValueError):
        hybrid_search_instance.collection_name = ""  # Empty string

def test_sparse_search_params_validation(hybrid_search_instance):
    with pytest.raises(ValueError):
        hybrid_search_instance.sparse_search_params = "not_a_dict"  # Invalid type

def test_dense_search_params_validation(hybrid_search_instance):
    with pytest.raises(ValueError):
        hybrid_search_instance.dense_search_params = []  # Invalid type

@patch("hybrid_rag.src.hybrid_search.EmbeddingModels")
def test_generate_embeddings(mock_embedding_models, hybrid_search_instance):
    mock_model = MagicMock()
    mock_model.sparse_embedding_model.return_value = "sparse_emb"
    mock_model.dense_embedding_model.return_value = "dense_emb"
    mock_embedding_models.return_value = mock_model

    sparse_emb, dense_emb = hybrid_search_instance.generate_embeddings("test question")
    assert sparse_emb == "sparse_emb"
    assert dense_emb == "dense_emb"

def test_perform_search(hybrid_search_instance):
    hybrid_search_instance.milvus_collection.hybrid_search.return_value = [
        [SearchResult(entity={"text": "doc1"}), SearchResult(entity={"text": "doc2"})]
    ]
    res = hybrid_search_instance.perform_search("sparse_emb", "dense_emb", search_limit=2)
    assert len(res) == 1
    assert len(res[0]) == 2

def test_process_results(hybrid_search_instance):
    mock_results = [[
        SearchResult(entity={
            "text": "doc text",
            "source_link": "http://example.com",
            "author_name": "Author",
            "related_topics": ["topic1", "topic2"],
            "pdf_links": ["http://example.com/file.pdf"]
        })
    ]]
    output = hybrid_search_instance.process_results(mock_results)
    assert len(output) == 1
    doc = output[0]
    assert doc.page_content == "doc text"
    assert doc.metadata["source_link"] == "http://example.com"

@patch.object(MilvusHybridSearch, "generate_embeddings")
@patch.object(MilvusHybridSearch, "perform_search")
@patch.object(MilvusHybridSearch, "process_results")
def test_hybrid_search(mock_process_results, mock_perform_search, mock_generate_embeddings, hybrid_search_instance):
    mock_generate_embeddings.return_value = ("sparse_emb", "dense_emb")
    mock_perform_search.return_value = []
    mock_process_results.return_value = [Document(page_content="doc")]

    output = hybrid_search_instance.hybrid_search("test question", 3)
    mock_generate_embeddings.assert_called_once_with("test question")
    mock_perform_search.assert_called_once_with("sparse_emb", "dense_emb", 3)
    mock_process_results.assert_called_once_with([])
    assert len(output) == 1
    assert output[0].page_content == "doc"
