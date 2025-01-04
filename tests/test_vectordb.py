import pytest
from unittest.mock import MagicMock, patch
from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager
from pymilvus import Collection, utility


@pytest.fixture
def mock_logger():
    """Fixture for the logger mock."""
    return MagicMock()


@pytest.fixture
def vector_store_manager(mock_logger):
    """Fixture for initializing VectorStoreManager."""
    return VectorStoreManager(
        zilliz_cloud_uri="your_zilliz_cloud_uri",
        zilliz_cloud_api_key="your_zilliz_cloud_api_key",
        logger=mock_logger,
    )


# Test for initializing the VectorStoreManager
def test_vector_store_manager_initialization(mock_logger):
    manager = vector_store_manager(mock_logger)
    assert isinstance(manager, VectorStoreManager)
    assert manager.zilliz_cloud_uri == "your_zilliz_cloud_uri"
    assert manager._VectorStoreManager__zilliz_cloud_api_key == "your_zilliz_cloud_api_key"


# Test for successful collection loading
def test_load_collection_success(vector_store_manager, mock_logger):
    with patch("pymilvus.Collection.load") as mock_load:
        mock_load.return_value = None  # Mock successful collection load
        collection_name = "test_collection"
        collection = vector_store_manager.load_collection(collection_name)
        assert isinstance(collection, Collection)
        mock_logger.info.assert_called_once_with(f"Collection {collection_name} loaded successfully.")


# Test for failed collection loading
def test_load_collection_failure(vector_store_manager, mock_logger):
    with patch("pymilvus.Collection.load", side_effect=Exception("Load error")):
        with pytest.raises(Exception):
            vector_store_manager.load_collection("test_collection")
        mock_logger.error.assert_called_once_with("Failed to load collection test_collection: Load error")


# Test for successful collection dropping
def test_drop_collection_success(vector_store_manager, mock_logger):
    with patch("pymilvus.utility.drop_collection") as mock_drop:
        mock_drop.return_value = None  # Mock successful collection drop
        vector_store_manager.drop_collection("test_collection")
        mock_logger.info.assert_called_once_with("Collection test_collection dropped successfully.")


# Test for failed collection dropping
def test_drop_collection_failure(vector_store_manager, mock_logger):
    with patch("pymilvus.utility.drop_collection", side_effect=Exception("Drop error")):
        with pytest.raises(Exception):
            vector_store_manager.drop_collection("test_collection")
        mock_logger.error.assert_called_once_with("Failed to drop collection test_collection: Drop error")


# Test for successful vector store initialization
def test_initialise_vector_store_success(vector_store_manager, mock_logger):
    with patch("langchain_community.vectorstores.Milvus") as mock_milvus:
        mock_milvus.return_value = MagicMock()  # Mock Milvus initialization
        vector_store = vector_store_manager.initialise_vector_store(
            "vector_field", {"search_method": "cosine"}, "dense_embedding_model", "test_collection"
        )
        assert vector_store is not None
        mock_logger.info.assert_called_once_with(
            "Vector store for collection test_collection initialized successfully."
        )


# Test for failed vector store initialization
def test_initialise_vector_store_failure(vector_store_manager, mock_logger):
    with patch("langchain_community.vectorstores.Milvus", side_effect=Exception("Initialization error")):
        with pytest.raises(Exception):
            vector_store_manager.initialise_vector_store(
                "vector_field", {"search_method": "cosine"}, "dense_embedding_model", "test_collection"
            )
        mock_logger.error.assert_called_once_with(
            "Failed to initialize vector store for collection test_collection: Initialization error"
        )
