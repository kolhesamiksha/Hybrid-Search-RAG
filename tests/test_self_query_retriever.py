import os
from unittest.mock import MagicMock, patch
import pytest
from hybrid_rag.src.advance_rag.self_query_retriever import SelfQueryRetrieval

# Mock environment variables based on your .env.example file
MOCK_ENV_VARS = {
    "DENSE_EMBEDDING_MODEL": "jinaai/jina-embeddings-v2-base-en",
    "COLLECTION_NAME": "ey_data_1511",
    "HYBRID_SEARCH_TOPK": "6",
    "RERANK_TOPK": "3",
}

@pytest.fixture(scope="module", autouse=True)
def mock_env():
    with patch.dict(os.environ, MOCK_ENV_VARS):
        yield


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    return logger


@pytest.fixture
def mock_llm_model_initializer():
    mock_instance = MagicMock()
    mock_instance.initialise_llm_model.return_value = MagicMock()
    return mock_instance


@pytest.fixture
def mock_vector_db_initializer():
    mock_instance = MagicMock()
    mock_instance.initialise_vector_store.return_value = MagicMock()
    return mock_instance


@pytest.fixture
def self_query_retrieval(mock_logger, mock_llm_model_initializer, mock_vector_db_initializer):
    dense_search_params = {"metric": "L2", "nlist": 128}
    return SelfQueryRetrieval(
        collection_name=os.getenv("COLLECTION_NAME"),
        dense_search_params=dense_search_params,
        dense_embedding_model=os.getenv("DENSE_EMBEDDING_MODEL"),
        llmModelInstance=mock_llm_model_initializer,
        vectorDbInstance=mock_vector_db_initializer,
        logger=mock_logger,
    )

def test_initialization(self_query_retrieval, mock_logger):
    assert self_query_retrieval.collection_name == os.getenv("COLLECTION_NAME")
    assert self_query_retrieval.dense_embedding_model == os.getenv("DENSE_EMBEDDING_MODEL")
    assert self_query_retrieval.llm_model_instance is not None
    assert self_query_retrieval.vector_store is not None
    mock_logger.info.assert_any_call(f"Initialized the SelfQueryRetrieval object successfully.")

def test_retrieve_query(self_query_retrieval, mock_logger):
    question = "What is Generative AI?"
    
    # Mock the behavior of query and filtering methods
    self_query_retrieval.selfq_retriever.query_constructor.invoke.return_value = "structured_query_mock"
    self_query_retrieval.selfq_retriever._prepare_query.return_value = ("new_query_mock", {"filters": "mock_filters"})
    
    new_query, search_kwargs = self_query_retrieval.retrieve_query(question)
    
    assert new_query == "new_query_mock"
    assert search_kwargs == {"filters": "mock_filters"}
    mock_logger.info.assert_called_with(
        "Succesfully Executed the SelfQuery & generated metafieltering params and new query"
    )

def test_initialization_exception(mock_logger, mock_llm_model_initializer, mock_vector_db_initializer):
    mock_vector_db_initializer.initialise_vector_store.side_effect = Exception("VectorStore Initialization Error")
    
    dense_search_params = {"metric": "L2", "nlist": 128}
    with pytest.raises(Exception, match="VectorStore Initialization Error"):
        SelfQueryRetrieval(
            collection_name=os.getenv("COLLECTION_NAME"),
            dense_search_params=dense_search_params,
            dense_embedding_model=os.getenv("DENSE_EMBEDDING_MODEL"),
            llmModelInstance=mock_llm_model_initializer,
            vectorDbInstance=mock_vector_db_initializer,
            logger=mock_logger,
        )
    mock_logger.error.assert_called_once()

def test_retrieve_query_exception(self_query_retrieval, mock_logger):
    question = "Invalid Question"
    
    # Simulate an exception in the retrieve_query process
    self_query_retrieval.selfq_retriever.query_constructor.invoke.side_effect = Exception("Query Constructor Error")
    
    with pytest.raises(Exception, match="Query Constructor Error"):
        self_query_retrieval.retrieve_query(question)
    
    mock_logger.error.assert_called_with(
        "Failed to Generate Metadata Fielters and New Query by SelfQuery Retrierver Reason: Query Constructor Error -> TRACEBACK: "
    )

def test_metadata_field_info_and_description(self_query_retrieval):
    metadata_field_info = self_query_retrieval.metadata_field_info
    
    assert len(metadata_field_info) == 4
    assert metadata_field_info[0].name == "source_link"
    assert metadata_field_info[0].description == "Defines the source link of the file."
    assert metadata_field_info[1].name == "author_name"
    assert metadata_field_info[1].description == "The author of the file."
    assert self_query_retrieval.document_content_description == "Brief summary of a file."

