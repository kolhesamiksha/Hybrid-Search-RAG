import os
from unittest.mock import patch, MagicMock
import pytest
from hybrid_rag.src.advance_rag.reranker import DocumentReranker
from langchain_core.documents import Document

# Mock environment variables based on .env.example
MOCK_ENV_VARS = {
    "LLM_MODEL_NAME": "llama-3.1-70b-versatile",
    "DENSE_EMBEDDING_MODEL": "jinaai/jina-embeddings-v2-base-en",
    "SPARSE_EMBEDDING_MODEL": "Qdrant/bm42-all-minilm-l6-v2-attentions",
    "GROQ_API_KEY": "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",
    "OPENAI_API_BASE": "https://api.aimlapi.com/v1",
    "CHAT_HISTORY": "2",
    "IS_EVALUATE": "True",
    "IS_RERANK": "False",
    "IS_GUARDRAIL": "False",
    "LANGCHAIN": "True",
    "QUESTION": "What is Generative AI?",
    "TEMPERATURE": "0.0",
    "TOP_P": "0.3",
    "FREQUENCY_PENALTY": "1.0",
    "ZILLIZ_CLOUD_URI": "https://in03-c2cc7c5da8decab.api.gcp-us-west1.zillizcloud.com",
    "ZILLIZ_CLOUD_API_KEY": "1a8a395165813d72085da55dde1db494f31fc95444ebb4f9e1f9e38127406c2ba82eb29f8b8e3d79c4a06618016e67ac77816ad9",
    "COLLECTION_NAME": "ey_data_1511",
    "HYBRID_SEARCH_TOPK": "6",
    "RERANK_TOPK": "3",
}

@pytest.fixture(scope="module", autouse=True)
def mock_env():
    with patch.dict(os.environ, MOCK_ENV_VARS):
        yield


@pytest.fixture
def mock_vector_db_instance():
    mock_instance = MagicMock()
    return mock_instance


@pytest.fixture
def document_reranker(mock_vector_db_instance):
    with patch("hybrid_rag.src.models.retriever_model.models.EmbeddingModels") as mock_embedding_cls, \
         patch("langchain_community.document_compressors.flashrank_rerank.FlashrankRerank") as mock_compressor_cls:
        mock_embedding_cls.return_value.retrieval_embedding_model.return_value = "mock_embedding_instance"
        mock_compressor_cls.return_value = MagicMock()
        return DocumentReranker(
            dense_embedding_model=os.getenv("DENSE_EMBEDDING_MODEL"),
            zillinz_cloud_uri=os.getenv("ZILLIZ_CLOUD_URI"),
            zillinz_cloud_api_key=os.getenv("ZILLIZ_CLOUD_API_KEY"),
            dense_search_params={"param1": "value1"},
            vectorDbInstance=mock_vector_db_instance,
        )

def test_initialization(document_reranker):
    assert document_reranker.dense_embedding_model == os.getenv("DENSE_EMBEDDING_MODEL")
    assert document_reranker.zillinz_cloud_uri == os.getenv("ZILLIZ_CLOUD_URI")
    assert document_reranker.embeddings == "mock_embedding_instance"
    assert document_reranker.compressor is not None

def test_rerank_docs(document_reranker, mock_vector_db_instance):
    with patch.object(document_reranker, "milvus_store_docs_to_rerank", return_value=MagicMock(as_retriever=MagicMock())) as mock_store, \
         patch("langchain.retrievers.ContextualCompressionRetriever.invoke", return_value=["compressed_doc1", "compressed_doc2"]) as mock_invoke:
        
        question = os.getenv("QUESTION")
        docs = [Document(page_content="doc1"), Document(page_content="doc2")]
        result = document_reranker.rerank_docs(question, docs, rerank_topk=int(os.getenv("RERANK_TOPK")))
        
        assert result == ["compressed_doc1", "compressed_doc2"]
        mock_store.assert_called_once_with(docs, document_reranker.dense_search_params)
        mock_invoke.assert_called_once_with(question)
        mock_vector_db_instance.drop_collection.assert_called_once_with("reranking_docs")

def test_env_variable_usage():
    assert os.getenv("DENSE_EMBEDDING_MODEL") == "jinaai/jina-embeddings-v2-base-en"
    assert os.getenv("RERANK_TOPK") == "3"
    assert os.getenv("QUESTION") == "What is Generative AI?"

