import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset
from langchain_core.documents import Document
from hybrid_rag.src.evaluate.rag_evaluation import RAGAEvaluator

@pytest.fixture
def mock_logger():
    """Fixture for a mocked logger."""
    return MagicMock()

@pytest.fixture
def rag_evaluator(mock_logger):
    """Fixture for initializing the RAGAEvaluator."""
    return RAGAEvaluator(
        llm_model_name="gpt-4",
        openai_api_base="https://api.openai.com/v1",
        groq_api_key="fake_api_key",
        dense_embedding_model="fake_dense_model",
        logger=mock_logger,
    )

@pytest.fixture
def sample_documents():
    """Fixture for creating sample documents."""
    return [
        Document(page_content="This is context 1."),
        Document(page_content="This is context 2."),
    ]

def test_initialization(rag_evaluator, mock_logger):
    assert rag_evaluator.llm_model_name == "gpt-4"
    assert rag_evaluator.openai_api_base == "https://api.openai.com/v1"
    assert rag_evaluator.__groq_api_key == "fake_api_key"
    assert rag_evaluator.dense_embedding_model == "fake_dense_model"
    mock_logger.info.assert_not_called()  # No logs should occur on init

def test_validate_column_dtypes_pass(rag_evaluator, mock_logger):
    dataset = Dataset.from_dict({
        "question": ["What is AI?"],
        "answer": ["AI is artificial intelligence."],
        "ground_truth": ["AI stands for artificial intelligence."],
        "contexts": [["AI enables automation."]],
    })
    
    result = rag_evaluator._validate_column_dtypes(dataset)
    assert result == "PASS"
    mock_logger.error.assert_not_called()

def test_validate_column_dtypes_fail(rag_evaluator, mock_logger):
    dataset = Dataset.from_dict({
        "question": ["What is AI?"],
        "answer": ["AI is artificial intelligence."],
        "contexts": ["AI enables automation."],  # Invalid type
    })
    
    result = rag_evaluator._validate_column_dtypes(dataset)
    assert result == "FAIL"
    mock_logger.error.assert_called_once_with(
        'Dataset feature "contexts" should be of type Sequence[string], but got <class \'datasets.features.features.Value\'> with feature type None'
    )

def test_prepare_context_for_ragas(rag_evaluator, sample_documents):
    result = rag_evaluator._prepare_context_for_ragas(sample_documents)
    assert result == [["This is context 1.", "This is context 2."]]

@patch("hybrid_rag.src.hybrid_search.evaluate")
@patch("hybrid_rag.src.hybrid_search.FastEmbedEmbeddings")
@patch("hybrid_rag.src.hybrid_search.ChatOpenAI")
def test_evaluate_rag_success(mock_chat_openai, mock_embeddings, mock_evaluate, rag_evaluator, sample_documents):
    # Mocking ChatOpenAI and evaluate
    mock_chat_openai.return_value = MagicMock()
    mock_embeddings.return_value = MagicMock()
    mock_evaluate.return_value = {"faithfulness": 0.85}

    question = ["What is AI?"]
    answer = ["AI is artificial intelligence."]
    result = rag_evaluator.evaluate_rag(question, answer, sample_documents)
    
    assert result == {"faithfulness": 0.85}
    mock_evaluate.assert_called_once()

@patch("hybrid_rag.src.hybrid_search.evaluate")
def test_evaluate_rag_dataset_validation_fail(mock_evaluate, rag_evaluator, sample_documents, mock_logger):
    question = ["What is AI?"]
    answer = ["AI is artificial intelligence."]
    
    # Invalid dataset (missing required column)
    with patch.object(rag_evaluator, "_validate_column_dtypes", return_value="FAIL"):
        with pytest.raises(ValueError, match="Dataset validation failed"):
            rag_evaluator.evaluate_rag(question, answer, sample_documents)
    
    mock_logger.error.assert_called_once_with(
        "Failed to evaluate RAG: Dataset validation failed -> TRACEBACK: "
    )
    mock_evaluate.assert_not_called()

@patch("hybrid_rag.src.hybrid_search.evaluate", side_effect=Exception("Evaluation Error"))
def test_evaluate_rag_exception(mock_evaluate, rag_evaluator, sample_documents, mock_logger):
    question = ["What is AI?"]
    answer = ["AI is artificial intelligence."]
    
    with pytest.raises(Exception, match="Evaluation Error"):
        rag_evaluator.evaluate_rag(question, answer, sample_documents)
    
    mock_logger.error.assert_called_with(
        "Failed to evaluate RAG: Evaluation Error -> TRACEBACK: "
    )
