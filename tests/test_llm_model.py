import pytest
from unittest.mock import patch, MagicMock
from hybrid_rag.src.models.llm_model.model import LLMModelInitializer

@pytest.fixture
def mock_logger():
    """Fixture for a mocked logger."""
    return MagicMock()

@pytest.fixture
def llm_initializer(mock_logger):
    """Fixture for initializing the LLMModelInitializer."""
    return LLMModelInitializer(
        llm_model_name="test-model",
        groq_api_key="test-api-key",
        temperature=0.5,
        top_p=0.7,
        frequency_penalty=0.2,
        logger=mock_logger,
    )

def test_initialization(llm_initializer, mock_logger):
    assert llm_initializer.llm_model_name == "test-model"
    assert llm_initializer.groq_api_key == "test-api-key"
    assert llm_initializer.temperature == 0.5
    assert llm_initializer.top_p == 0.7
    assert llm_initializer.frequency_penalty == 0.2
    assert llm_initializer.llm_model is None
    mock_logger.info.assert_not_called()  # Logger shouldn't log on initialization

def test_groq_api_key_setter(llm_initializer):
    with pytest.raises(ValueError, match="groq_api_key must be a non-empty string."):
        llm_initializer.groq_api_key = ""  # Invalid key
    llm_initializer.groq_api_key = "new-api-key"  # Valid key
    assert llm_initializer.groq_api_key == "new-api-key"

def test_temperature_setter(llm_initializer):
    with pytest.raises(ValueError, match="temperature must be a float in the range \\[0.0, 1.0\\]."):
        llm_initializer.temperature = -0.1  # Invalid value
    with pytest.raises(ValueError):
        llm_initializer.temperature = 1.1  # Invalid value
    llm_initializer.temperature = 0.3  # Valid value
    assert llm_initializer.temperature == 0.3

def test_top_p_setter(llm_initializer):
    with pytest.raises(ValueError, match="top_p must be a float in the range \\[0.0, 1.0\\]."):
        llm_initializer.top_p = -0.5  # Invalid value
    with pytest.raises(ValueError):
        llm_initializer.top_p = 1.5  # Invalid value
    llm_initializer.top_p = 0.9  # Valid value
    assert llm_initializer.top_p == 0.9

def test_frequency_penalty_setter(llm_initializer):
    with pytest.raises(ValueError, match="frequency_penalty must be a float in the range \\[-2.0, 2.0\\]."):
        llm_initializer.frequency_penalty = -3.0  # Invalid value
    with pytest.raises(ValueError):
        llm_initializer.frequency_penalty = 2.5  # Invalid value
    llm_initializer.frequency_penalty = 1.5  # Valid value
    assert llm_initializer.frequency_penalty == 1.5

@patch("hybrid_rag.src.hybrid_search.ChatGroq")
def test_initialise_llm_model_success(mock_chatgroq, llm_initializer, mock_logger):
    # Mock ChatGroq instance
    mock_instance = MagicMock()
    mock_chatgroq.return_value = mock_instance
    
    llm_model = llm_initializer.initialise_llm_model()
    assert llm_model == mock_instance
    mock_chatgroq.assert_called_once_with(
        model="test-model",
        api_key="test-api-key",
        temperature=0.5,
        top_p=0.7,
        frequency_penalty=0.2,
        max_retries=2,
    )
    mock_logger.info.assert_called_once_with("Successfully Initialized the LLM model")

@patch("hybrid_rag.src.hybrid_search.ChatGroq", side_effect=Exception("Initialization Error"))
def test_initialise_llm_model_failure(mock_chatgroq, llm_initializer, mock_logger):
    llm_model = llm_initializer.initialise_llm_model()
    assert llm_model is None
    mock_logger.error.assert_called_once_with(
        "Failed to Initialize LLM model. Reason: Initialization Error -> TRACEBACK: "
    )

