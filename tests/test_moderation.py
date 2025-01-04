import pytest
from unittest.mock import MagicMock, patch
from hybrid_rag.src.moderation.question_moderator import QuestionModerator
from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


@pytest.fixture
def mock_llm_model_initializer():
    """Fixture for LLMModelInitializer mock."""
    llm_model_mock = MagicMock(spec=LLMModelInitializer)
    return llm_model_mock


@pytest.fixture
def question_moderator(mock_llm_model_initializer):
    """Fixture for initializing QuestionModerator."""
    return QuestionModerator(llmModelInstance=mock_llm_model_initializer)


@pytest.fixture
def mock_chain():
    """Fixture for mocking the chain."""
    return MagicMock()


@pytest.fixture
def mock_response():
    """Fixture for mocking the response."""
    return "Moderated Content Detected"


# Test for initializing the QuestionModerator
def test_question_moderator_initialization(mock_llm_model_initializer):
    moderator = QuestionModerator(llmModelInstance=mock_llm_model_initializer)
    assert isinstance(moderator, QuestionModerator)
    assert moderator.llmModelInstance == mock_llm_model_initializer


# Test for detect method with successful moderation detection
@patch("hybrid_rag.src.models.moderation.RunnablePassthrough")
@patch("hybrid_rag.src.models.moderation.PromptTemplate")
def test_detect_moderated_content(mock_prompt_template, mock_runnable_passthrough, mock_llm_model_initializer, mock_response, question_moderator):
    # Mock the LLM model's behavior
    mock_llm_model_initializer.initialise_llm_model.return_value = MagicMock()

    # Mock the PromptTemplate and RunnablePassthrough behavior
    mock_prompt_instance = MagicMock()
    mock_prompt_template.return_value = mock_prompt_instance
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    # Override the method in question_moderator
    question_moderator.llm_model_instance = mock_chain

    question = "What is AI?"
    moderation_prompt = "Please detect if the question has any harmful content."

    # Test the detect method
    response = question_moderator.detect(question, moderation_prompt)

    assert response == mock_response
    mock_chain.invoke.assert_called_once_with(
        {"question": question, "QUESTION_MODERATION_PROMPT": moderation_prompt}
    )


# Test for detect method failure
@patch("hybrid_rag.src.models.moderation.RunnablePassthrough")
@patch("hybrid_rag.src.models.moderation.PromptTemplate")
def test_detect_moderated_content_failure(mock_prompt_template, mock_runnable_passthrough, mock_llm_model_initializer, question_moderator):
    # Mock failure in the chain
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("Failed to process the question.")

    # Override the method in question_moderator
    question_moderator.llm_model_instance = mock_chain

    question = "What is AI?"
    moderation_prompt = "Please detect if the question has any harmful content."

    # Test the detect method to raise a RuntimeError
    with pytest.raises(RuntimeError, match="Failed to detect moderated content"):
        question_moderator.detect(question, moderation_prompt)

