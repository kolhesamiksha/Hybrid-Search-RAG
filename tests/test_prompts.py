import pytest
from unittest.mock import MagicMock, patch
from hybrid_rag.src.prompts.prompt import SupportPromptGenerator
from langchain_core.prompts.prompt import PromptTemplate


@pytest.fixture
def mock_logger():
    """Fixture for the logger mock."""
    return MagicMock()


@pytest.fixture
def support_prompt_generator(mock_logger):
    """Fixture for initializing SupportPromptGenerator."""
    return SupportPromptGenerator(
        llm_model_name="llama_model",
        master_prompt="This is a master prompt.",
        llama3_user_tag="user_tag",
        llama3_system_tag="system_tag",
        llama3_assistant_tag="assistant_tag",
        logger=mock_logger,
    )


# Test for initializing the SupportPromptGenerator
def test_support_prompt_generator_initialization(mock_logger):
    support_prompt_generator = SupportPromptGenerator(
        llm_model_name="llama_model",
        master_prompt="Master prompt",
        llama3_user_tag="user_tag",
        llama3_system_tag="system_tag",
        llama3_assistant_tag="assistant_tag",
        logger=mock_logger,
    )
    assert isinstance(support_prompt_generator, SupportPromptGenerator)
    assert support_prompt_generator.LLAMA3_SYSTEM_TAG == "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    assert support_prompt_generator.LLAMA3_USER_TAG == "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    assert support_prompt_generator.LLAMA3_ASSISTANT_TAG == "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    assert support_prompt_generator.MASTER_PROMPT == "Master prompt"


# Test for generating prompt template with valid data
def test_generate_prompt(support_prompt_generator, mock_logger):
    # Call the generate_prompt method
    qa_prompt = support_prompt_generator.generate_prompt()

    # Assert that the returned object is of type PromptTemplate
    assert isinstance(qa_prompt, PromptTemplate)

    # Verify the logging call for success
    mock_logger.info.assert_called_once_with("Successfully generated the QA Prompt Template.")


# Test for generating prompt with missing parameters
def test_generate_prompt_failure(support_prompt_generator, mock_logger):
    # Simulate failure in generating the prompt template
    with patch("langchain_core.prompts.prompt.PromptTemplate") as mock_prompt_template:
        mock_prompt_template.side_effect = Exception("Error generating prompt.")

        with pytest.raises(Exception, match="Error generating prompt."):
            support_prompt_generator.generate_prompt()

        mock_logger.error.assert_called_once()
        assert "Failed to Create Prompt template" in str(mock_logger.error.call_args)
