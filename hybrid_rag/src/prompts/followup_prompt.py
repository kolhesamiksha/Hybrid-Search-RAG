"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
import traceback
from typing import Optional

from langchain_core.prompts.prompt import PromptTemplate

from hybrid_rag.src.utils.logutils import Logger

class FollowupPromptGenerator:
    """
    A class to generate a structured support prompt for followup questions.

    Attributes:
        logger (logging.Logger): Logger instance for logging information or errors.
        LLAMA3_SYSTEM_TAG (str): The system tag for the Llama model.
        LLAMA3_USER_TAG (str): The user tag for the Llama model.
        LLAMA3_ASSISTANT_TAG (str): The assistant tag for the Llama model.
        MASTER_PROMPT (str): The master prompt that provides the base for the prompt template.

    Methods:
        __init__(llm_model_name: str, master_prompt: str, llama3_user_tag: str, llama3_system_tag: str, llama3_assistant_tag: str, logger: Optional[logging.Logger] = None):
            Initializes the SupportPromptGenerator with necessary system tags and master prompt.
        
        generate_prompt() -> PromptTemplate:
            Generates the QA prompt template using the initialized values.
            Returns:
                PromptTemplate: The generated QA prompt template.
    """

    def __init__(
        self,
        prompt_template: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the FollowupPrompt with necessary followup prompt template.

        Args:
            prompt_template (str): The prompt template used for the followup questions generation task.
            logger (Optional[logging.Logger]): Logger instance for logging. Defaults to None.

        Returns:
            None: Initializes the instance with the provided arguments.
        """

        self.logger = logger if logger else Logger().get_logger()
        self.PROMPT_TEMPLATE = prompt_template

    def generate_prompt(self) -> PromptTemplate:
        """
        Generates the QA prompt template using the initialized values.

        Args:
            None: This method uses the instance attributes to create the template.

        Returns:
            PromptTemplate: The generated QA prompt template that can be used in LLM tasks.
        """
        
        support_template = f"""
        {self.PROMPT_TEMPLATE}

        Please consider below knowldge as reference, don't generate any questions outside of below context.
        ---------------------
        QUESTION:
        {{question}}

        Response:
        {{response}}

        CONTEXT:
        {{context}}
        """

        try:
            follwup_prompt = PromptTemplate(
                template=support_template,
                input_variables=["question", "response", "context"],
            )
            self.logger.info("Successfully generated the Followup Prompt Template.")
            return follwup_prompt
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Create Prompt template Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise
