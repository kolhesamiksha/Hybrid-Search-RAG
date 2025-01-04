"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import traceback
from typing import Optional
import logging

from hybrid_rag.src.utils.logutils import Logger
from langchain_core.prompts.prompt import PromptTemplate


class SupportPromptGenerator:
    """
    A class to generate a structured support prompt for QA systems.
    """

    def __init__(
        self,
        llm_model_name: str,
        master_prompt: str,
        llama3_user_tag: str,
        llama3_system_tag: str,
        llama3_assistant_tag: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the SupportPromptGenerator with necessary system tags and master prompt.
        """

        self.logger = logger if logger else Logger().get_logger()

        if "llama" in llm_model_name and not llama3_system_tag:
            self.LLAMA3_SYSTEM_TAG = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            )
        else:
            self.LLAMA3_SYSTEM_TAG = llama3_system_tag

        if "llama" in llm_model_name and not llama3_user_tag:
            self.LLAMA3_USER_TAG = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        else:
            self.LLAMA3_USER_TAG = llama3_user_tag

        if "llama" in llm_model_name and not llama3_assistant_tag:
            self.LLAMA3_ASSISTANT_TAG = (
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            )
        else:
            self.LLAMA3_ASSISTANT_TAG = llama3_assistant_tag

        self.MASTER_PROMPT = master_prompt

    def generate_prompt(self) -> PromptTemplate:
        """
        Generates the QA prompt template using the initialized values.

        :return: A PromptTemplate instance for QA tasks.
        """
        support_template = f"""
        {self.LLAMA3_SYSTEM_TAG}
        {self.MASTER_PROMPT}
        {self.LLAMA3_USER_TAG}

        Use the following context to answer the question.
        CONTEXT:
        {{context}}

        CHAT HISTORY:
        {{chat_history}}

        Question: {{question}}
        {self.LLAMA3_ASSISTANT_TAG}
        """

        try:
            qa_prompt = PromptTemplate(
                template=support_template,
                input_variables=["context", "chat_history", "question"],
            )
            self.logger.info("Successfully generated the QA Prompt Template.")
            return qa_prompt
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Create Prompt template Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise
