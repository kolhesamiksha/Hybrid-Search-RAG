"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""

#question & response 

#summarization support

#guardrails support: ingrained inside the module

import re
import logging
import traceback
from typing import Optional, List

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from hybrid_rag.src.utils.logutils import Logger
from hybrid_rag.src.models import LLMModelInitializer
from hybrid_rag.src.prompts.followup_prompt import (
    FollowupPromptGenerator,
)

# Custom Query Expansion Class
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        """
        Parses the given text to extract a list of lines, cleaning each line by removing leading numbers and spaces.

        Args:
            text (str): The input string containing multiple lines, where each line starts with a number followed by a period.

        Returns:
            List[str]: A list of cleaned strings, with leading numbers and spaces removed from each line.
        """
        lines = text.strip().split("\n")
        cleaned_lines = [re.sub(r"^\d+\.\s*", "", line) for line in lines]
        return cleaned_lines

class FollowupQGeneration:
    """
    A class for generating follow-up questions based on a given question, response, and context. It uses a language model 
    to process the input and generate follow-up questions.

    Methods:
        __init__(llm_model: LLMModelInitializer, followup_template: FollowupPromptGenerator, logger: Optional[logging.Logger] = None):
            Initializes the FollowupQGeneration class with the necessary LLM model and prompt generator.

        generate_followups(question: str, context: List[Document], response: str) -> List[str]:
            Generates a list of follow-up questions using the LLM model and the provided context and response.
    """
    def __init__(self, llm_model:LLMModelInitializer, followup_template: FollowupPromptGenerator, logger: Optional[logging.Logger] = None):
        """
        Initializes the FollowupQGeneration instance with the LLM model and follow-up prompt template.

        Args:
            llm_model (LLMModelInitializer): The LLM model used for generating follow-up questions.
            followup_template (FollowupPromptGenerator): The template used for generating the follow-up questions.
            logger (Optional[logging.Logger]): A logger instance for logging. Defaults to None.
        """
        self.llm_model = llm_model.initialise_llm_model()
        self.FOLLOWUP_PROMPT_TEMPLATE = followup_template.generate_prompt()
        self.logger = logger

    async def generate_followups(self, question: str, context: List[Document], response:str) -> List[str]:
        """
        Asynchronously generates follow-up questions based on the provided question, context, and response using 
        a language model.

        Args:
            question (str): The initial question that was asked.
            context (List[Document]): The context (documents) related to the question.
            response (str): The response generated for the question.

        Returns:
            List[str]: A list of follow-up questions generated based on the input.

        Raises:
            Exception: If an error occurs while generating follow-up questions, an exception is raised with error details.
        """
        try:

            #Rnnable Parallel Chaining reduces the chaining time by 30%
            output_parser = LineListOutputParser()
            chain = (
                RunnableParallel(
                    {
                        "response": RunnablePassthrough(),
                        "context": RunnablePassthrough(),
                        "question": RunnablePassthrough(),
                    }
                )
                | self.FOLLOWUP_PROMPT_TEMPLATE 
                | self.llm_model 
                | output_parser
            )

            result = await chain.ainvoke({"question": question, "context": context, "response": response})
            self.logger.info("Successfully Generated the Followup Questions!")
            return result
        except Exception as e:
            self.logger.info(f"Error while Generating the Followup question : {str(e)} traceback: {traceback.format_exc()}")
            raise


