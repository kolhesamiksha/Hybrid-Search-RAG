"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""

## Write all necessary summarization prompts
import logging
import traceback
from typing import Optional

from langchain_core.prompts.prompt import PromptTemplate

from hybrid_rag.src.utils.logutils import Logger

class SummarizationPrompt:
    """
        A class that handles the creation of summarization prompt templates used for generating summaries 
        using a language model. It provides methods to generate summary prompts and map-reduce prompts.

        Methods:
            __init__(summary_prompt: str, map_reduce_prompt: str, logger: Optional[logging.Logger] = None):
                Initializes the SummarizationPrompt with the necessary summary and map-reduce prompt templates.

            generate_summary_prompt() -> PromptTemplate:
                Generates and returns the summary prompt template.

            generate_map_reduce_prompt() -> PromptTemplate:
                Generates and returns the map-reduce prompt template.
    """
    def __init__(
        self,
        summary_prompt: str,
        map_prompt: str,
        reduce_prompt: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the SummarizationPrompt with the provided summary and map-reduce prompt templates.

        Args:
            summary_prompt (str): The prompt template used for summary generation.
            map_reduce_prompt (str): The prompt template used for map-reduce summarization.
            logger (Optional[logging.Logger]): A logger instance to log messages. Defaults to None.
        
        Returns:
            None: Initializes the instance with the provided arguments.
        """
    
        self.logger = logger if logger else Logger().get_logger()
        self.SUMMARY_PROMPT_TEMPLATE = summary_prompt
        self.MAP_PROMPT_TEMPLATE = map_prompt
        self.REDUCE_PROMPT_TEMPLATE = reduce_prompt

    def extract_metadata_prompt(self) -> PromptTemplate:
        """
        Generates the summary prompt template using the provided summary prompt template and input variables.

        Args:
            None: This method uses the initialized summary prompt template to create the prompt.

        Returns:
            PromptTemplate: The generated summary prompt template, which can be used in a summarization system.
        
        Raises:
            Exception: If an error occurs during prompt template generation, an exception is raised with an error message.
        """

        support_template = f"""
        {self.SUMMARY_PROMPT_TEMPLATE}

        Please consider below knowldge as reference, don't generate any questions outside of below context.
        ---------------------
        QUESTION:
        {{question}}

        METADATA_FIELD:
        {{metadata_field}}
        """

        try:
            extractor_prompt = PromptTemplate(
                template=support_template,
                input_variables=["question", "metadata_field"],
            )
            self.logger.info("Successfully generated the Followup Prompt Template.")
            return extractor_prompt
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Create Prompt template Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise
    
    def generate_map_prompt(self) -> PromptTemplate:
        """
        Generates the map prompt template using the provided map prompt template and input variables.

        Args:
            None: This method uses the initialized map prompt template to create the prompt.

        Returns:
            PromptTemplate: The generated map prompt template, which can be used in map LLM tasks.
        
        Raises:
            Exception: If an error occurs during prompt template generation, an exception is raised with an error message.
        
        """

        support_template = f"""
        {self.MAP_PROMPT_TEMPLATE}

        Please consider below set of documents:
        ---------------------
        DOCUMENTS_TO_SUMMARIZE:
        {{text}}

        \nHelpful Answer:
        """

        try:
            map_prompt = PromptTemplate(
                template=support_template,
                input_variables=["text"],
            )
            self.logger.info("Successfully generated the Map Prompt Template.")
            return map_prompt
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Create Prompt template Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise
    
    def generate_reduce_prompt(self) -> PromptTemplate:
        """
        Generates the reduce prompt template using the provided reduce prompt template and input variables.

        Args:
            None: This method uses the initialized reduce prompt template to create the prompt.

        Returns:
            PromptTemplate: The generated reduce prompt template, which can be used in reduce LLM tasks.
        
        Raises:
            Exception: If an error occurs during prompt template generation, an exception is raised with an error message.
        
        """

        support_template = f"""
        {self.REDUCE_PROMPT_TEMPLATE}

        Please consider below set of summaries:
        ---------------------
        SUMMARIES:
        {{text}}

        \nHelpful Answer:
        """

        try:
            reduce_prompt = PromptTemplate(
                template=support_template,
                input_variables=["text"],
            )
            self.logger.info("Successfully generated the Reduce Prompt Template.")
            return reduce_prompt
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Create Prompt template Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise