"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
import ast
import traceback
import re
from typing import Optional
from typing import List
from hybrid_rag.src.prompts.summarization_prompt import SummarizationPrompt
from hybrid_rag.src.models import LLMModelInitializer
from hybrid_rag.src.utils.logutils import Logger

from langchain.callbacks import get_openai_callback
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)

from langchain_core.output_parsers import BaseOutputParser

# Custom Query Expansion Class
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        """
        Parses the input text into a list of cleaned lines.

        Args:
            text (str): The input text to be parsed.

        Returns:
            List[str]: A list of cleaned lines with numerical prefixes removed.
        """
        lines = text.strip().split("\n")
        cleaned_lines = [re.sub(r"^\d+\.\s*", "", line) for line in lines]
        return cleaned_lines

class MetadataExtractor:
    """
    MetadataExtractor is a class that handles metadata extraction using a chain of prompts, LLM models, and output parsers.

    Methods:
        __init__(metadata_field: str, prompt_template: SummarizationPrompt, llm: LLMModelInitializer, logger: Optional[logging.Logger]):
            Initializes the MetadataExtractor with the required prompt template, LLM model, and logger.

        extractor(question: str, metadata_field: str) -> List[str]:
            Asynchronously extracts metadata fields by passing the input question and metadata field through the processing chain.
    """
    
    def __init__(self, prompt_template: SummarizationPrompt, llm: LLMModelInitializer, logger: Optional[logging.Logger] = None,):
        """
        Initializes the MetadataExtractor with the specified metadata field, prompt, LLM model, and logger.

        Args:
            metadata_field (str): The metadata field to be extracted.
            prompt_template (SummarizationPrompt): The prompt template used for metadata extraction.
            llm (LLMModelInitializer): The initialized LLM model to process the prompts.
            logger (Optional[logging.Logger]): Logger for logging information and errors. Defaults to a custom logger if not provided.
        """
        self.prompt = prompt_template.extract_metadata_prompt()
        self.llm_model = llm.initialise_llm_model()
        self.logger = logger if logger else Logger().get_logger()
    
    async def extractor(self, question: str, metadata_field: str) -> List[str]:
        """
        Asynchronously extracts metadata fields using a processing chain.

        Args:
            question (str): The input question to extract metadata for.
            metadata_field (str): The metadata field to be extracted.

        Returns:
            List[str]: A list of extracted metadata fields, or an error message in case of failure.

        Raises:
            Exception: Captures and logs any exceptions that occur during metadata extraction.
        """
        output_parser = LineListOutputParser()
        chain = (
            RunnableParallel(
                {
                    "metadata_field": RunnablePassthrough(),
                    "question": RunnablePassthrough(),
                }
            )
            | self.prompt
            | self.llm_model
            | output_parser
        )

        self.logger.info("Successfully Initialised the Metadata Extractor Chain.")
        try:
            # Use the OpenAI callback to monitor API usage.
            with get_openai_callback() as cb:
                response = await chain.ainvoke(
                    {
                        "metadata_field": metadata_field,
                        "question": question,
                    },
                    {"callbacks": [cb]},
                )
                self.logger.info(f"Successfully Extracted Metadata Fields: {response}")
                if isinstance(response, str):
                    response = ast.literal_eval(response)
                if isinstance(response[0], str):
                    response[0] = ast.literal_eval(response[0])
                return response[0]
    
        except Exception as e:
            self.logger.info(f"ERROR: {traceback.format_exc()}")
            return [str(e)]


