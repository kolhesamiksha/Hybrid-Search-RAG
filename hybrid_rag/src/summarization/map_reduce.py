"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""
from langchain.chains.summarize import load_summarize_chain
import logging
from typing import Optional
from hybrid_rag.src.models import LLMModelInitializer
from hybrid_rag.src.prompts.summarization_prompt import SummarizationPrompt
from hybrid_rag.src.utils.logutils import Logger
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain

class MapReduceChain:
    """
    A class that encapsulates the Map-Reduce summarization chain using an LLM model.

    This class initializes a summarization chain with a specified language model and provides a method
    to generate summaries for a given set of documents using a Map-Reduce approach.

    Methods:
        __init__(llm: LLMModelInitializer, logger: Optional[logging.Logger] = None):
            Initializes the MapReduceChain with an LLM model and optional logger.
        
        generate_summary(docs: list) -> str:
            Generates a summary for the given documents using the Map-Reduce chain.
    """
    def __init__(self,
                 prompt_template: SummarizationPrompt,
                 llm: LLMModelInitializer,
                 logger: Optional[logging.Logger] = None,
                 ):
        """
        Initializes the MapReduceChain with a specified language model and an optional logger.

        Args:
            llm (LLMModelInitializer): An instance of the language model to be used for summarization.
            logger (Optional[logging.Logger]): A logger instance to log messages and errors. Defaults to a custom logger if not provided.
        """
        self.llm = llm.initialise_llm_model()
        self.map_prompt = prompt_template.generate_map_prompt()
        self.reduce_prompt = prompt_template.generate_reduce_prompt()
        self.logger = logger if logger else Logger().get_logger()

    def generate_summary(self, docs):
        """
        Generates a summary for the given documents using the map-reduce summarization chain.

        Args:
            docs (list): A list of documents or text chunks to be summarized.

        Returns:
            str: A string containing the generated summary.

        Raises:
            Exception: If an error occurs during the summary generation process, an empty string is returned.
        """
        chain = load_summarize_chain(self.llm, chain_type="map_reduce", map_prompt=self.map_prompt, combine_prompt=self.reduce_prompt)
        try:
            result = chain.invoke(docs)
            self.logger.info(f"Successfully Executed the Map Reduce chain, Generated Summary: {result}")
            return result.get('output_text', "")
            
        except Exception as e:
            self.logger.error(f"An error occurred while generating the summary: {e}")
            return ""


