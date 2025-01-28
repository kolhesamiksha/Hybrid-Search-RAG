"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""

# initialise llm model
import logging
from typing import Optional
import asyncio
from typing import List
from hybrid_rag.src.config import Config
from hybrid_rag.src.utils import Logger
from dotenv import load_dotenv

from langchain_core.documents import Document
from hybrid_rag.src.summarization.map_reduce import MapReduceChain
from hybrid_rag.src.prompts.summarization_prompt import SummarizationPrompt
from hybrid_rag.src.models import LLMModelInitializer
from hybrid_rag.src.vectordb import VectorStoreManager
from hybrid_rag.src.summarization.meta_extractor_chain import MetadataExtractor
from hybrid_rag.src.advance_rag import (
    MilvusHybridSearch,
)
from hybrid_rag.src.summarization.retrieve_docs_filter import MilvusMetaFiltering

class Summarization:
    """
    Summarization is a class that encapsulates the logic for extracting metadata, retrieving documents, and summarizing content using a map-reduce approach.

    Methods:
        _metadata_extractor(meta_field: str) -> List[str]:
            Asynchronously extracts metadata fields based on the specified meta_field.

        _map_reduce(docs: List[str]) -> str:
            Generates a summarized output using a map-reduce approach on the input documents.

        _summarize_chatbot_async(question: str) -> str:
            Asynchronously generates a summarized response for the input question.

        summarize_chatbot(question: str) -> str:
            Generates a summarized response for the input question, managing the event loop as needed.
    """
    def __init__(
        self, config: Config, logger: Optional[logging.Logger], **kwargs: dict
    ) -> None:
        """
        Initialize the Summarization class with configuration, logger, and optional parameters.

        Args:
            config (Config): Configuration object containing necessary parameters.
            logger (Optional[logging.Logger]): Logger instance for logging, optional.
            kwargs (dict): Additional keyword arguments.
        """
        self.config = config
        self.logger = logger if logger else Logger().get_logger()

        self.summary_prompt = SummarizationPrompt(
            self.config.META_FILTERING_PROMPT,
            self.config.MAP_PROMPT,
            self.config.REDUCE_PROMPT,
            self.logger,
        )

        self.llmModelInitializer = LLMModelInitializer(
            self.config.LLM_MODEL_NAME,
            self.config.PROVIDER_BASE_URL,
            self.config.LLM_API_KEY,
            self.config.TEMPERATURE,
            self.config.TOP_P,
            self.config.FREQUENCY_PENALTY,
            self.logger,
        )

        self.meta_filter_retriever = MilvusMetaFiltering(
            self.config.ZILLIZ_CLOUD_URI,
            self.config.ZILLIZ_CLOUD_API_KEY,
            self.config.COLLECTION_NAME,
            self.config.SELF_RAG_METADATA_ATTRIBUTES,
            self.logger,
        )

        self.map_reduce_chain = MapReduceChain(
            self.summary_prompt,
            self.llmModelInitializer, 
            self.logger,
        )

        self.meta_extractor_chain = MetadataExtractor(
            self.summary_prompt,
            self.llmModelInitializer,
            self.logger,
        )
        
        self.meta_field = self.config.METADATA_FILTERATION_FIELD
        self.batch_size = self.config.MILVUS_ITERATOR_BATCH_SIZE
        self.search_limit = self.config.ITERATIVE_SEARCH_LIMIT

    async def _metadata_extractor(self, question, meta_field) -> List[str]:
        """
        Asynchronously extract metadata fields using the metadata extractor chain.

        Args:
            meta_field (str): The field to extract metadata for.

        Returns:
            List[str]: A list of extracted metadata fields.
        """
        try:
            extraction_task = asyncio.create_task(self.meta_extractor_chain.extractor(question, meta_field))
            extracted_meta_fields = await extraction_task
            return extracted_meta_fields
        except Exception as e:
            self.logger.error(f"Error while extracting metadata for {meta_field}: {e}")
            raise
    
    def _map_reduce(self, docs):
        """
        Perform map-reduce summarization on the retrieved documents.

        Args:
            docs (List[str]): List of documents to summarize.

        Returns:
            str: Summarized text.
        """
        result = self.map_reduce_chain.generate_summary(docs)
        return result
    
    async def _summarize_chatbot_async(self, question):
        """
        Asynchronously summarize documents for a chatbot based on a question.

        Args:
            question (str): The input question to generate the summary for.

        Returns:
            str: Summarized output as a string.
        """
        # Step 1: Extract metadata topics
        topics_extracted = await self._metadata_extractor(question, self.meta_field)
        self.logger.info(f"Type of Topic or Metadata Extracted: {type(topics_extracted)}")
        self.logger.info(f"Topics Extracted : {topics_extracted}")
        
        # Step 2: Prepare metadata expression
        meta_expr = self.meta_filter_retriever.prepare_metadata_expression(self.meta_field, topics_extracted)

        # Step 3: Retrieve relevant documents
        retrieved_docs = []
        for doc in self.meta_filter_retriever.query_documents(meta_expr):
            document = Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {}))
            retrieved_docs.append(document)

        # Step 4: Generate the summary using map-reduce
        output_summary = self._map_reduce(retrieved_docs)
        return output_summary

    def summarize_chatbot(self, question: str):
        """
        Summarize chatbot response for a given question using an event loop.

        Args:
            question (str): The input question to generate the summary for.

        Returns:
            str: Summarized output as a string.
        """
        # profiler = cProfile.Profile()
        # profiler.enable()
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # In an existing event loop, schedule the coroutine as a task and wait for it
            future = asyncio.run_coroutine_threadsafe(
                self._summarize_chatbot_async(question), loop
            )
            result = future.result() 
        else:
            # Otherwise, run a new event loop
            result = asyncio.run(self._summarize_chatbot_async(question))
        
        # profiler.disable()
        # profiler.print_stats(sort="time")
        return result