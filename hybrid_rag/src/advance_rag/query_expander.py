"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
import traceback
from typing import List
from typing import Optional
import asyncio

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun

from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from hybrid_rag.src.utils.custom_utils import CustomMultiQueryRetriever
from hybrid_rag.src.utils.logutils import Logger
from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager

logger = Logger().get_logger()


class CustomQueryExpander:
    def __init__(
        self,
        collection_name: str,
        dense_search_params: dict,
        dense_embedding_model: str,
        llmModelInstance: LLMModelInitializer,
        vectorDbInstance:VectorStoreManager,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the CustomQueryExpander with LLM model and search parameters.

        Args:
            collection_name (str): The name of the Milvus vector database collection.
            dense_search_params (dict): Parameters for the dense search initialization.
            dense_embedding_model (str): The dense embedding model name supported by FastEmbedding.
            llmModelInstance (LLMModelInitializer): The LLM model initializer instance to initialize the LLM model.
            vectorDbInstance (VectorStoreManager): The vector store manager instance for initializing the vector store.
            logger (logging.Logger): Optional logger instance to log messages.

        Returns:
            None
        """
        self.logger = logger if logger else Logger().get_logger()
        self.dense_search_params = dense_search_params
        self.dense_embedding_model = dense_embedding_model
        self.collection_name = collection_name
        
        # Initialize the LLM model
        self.llm_model_instance = llmModelInstance.initialise_llm_model()

        # Initialize the vector store
        self.vector_store = vectorDbInstance.initialise_vector_store(
            "dense_vector",
            self.dense_search_params,
            self.dense_embedding_model,
            self.collection_name,
        )

        self.retriever_obj = CustomMultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(),  # default k value=4
            llm=self.llm_model_instance,
            include_original=True,
        )

        self.run_manager = AsyncCallbackManagerForRetrieverRun(
            run_id="example_run", handlers=[], inheritable_handlers={}
        )

    async def expand_query_async(self, question: str) -> List:
        """
        Expands the given question into multiple queries using the retriever.

        Args:
            question (str): The input question to be expanded.

        Returns:
            List[str]: A list of expanded queries generated from the input question.
        """
        try:
            multiq_queries = await self.retriever_obj.agenerate_queries(
                question, self.run_manager
            )
            self.logger.info("Successfully Executed the MultiQuery Retriever")
            return multiq_queries
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Generate Multiple Queries Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise
    
    def expand_query(self, question: str) -> List:
        """
        Synchronously expands the Question

        Args:
            question (str): The input question to be expanded.

        Returns:
            List[str]: A list of expanded queries generated from the input question.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already running an event loop, use run_coroutine_threadsafe
                self.logger.info("Running in an existing event loop.")
                future = asyncio.run_coroutine_threadsafe(
                    self.expand_query_async(question), loop
                )
                return future.result()
            else:
                # If no event loop is running, create a new one
                self.logger.info("Starting a new event loop.")
                return asyncio.run(self.expand_query_async(question))
        except Exception as e:
            self.logger.error(f"Error in async_rag: {e}")
            raise
