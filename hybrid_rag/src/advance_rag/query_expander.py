"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import traceback
from typing import List
from typing import Optional
import logging

from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from hybrid_rag.src.utils.custom_utils import CustomMultiQueryRetriever
from hybrid_rag.src.utils.logutils import Logger
from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logger = Logger().get_logger()


class CustomQueryExpander:
    def __init__(
        self,
        collection_name: str,
        dense_search_params: dict,
        dense_embedding_model: str,
        llmModelInstance: LLMModelInitializer,
        vectorDbInstance=VectorStoreManager,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the CustomQueryExpander with LLM model and search parameters.

        :param collection_name: Milvus vectordb collection name
        :param dense_search_params: Parameters for dense search initialization.
        :param dense_search_params: Parameters for dense search initialization.
        :param llmModelInstance: LLMModelInitializer class instance, class object as parameter
        :param vectorDbInstance: VectorStoreManager class instance, lass object as parameter
        """
        self.logger = logger if logger else Logger().get_logger()
        self.dense_search_params = dense_search_params
        self.dense_embedding_model = dense_embedding_model
        # self.zillinz_cloud_uri = zillinz_cloud_uri
        # self.__zillinz_cloud_api_key = zillinz_cloud_api_key
        # self.__groq_api_key = groq_api_key
        self.collection_name = collection_name
        self.llmModelInstance = llmModelInstance
        self.vectorDbInstance = vectorDbInstance

        # Initialize the LLM model
        # llmModelInitializer = LLMModelInitializer(self.llm_model, self.__groq_api_key)
        llmModelInitializer = self.llmModelInstance
        self.llm_model_instance = llmModelInitializer.initialise_llm_model()

        # vectorStoreManager = VectorStoreManager(self.zillinz_cloud_uri, self.__zillinz_cloud_api_key)
        vectorDbInitializer = self.vectorDbInstance
        # Initialize the vector store
        self.vector_store = vectorDbInitializer.initialise_vector_store(
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

        self.run_manager = CallbackManagerForRetrieverRun(
            run_id="example_run", handlers=[], inheritable_handlers={}
        )

    def expand_query(self, question: str) -> List:
        """
        Expands the given question into multiple queries using the retriever.

        :param question: The input question to be expanded.
        :return: A list of expanded queries.
        """
        try:
            multiq_queries = self.retriever_obj.generate_queries(
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
