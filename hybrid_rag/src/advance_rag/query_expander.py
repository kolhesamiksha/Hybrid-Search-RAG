import os
import traceback
from typing import List
from hybrid_rag.src.utils.custom_utils import CustomMultiQueryRetriever, CallbackManagerForRetrieverRun
from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager
from hybrid_rag.src.utils.logutils import Logger

logger = Logger().get_logger()

class CustomQueryExpander:
    def __init__(self, llm_model: str, dense_search_params: dict, dense_embedding_model: str, 
                 zillinz_cloud_uri: str, zillinz_cloud_api_key: str, collection_name: str, groq_api_key:str):
        """
        Initialize the CustomQueryExpander with LLM model and search parameters.
        
        :param llm_model: The LLM model to be used for query expansion.
        :param dense_search_params: Parameters for dense search initialization.
        """
        self.llm_model = llm_model
        self.dense_search_params = dense_search_params
        self.dense_embedding_model = dense_embedding_model
        self.zillinz_cloud_uri = zillinz_cloud_uri
        self.__zillinz_cloud_api_key = zillinz_cloud_api_key
        self.__groq_api_key = groq_api_key
        self.collection_name = collection_name
        
        # Initialize the LLM model
        llmModelInitializer = LLMModelInitializer(self.llm_model, self.__groq_api_key)
        self.llm_model_instance = llmModelInitializer.initialise_llm_model()
        
        vectorStoreManager = VectorStoreManager(self.zillinz_cloud_uri, self.__zillinz_cloud_api_key)
        
        # Initialize the vector store
        self.vector_store = vectorStoreManager.initialise_vector_store(
            "dense_vector", 
            self.dense_search_params, 
            self.dense_embedding_model,
            self.collection_name
        )

        self.retriever_obj = CustomMultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(),
            llm=self.llm_model_instance,
            include_original=True
        )
        
        self.run_manager = CallbackManagerForRetrieverRun(run_id="example_run", handlers=[], inheritable_handlers={})

    def expand_query(self, question: str) -> List:
        """
        Expands the given question into multiple queries using the retriever.
        
        :param question: The input question to be expanded.
        :return: A list of expanded queries.
        """
        try:
            multiq_queries = self.retriever_obj.generate_queries(question, self.run_manager)
            logger.info("Successfully Executed the MultiQuery Retriever")
            return multiq_queries
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to Generate Multiple Queries Reason: {error} -> TRACEBACK : {traceback.format_exc()}")
