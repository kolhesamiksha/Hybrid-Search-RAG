import os
import traceback
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager
from hybrid_rag.src.utils.logutils import Logger

logger = Logger().get_logger()

class SelfQueryRetrieval:
    def __init__(self, llm_model: str, dense_search_params: dict, dense_embedding_model: str, 
                 zillinz_cloud_uri: str, zillinz_cloud_api_key: str, collection_name: str, groq_api_key:str):
        """
        Initialize the SelfQueryRetrieval object with necessary parameters.
        
        :param llm_model: The LLM model name for initializing the LLM model.
        :param dense_search_params: Parameters for dense search.
        :param dense_embedding_model: Dense embedding model for vector initialization.
        :param zillinz_cloud_uri: Zilliz Cloud URI.
        :param zillinz_cloud_api_key: Zilliz Cloud API Key.
        :param collection_name: The collection name in the vector store.
        """
        self.llm_model = llm_model
        self.dense_search_params = dense_search_params
        self.dense_embedding_model = dense_embedding_model
        self.zillinz_cloud_uri = zillinz_cloud_uri
        self.__zillinz_cloud_api_key = zillinz_cloud_api_key
        self.__groq_api_key = groq_api_key
        self.collection_name = collection_name
        
        try:
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
            
            # Metadata field info for SelfQueryRetriever
            self.metadata_field_info = [
                AttributeInfo(
                    name="source_link",
                    description="Defines the source link of the file.",
                    type="string",
                ),
                AttributeInfo(
                    name="author_name",
                    description="The author of the file.",
                    type="string",
                ),
                AttributeInfo(
                    name="related_topics",
                    description="The topics related to the file.",
                    type="array",
                ),
                AttributeInfo(
                    name="pdf_links", 
                    description="The PDF links which contain extra information about the file.", 
                    type="array"
                ),
            ]
            self.document_content_description = "Brief summary of a file."
            
            # Initialize SelfQueryRetriever
            self.selfq_retriever = SelfQueryRetriever.from_llm(
                self.llm_model_instance, 
                self.vector_store, 
                self.document_content_description, 
                self.metadata_field_info, 
                verbose=True
            )
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to Initialize the parameters inside SelfQueryRetriever Constructor Class Reason: {error} -> TRACEBACK: {traceback.format_exc()}")
            raise

    def retrieve_query(self, question: str):
        """
        Retrieves the structured query and search arguments for the given question.
        
        :param question: The input question to process.
        :return: A tuple containing the new query and search arguments.
        """
        try:
            structured_query = self.selfq_retriever.query_constructor.invoke({"query": question})
            new_query, search_kwargs = self.selfq_retriever._prepare_query(question, structured_query)
            logger.info("Succesfully Executed the SelfQuery & generated metafieltering params and new query")
            return new_query, search_kwargs
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to Generate Metadata Fielters and New Query by SelfQuery Retrierver Reason: {error} -> TRACEBACK: {traceback.format_exc()}")
            raise  # Re-raise the exception after logging it
