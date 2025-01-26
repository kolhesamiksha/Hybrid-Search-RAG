"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
import traceback
from typing import Optional
from typing import List
from typing import Dict
import asyncio

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from hybrid_rag.src.utils.logutils import Logger
from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager

class SelfQueryRetrieval:
    def __init__(
        self,
        collection_name: str,
        dense_search_params: dict,
        dense_embedding_model: str,
        metadata_attributes: List[Dict[str,str]],
        document_info: str,
        llmModelInstance: LLMModelInitializer,
        vectorDbInstance: VectorStoreManager,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the SelfQueryRetrieval instance with necessary parameters.

        Args:
            collection_name (str): The collection name in the vector store.
            dense_search_params (Dict[str, Any]): Parameters for dense search.
            dense_embedding_model (str): Dense embedding model for vector initialization.
            metadata_attributes (List[Dict[str, str]]): Metadata field attributes containing 'name', 'description', and 'type'.
            document_info (str): Description of the document's content.
            llmModelInstance (LLMModelInitializer): LLM model instance of the LLMModelInitializer class.
            vectorDbInstance (VectorStoreManager): VectorStoreManager class instance to manage the vector store.
            logger (Optional[logging.Logger]): Optional logger instance to log messages.

        Returns:
            None: Initializes the object with the provided parameters, no return value.
        """
        # self.llm_model = llm_model
        self.logger = logger if logger else Logger().get_logger()
        self.dense_search_params = dense_search_params
        self.dense_embedding_model = dense_embedding_model
        # self.__groq_api_key = groq_api_key
        self.collection_name = collection_name

        try:
            self.llm_model_instance = llmModelInstance.initialise_llm_model()
            # vectorStoreManager = VectorStoreManager(self.zillinz_cloud_uri, self.__zillinz_cloud_api_key)
            # Initialize the vector store
            self.vector_store = vectorDbInstance.initialise_vector_store(
                "dense_vector",
                self.dense_search_params,
                self.dense_embedding_model,
                self.collection_name,
            )

            # Metadata field info for SelfQueryRetriever
            self.metadata_field_info = self._create_attribute_info(metadata_attributes)
            self.document_content_description = document_info

            # Initialize SelfQueryRetriever
            self.selfq_retriever = SelfQueryRetriever.from_llm(
                llm=self.llm_model_instance,
                vectorstore=self.vector_store,
                document_contents=self.document_content_description,
                metadata_field_info=self.metadata_field_info,
                verbose=True,
            )
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Initialize the parameters inside SelfQueryRetriever Constructor Class Reason: {error} -> TRACEBACK: {traceback.format_exc()}"
            )
            raise

    def _create_attribute_info(self, attributes: List[Dict[str, str]]) -> List[AttributeInfo]: 
        attribute_info_list = []
        try:
            for attribute in attributes:
                
                # Extract name, description, and type from each dictionary
                name = attribute.get("name", "name")
                description = attribute.get("description", "description")
                type_ = attribute.get("type", "description")

                # Check if all required keys are present
                if not name or not description or not type_:
                    raise ValueError(f"Each dictionary must contain 'name', 'description', and 'type' keys. Missing data: {attribute}")
                
                # Create AttributeInfo and append to the list
                attribute_info_list.append(AttributeInfo(name=name, description=description, type=type_))
                self.logger.info(f"Successfully extracted metadata attributes from List of dict to list {attribute_info_list}")
        except Exception as e:
            self.logger.error("Error while converting the metadata to a list")
        
        return attribute_info_list                
    
    async def retrieve_query_async(self, question: str) -> tuple[str, dict]:
        """
        Retrieves the structured query and search arguments for the given question.

        Args:
            question (str): The input question to process.

        Returns:
            tuple:
                - A string representing the new query.
                - A dictionary containing the metadata parameters to use post_filtering of metadata for milvus search.
        """
        try:
            structured_query = self.selfq_retriever.query_constructor.invoke(
                {"query": question}
            )
            new_query, search_kwargs = self.selfq_retriever._prepare_query(
                question, structured_query
            )
            self.logger.info(
                f"Succesfully Executed the SelfQuery & generated metafieltering params and new query New Query{new_query} & Metafilters {search_kwargs}"
            )
            return new_query, search_kwargs
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Generate Metadata Fielters and New Query by SelfQuery Retrierver Reason: {error} -> TRACEBACK: {traceback.format_exc()}"
            )
            raise  # Re-raise the exception after logging it
    
    def retrieve_query(self, question: str) -> tuple[str, dict]:
        """
            Synchronous wrapper for the async retrieve_query_async method.

            Args:
                question (str): The input question to be checked.

            Returns:
                tuple:
                    - A string representing the new query.
                    - A dictionary containing the metadata parameters to use post_filtering of metadata for milvus search.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already running an event loop, use run_coroutine_threadsafe
                self.logger.info("Running in an existing event loop.")
                future = asyncio.run_coroutine_threadsafe(
                    self.retrieve_query_async(question), loop
                )
                return future.result()
            else:
                # If no event loop is running, create a new one
                self.logger.info("Starting a new event loop.")
                return asyncio.run(self.retrieve_query_async(question))
        except Exception as e:
            self.logger.error(f"Error in async_rag: {e}")
            raise