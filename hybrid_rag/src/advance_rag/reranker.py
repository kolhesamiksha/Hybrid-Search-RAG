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

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun

from hybrid_rag.src.models.retriever_model.models import EmbeddingModels
from hybrid_rag.src.utils.logutils import Logger
from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager


class DocumentReranker:
    def __init__(
        self,
        dense_embedding_model: str,
        zillinz_cloud_uri: str,
        zillinz_cloud_api_key: str,
        dense_search_params: dict,
        vectorDbInstance: VectorStoreManager,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the DocumentReranker with the required parameters.

        Args:
            dense_embedding_model (str): The dense embedding model name supported by FastEmbedding.
            zillinz_cloud_uri (str): The URI for the Zilliz Cloud instance.
            zillinz_cloud_api_key (str): The API key for accessing Zilliz Cloud.
            dense_search_params (dict): The parameters for performing dense search operations.
            vectorDbInstance (VectorStoreManager): Instance of VectorStoreManager for vector database management.
            logger (logging.Logger): Optional logger instance to log messages.
        
        Returns:
            None: This method initializes the reranker but does not return any value.
        """

        self.logger = logger if logger else Logger().get_logger()
        self.dense_embedding_model = dense_embedding_model
        self.zillinz_cloud_uri = zillinz_cloud_uri
        self.__zillinz_cloud_api_key = zillinz_cloud_api_key
        self.dense_search_params = dense_search_params
        self.vectorDbInstance = vectorDbInstance

        embeddingModel = EmbeddingModels(self.dense_embedding_model)
        self.embeddings = embeddingModel.retrieval_embedding_model()
        self.compressor = FlashrankRerank()
        self.run_manager = AsyncCallbackManagerForRetrieverRun(
            run_id="reranker_run", handlers=[], inheritable_handlers={}
        )
        self.batch_size: int = 30

    async def milvus_store_docs_to_rerank(
        self, docs_to_rerank, search_params: dict
    ) -> Milvus:
        """
        Initializes the DocumentReranker with the required parameters.

        Args:
            dense_embedding_model (str): The model name for dense embeddings.
            zillinz_cloud_uri (str): The URI for the Zilliz Cloud instance.
            zillinz_cloud_api_key (str): The API key for Zilliz Cloud.
            dense_search_params (dict): Parameters for the dense search.
            vectorDbInstance (VectorStoreManager): VectorStoreManager class instance.

        Returns:
            None
        """
        try:

            #batch approach to store the data incrementally inside the milvus store 
            for i in range(0, len(docs_to_rerank), self.batch_size):
                retriever = Milvus.from_documents(
                    docs_to_rerank[i:i + self.batch_size],
                    self.embeddings,
                    connection_args={
                        "uri": self.zillinz_cloud_uri,
                        "token": self.__zillinz_cloud_api_key,
                        "secure": True,
                    },
                    collection_name="reranking_docs",  # custom collection name
                    search_params=search_params,
                )
            self.logger.info(
                "Successfully Store the Retrieved Data for Reranking into Milvus with Collection: reranking_docs"
            )
            return retriever
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Store the Retrieved Data into FAISS for Rerank Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise

    async def rerank_docs_async(
        self, question: str, docs_to_rerank: List[Document], rerank_topk: int
    ) -> List[Document]:
        """
        Reranks documents based on the given question and returns the top-ranked documents.

        Args:
            question (str): The input question for reranking the documents.
            docs_to_rerank (List[str]): List of documents to be reranked.
            rerank_topk (int): The number of top documents to return after reranking.

        Returns:
            List[str]: The list of compressed (reranked) documents.
        """
        try:
            retriever = await self.milvus_store_docs_to_rerank(
                docs_to_rerank, self.dense_search_params
            )

            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=retriever.as_retriever(search_kwargs={"k": rerank_topk}),
            )
            compressed_docs = await compression_retriever._aget_relevant_documents(
                question, 
                run_manager=self.run_manager,
            )
            
            self.logger.info("Successfully Compressed and Reranked the documents")
            self.vectorDbInstance.drop_collection("reranking_docs")
            return compressed_docs
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Rranked the documents Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise
    
    def rerank_docs(self, question: str, docs_to_rerank: List[Document], rerank_topk: int) -> List[Document]:
        """
        Synchronously reranks the given documents based on the provided question.

        Args:
            question (str): The input question used for reranking the documents.
            docs_to_rerank (List[Document]): A list of documents to be reranked.
            rerank_topk (int): The number of top documents to return after reranking.

        Returns:
            List[Document]: The list of reranked documents.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already running an event loop, use run_coroutine_threadsafe
                self.logger.info("Running in an existing event loop.")
                future = asyncio.run_coroutine_threadsafe(
                    self.rerank_docs_async(question, docs_to_rerank, rerank_topk), loop
                )
                return future.result()
            else:
                # If no event loop is running, create a new one
                self.logger.info("Starting a new event loop.")
                return asyncio.run(self.rerank_docs_async(question, docs_to_rerank, rerank_topk))
        except Exception as e:
            self.logger.error(f"Error in async_rag: {e}")
            raise