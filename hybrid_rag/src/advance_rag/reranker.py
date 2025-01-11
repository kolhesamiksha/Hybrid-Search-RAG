"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
import traceback
from typing import List
from typing import Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document

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
        Initialize the DocumentReranker with the required parameters.

        :param dense_embedding_model: The model name for dense embeddings.
        :param zillinz_cloud_uri: The URI for the Zilliz Cloud instance.
        :param zillinz_cloud_api_key: The API key for Zilliz Cloud.
        :param dense_search_params: Parameters for the dense search.
        :param vectorDbInstance: VectorStoreManager class instance, class object as parameter
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

    def milvus_store_docs_to_rerank(
        self, docs_to_rerank, search_params: dict
    ) -> Milvus:
        """
        Converts documents to be reranked and stores them in Milvus for retrieval.

        :param docs_to_rerank: List of documents to be reranked.
        :param search_params: Parameters for search in Milvus.
        :return: A Milvus retriever object.
        """
        try:
            retriever = Milvus.from_documents(
                docs_to_rerank,
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

    def rerank_docs(
        self, question: str, docs_to_rerank: List[Document], rerank_topk: int
    ) -> List[Document]:
        """
        Reranks documents based on the given question and returns the top-ranked documents.

        :param question: The input question for reranking the documents.
        :param docs_to_rerank: List of documents to be reranked.
        :param rerank_topk: The number of top documents to return after reranking.
        :return: The list of compressed (reranked) documents.
        """
        try:
            retriever = self.milvus_store_docs_to_rerank(
                docs_to_rerank, self.dense_search_params
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=retriever.as_retriever(search_kwargs={"k": rerank_topk}),
            )
            compressed_docs = compression_retriever.invoke(question)
            self.logger.info("Successfully Compressed and Reranked the documents")
            self.vectorDbInstance.drop_collection("reranking_docs")
            return compressed_docs
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Rranked the documents Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise
