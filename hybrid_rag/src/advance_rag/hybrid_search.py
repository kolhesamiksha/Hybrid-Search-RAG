"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import asyncio

from langchain_core.documents import Document
from pymilvus import AnnSearchRequest
from pymilvus import RRFRanker
from pymilvus import SearchResult

from hybrid_rag.src.models.retriever_model.models import EmbeddingModels
from hybrid_rag.src.utils.logutils import Logger
from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager


class MilvusHybridSearch:
    def __init__(
        self,
        collection_name: str,
        sparse_embedding_model: str,
        dense_embedding_model: str,
        sparse_search_params: dict,
        dense_search_params: dict,
        metadata_attributes: List[Dict[str,str]],
        vectorDbInstance: VectorStoreManager,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the MilvusHybridSearch object with necessary parameters.

        Args:
            collection_name (str): The name of the Milvus collection.
            sparse_embedding_model (str): The sparse embedding model.
            dense_embedding_model (str): The dense embedding model.
            sparse_search_params (dict): The parameters for sparse search.
            dense_search_params (dict): The parameters for dense search.
            vectorDbInstance (VectorStoreManager): VectorStoreManager class instance.
        """
        self.logger = logger if logger else Logger().get_logger()
        self.collection_name = collection_name
        self.sparse_embedding_model = sparse_embedding_model
        self.dense_embedding_model = dense_embedding_model
        self.sparse_search_params = sparse_search_params
        self.dense_search_params = dense_search_params
        self.vectorDbInstance = vectorDbInstance
        self.milvus_collection = self.vectorDbInstance.load_collection(
            self.collection_name
        )
        self.sparseEmbedModel = EmbeddingModels(self.sparse_embedding_model)
        self.denseEmbedModel = EmbeddingModels(self.dense_embedding_model)
        self.metadata_attributes_lst = self._create_metadata_list(metadata_attributes)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @collection_name.setter
    def collection_name(self, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Collection name must be a non-empty string.")
        self._collection_name = value

    @property
    def sparse_search_params(self) -> dict:
        return self._sparse_search_params

    @sparse_search_params.setter
    def sparse_search_params(self, value: dict) -> dict:
        if not isinstance(value, dict):
            raise ValueError("Sparse search parameters must be a dictionary")
        self._sparse_search_params = value

    @property
    def dense_search_params(self) -> dict:
        return self._dense_search_params

    @dense_search_params.setter
    def dense_search_params(self, value: dict) -> dict:
        if not isinstance(value, dict):
            raise ValueError("Dense search parameters must be a dictionary")
        self._dense_search_params = value

    async def generate_embeddings(self, question: str) -> Tuple[object, object]:
        """
        Generate sparse and dense embeddings for the given question.

        Args:
            question (str): The input question to generate embeddings for.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing sparse and dense embeddings as lists of floats.
        """
        
        sparse_task = self.sparseEmbedModel.sparse_embedding_model(question)
        dense_task = self.denseEmbedModel.dense_embedding_model(question)
        sparse_question_emb, dense_question_emb = await asyncio.gather(sparse_task, dense_task)
        return sparse_question_emb, dense_question_emb

    def _create_metadata_list(self, metadata_attributes: List[Dict[str,str]]) -> List[str]:
        attribute_info = []

        for i, meta in enumerate(metadata_attributes):
            attribute_name = meta.get(f'METADATA_ATTRIBUTE{i+1}_NAME', '').strip()
            if attribute_name:
                attribute_info.append(attribute_name)

        return attribute_info
    
    def perform_search(
        self, 
        sparse_question_emb: object, 
        dense_question_emb: object, 
        search_limit: int, 
        dense_search_limit: int, 
        sparse_search_limit: int,
        metadata_filters: dict, 
    ) -> List[List[SearchResult]]:
        """
        Perform hybrid search on the Milvus collection using sparse and dense embeddings.

        Args:
            sparse_question_emb (List[float]): The sparse question embedding.
            dense_question_emb (List[float]): The dense question embedding.

        Returns:
            List[Document]: A list of documents matching the search query.
        """
        
        # Create AnnSearchRequest for sparse and dense queries
        sparse_q = AnnSearchRequest(
            sparse_question_emb, "sparse_vector", self.sparse_search_params, limit=sparse_search_limit
        )
        dense_q = AnnSearchRequest(
            dense_question_emb, "dense_vector", self.dense_search_params, limit=dense_search_limit
        )

        # Perform hybrid search
        res = self.milvus_collection.hybrid_search(
            [sparse_q, dense_q],
            rerank=RRFRanker(),
            limit=search_limit,
            output_fields= ["text"] + self.metadata_attributes_lst,
        )

        return res

    def process_results(self, res: List[List[SearchResult]]) -> List[Document]:
        """
        Process the search results and create a list of Document objects.

        Args:
            res (Any): The search results from Milvus, could be a list, dictionary, or another format depending on the Milvus client.

        Returns:
            List[Document]: A list of Document objects containing page content and metadata.
        """
        output = []
        for _, hits in enumerate(res):
            for hit in hits:
                page_content = hit.entity.get("text")
                metadata = {attr: hit.entity.get(attr) for attr in self.metadata_attributes_lst}
                doc_chunk = Document(page_content=page_content, metadata=metadata)
                output.append(doc_chunk)
        return output

    async def hybrid_search_async(self, 
            question: str,
            metadata_filters: dict, 
            search_limit: int, 
            dense_search_limit: int, 
            sparse_search_limit:int, 
        ) -> List[Document]:
        """
        Perform hybrid search by generating embeddings, performing search, and processing results.

        Args:
            question (str): The input question to perform the search for.

        Returns:
            List[Document]: A list of Document objects containing the search results.
        """
        # Generate sparse and dense embeddings
        sparse_question_emb, dense_question_emb = await self.generate_embeddings(question)

        # Perform hybrid search
        res = self.perform_search(sparse_question_emb, dense_question_emb, search_limit, dense_search_limit, sparse_search_limit, metadata_filters)
        self.logger.info(f"Results After Hybrid Search: {res}")
        # Process and return results
        output = self.process_results(res)
        return output
    
    def hybrid_search(self, question: str, metadata_filters: dict, search_limit: int, dense_search_limit: int, sparse_search_limit:int,) -> List[Document]:
        """
        Synchronously Perform hybrid search by generating embeddings, performing search, and processing results.
        
        Args:
            question (str): The input question to perform the search for.

        Returns:
            List[Document]: A list of Document objects containing the search results.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already running an event loop, use run_coroutine_threadsafe
                self.logger.info("Running in an existing event loop.")
                future = asyncio.run_coroutine_threadsafe(
                    self.hybrid_search_async(question, metadata_filters, search_limit, dense_search_limit, sparse_search_limit), loop
                )
                return future.result()
            else:
                # If no event loop is running, create a new one
                self.logger.info("Starting a new event loop.")
                return asyncio.run(self.hybrid_search_async(question, metadata_filters, search_limit, dense_search_limit, sparse_search_limit))
        except Exception as e:
            self.logger.error(f"Error in async hybrid_search: {e}")
            raise
