"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
import traceback
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Generator, AsyncGenerator
import asyncio

from langchain_core.documents import Document

from hybrid_rag.src.models.retriever_model.models import EmbeddingModels
from hybrid_rag.src.utils.logutils import Logger
from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager

class MilvusIterativeSearch:
    def __init__(
            self,
            collection_name: str,
            dense_embedding_model: str,
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
            self.dense_embedding_model = dense_embedding_model
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

    def perform_iterative_search(
        self,
        dense_question_emb: object,
        batch_size: int, 
        search_limit: int,
        filter_expr: dict,
    ) -> Generator[Document, None, None]:
        """
        Performs metadata filtering with iterative search on a Milvus collection.

        Args:
            dense_question_emb (object): The dense vector representation of the question/query.
            batch_size (int): The number of search results to process in each batch.
            search_limit (int): The maximum number of search results to retrieve per query.
            filter_expr (dict): A dictionary containing filter expressions to apply during the search.

        Returns:
            List[List[SearchResult]]: A list of lists where each sublist contains `SearchResult` objects.
                Each `SearchResult` contains:
                    - `page_content`: The content of the retrieved document chunk.
                    - `metadata`: A dictionary containing metadata attributes for the document.

        Raises:
            Exception: Captures and logs any exceptions encountered during the search process.
        """
        iterator = self.milvus_collection.search_iterator(
            data=dense_question_emb,
            anns_field="dense_vector",
            param=self.dense_search_params,
            batch_size=batch_size,
            filter=filter_expr,
            limit=search_limit,
            output_fields= ["text"] + self.metadata_attributes_lst,
        )

        while True:
            # highlight-next-line
            result = iterator.next()
            if not result:
                # highlight-next-line
                iterator.close()
                break
            
            for _, hits in enumerate(result):
                for hit in hits:
                    page_content = hit.entity.get("text")
                    metadata = {attr: hit.entity.get(attr) for attr in self.metadata_attributes_lst}
                    doc_chunk = Document(page_content=page_content, metadata=metadata)
                    yield doc_chunk
    
    async def iterative_search_async(self,
            question: str,
            batch_size: int,
            filter_expr: dict, 
            search_limit: int,
        ) -> AsyncGenerator[Document, None, None]:
        
        """
        Asynchronously performs iterative search using a given question and filtering expression.

        This method generates embeddings for the input question and then performs an iterative search 
        based on the generated embeddings, batch size, search limit, and metadata filter expression.

        Args:
            question (str): The question to search for, which will be used to generate embeddings.
            batch_size (int): The batch size to use when performing the search.
            filter_expr (dict): A dictionary containing filter expressions to apply during the search.
            search_limit (int): The maximum number of search results to retrieve.

        Returns:
            List[Document]: A list of `Document` objects containing the search results, where each document contains 
                            the text and metadata from the search results.

        Raises:
            Exception: If an error occurs during the iterative search or embedding generation, an empty list is returned.
        """
        try:
            _, dense_question_emb = await self.generate_embeddings(question)
            for document in self.perform_iterative_search(dense_question_emb, batch_size, search_limit, filter_expr):
                yield document
        except Exception as e:
            self.logger.error("ERROR while ")
    
    def iterative_search(self, question: str, batch_size: int, filter_expr: dict, search_limit: int) -> List[Document]:
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
                    self.iterative_search_async(question, batch_size, filter_expr, search_limit), loop
                )
                result = future.result()
                return list(result)
            else:
                # If no event loop is running, create a new one
                self.logger.info("Starting a new event loop.")
                result = asyncio.run(self.iterative_search_async(question, batch_size, filter_expr, search_limit))
                return list(result)
        except Exception as e:
            self.logger.error(f"Error in async hybrid_search: {e}")
            raise