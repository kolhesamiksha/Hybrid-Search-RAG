import os
import time
from langchain_core.documents import Document
from pymilvus import Collection, utility, AnnSearchRequest, RRFRanker, connections

from hybrid_rag.src.vectordb.zillinz_milvus import VectorStoreManager
from hybrid_rag.src.models.retriever_model.models import EmbeddingModels

class MilvusHybridSearch:
    def __init__(self, collection_name: str, zillinz_cloud_uri:str, zillinz_cloud_api_key:str, sparse_embedding_model: str, dense_embedding_model: str, 
                 sparse_search_params: dict, dense_search_params: dict):
        """
        Initialize the MilvusHybridSearch object with necessary parameters.
        
        :param collection_name: The name of the Milvus collection.
        :param sparse_embedding_model: The sparse embedding model.
        :param dense_embedding_model: The dense embedding model.
        :param sparse_search_params: The parameters for sparse search.
        :param dense_search_params: The parameters for dense search.
        """
        self.collection_name = collection_name
        self.sparse_embedding_model = sparse_embedding_model
        self.dense_embedding_model = dense_embedding_model
        self.sparse_search_params = sparse_search_params
        self.dense_search_params = dense_search_params
        self.zillinz_cloud_uri = zillinz_cloud_uri
        self.__zillinz_cloud_api_key = zillinz_cloud_api_key
        
        # Load Milvus collection using VectorStoreManager
        vectorStoreManager = VectorStoreManager(self.zillinz_cloud_uri, self.__zillinz_cloud_api_key)
        self.milvus_collection = vectorStoreManager.load_collection(self.collection_name)

    @property
    def collection_name(self):
        return self._collection_name

    @collection_name.setter
    def collection_name(self, value: str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Collection name must be a non-empty string.")
        self._collection_name = value

    @property
    def sparse_search_params(self):
        return self._sparse_search_params

    @sparse_search_params.setter
    def sparse_search_params(self, value):
        if not isinstance(value, dict):
            raise ValueError("Sparse search parameters must be a dictionary")
        self._sparse_search_params = value

    @property
    def dense_search_params(self):
        return self._dense_search_params

    @dense_search_params.setter
    def dense_search_params(self, value):
        if not isinstance(value, dict):
            raise ValueError("Dense search parameters must be a dictionary")
        self._dense_search_params = value

    def generate_embeddings(self, question: str):
        """
        Generate sparse and dense embeddings for the given question.
        
        :param question: The input question to generate embeddings for.
        :return: A tuple containing sparse and dense embeddings.
        """
        sparseEmbedModel = EmbeddingModels(self.sparse_embedding_model)
        denseEmbedModel = EmbeddingModels(self.dense_embedding_model)
        sparse_question_emb = sparseEmbedModel.sparse_embedding_model(question)
        dense_question_emb = denseEmbedModel.dense_embedding_model(question)
        return sparse_question_emb, dense_question_emb

    def perform_search(self, sparse_question_emb, dense_question_emb, search_limit):
        """
        Perform hybrid search on the Milvus collection using sparse and dense embeddings.
        
        :param sparse_question_emb: The sparse question embedding.
        :param dense_question_emb: The dense question embedding.
        :return: A list of documents matching the search query.
        """
        # Create AnnSearchRequest for sparse and dense queries
        sparse_q = AnnSearchRequest(sparse_question_emb, "sparse_vector", self.sparse_search_params, limit=3)
        dense_q = AnnSearchRequest(dense_question_emb, "dense_vector", self.dense_search_params, limit=3)

        # Perform hybrid search
        res = self.milvus_collection.hybrid_search([sparse_q, dense_q], rerank=RRFRanker(), limit=search_limit,
                                                   output_fields=["source_link", "text", "author_name", "related_topics", "pdf_links"])

        return res

    def process_results(self, res):
        """
        Process the search results and create a list of Document objects.
        
        :param res: The search results from Milvus.
        :return: A list of Document objects containing page content and metadata.
        """
        output = []
        for _, hits in enumerate(res):
            for hit in hits:
                page_content = hit.entity.get("text")
                metadata = {
                    "source_link": hit.entity.get("source_link"),
                    "author_name": hit.entity.get("author_name"),
                    "related_topics": hit.entity.get("related_topics"),
                    "pdf_links": hit.entity.get("pdf_links")
                }
                doc_chunk = Document(page_content=page_content, metadata=metadata)
                output.append(doc_chunk)
        return output

    def hybrid_search(self, question: str, search_limit:int):
        """
        Perform hybrid search by generating embeddings, performing search, and processing results.
        
        :param question: The input question to perform the search for.
        :return: A list of Document objects containing the search results.
        """
        # Generate sparse and dense embeddings
        sparse_question_emb, dense_question_emb = self.generate_embeddings(question)

        # Perform hybrid search
        res = self.perform_search(sparse_question_emb, dense_question_emb, search_limit)

        # Process and return results
        output = self.process_results(res)
        return output