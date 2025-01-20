"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import traceback
from typing import List
import asyncio
from functools import lru_cache

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from hybrid_rag.src.utils.custom_utils import SparseFastEmbedEmbeddings
from hybrid_rag.src.utils.logutils import Logger

# TODO:add Logger & exceptions
logger = Logger().get_logger()


class EmbeddingModels:
    def __init__(self, embed_model: str):
        """
        Initializes the EmbeddingModels class with a model name.

        :param embed_model: The model name to use for embedding.
        """
        self.embed_model = embed_model

    @lru_cache(maxsize=1)  # Cache the SparseFastEmbedEmbeddings model initialization
    def _get_sparse_embedding_model(self)-> SparseFastEmbedEmbeddings:
        """
        Initializes and caches the sparse embedding model.
        
        :return: Instance of SparseFastEmbedEmbeddings.
        """
        try:
            embeddings = SparseFastEmbedEmbeddings(model_name=self.embed_model)
            logger.info(f"Successfully loaded Sparse embedding model: {self.embed_model}")
            return embeddings
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to load Sparse embedding model: {error}")
            raise

    async def sparse_embedding_model(self, texts: List[str]) -> List[List[float]]:
        """
        Creates sparse embeddings for the given texts.

        :param texts: List of texts to generate sparse embeddings for.
        :return: The sparse embeddings for the provided texts.
        """
        try:
            embeddings = self._get_sparse_embedding_model()
            print(f"Sparse Embedding Type: {texts}")
            query_embeddings = embeddings.embed_documents([texts])
            logger.info(
                f"Successfully Converted the text into Sparse Vectors Using model: {self.embed_model}"
            )
            return query_embeddings
        except Exception as e:
            error = str(e)
            logger.error(
                f"Failed to Initialised LLM model Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise
    
    @lru_cache(maxsize=1)  # Cache the SparseFastEmbedEmbeddings model initialization
    def _get_dense_embedding_model(self) -> FastEmbedEmbeddings:
        """
        Initializes and caches the sparse embedding model.
        
        :return: Instance of SparseFastEmbedEmbeddings.
        """
        try:
            embeddings = FastEmbedEmbeddings(model_name=self.embed_model)
            logger.info(f"Successfully loaded Sparse embedding model: {self.embed_model}")
            return embeddings
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to load Sparse embedding model: {error}")
            raise

    async def dense_embedding_model(self, texts: List[str]) -> List[List[float]]:
        """
        Creates dense embeddings for the given texts.

        :param texts: List of texts to generate dense embeddings for.
        :return: The dense embeddings for the provided texts.
        """
        try:
            embeddings = self._get_dense_embedding_model()
            print(f"Dense Embedding: {texts}")
            query_embeddings = embeddings.embed_documents([texts])
            logger.info(
                f"Successfully Converted the text into Dense Vectors Using Model : {self.embed_model}"
            )
            return query_embeddings
        except Exception as e:
            error = str(e)
            logger.error(
                f"Failed to Initialised LLM model Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
            raise
    
    @lru_cache(maxsize=3)
    def retrieval_embedding_model(self) -> FastEmbedEmbeddings:
        """
        Creates a retrieval embedding model.

        :return: An instance of the retrieval embedding model.
        """
        embed_model = None
        try:
            embed_model = self._get_dense_embedding_model()
            logger.info(
                f"Successfully Initialised FastEmbed retriever model: {self.embed_model}"
            )
        except Exception as e:
            error = str(e)
            logger.error(
                f"Failed to Initialised LLM model Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
        return embed_model
