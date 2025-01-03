"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import traceback
from typing import List

from hybrid_rag.src.utils.custom_utils import SparseFastEmbedEmbeddings
from hybrid_rag.src.utils.logutils import Logger
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# TODO:add Logger & exceptions
logger = Logger().get_logger()


class EmbeddingModels:
    def __init__(self, embed_model: str):
        """
        Initializes the EmbeddingModels class with a model name.

        :param embed_model: The model name to use for embedding.
        """
        self.embed_model = embed_model

    def sparse_embedding_model(self, texts: List[str]) -> List[List[float]]:
        """
        Creates sparse embeddings for the given texts.

        :param texts: List of texts to generate sparse embeddings for.
        :return: The sparse embeddings for the provided texts.
        """
        try:
            embeddings = SparseFastEmbedEmbeddings(model_name=self.embed_model)
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

    def dense_embedding_model(self, texts: List[str]) -> List[List[float]]:
        """
        Creates dense embeddings for the given texts.

        :param texts: List of texts to generate dense embeddings for.
        :return: The dense embeddings for the provided texts.
        """
        try:
            embeddings = FastEmbedEmbeddings(
                model_name="jinaai/jina-embeddings-v2-base-en"
            )
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

    def retrieval_embedding_model(self) -> FastEmbedEmbeddings:
        """
        Creates a retrieval embedding model.

        :return: An instance of the retrieval embedding model.
        """
        embed_model = None
        try:
            embed_model = FastEmbedEmbeddings(model_name=self.embed_model)
            logger.info(
                f"Successfully Initialised FastEmbed retriever model: {self.embed_model}"
            )
        except Exception as e:
            error = str(e)
            logger.error(
                f"Failed to Initialised LLM model Reason: {error} -> TRACEBACK : {traceback.format_exc()}"
            )
        return embed_model
