"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
from typing import Optional

from langchain_community.vectorstores import Milvus
from pymilvus import Collection
from pymilvus import connections
from pymilvus import utility

from hybrid_rag.src.models.retriever_model.models import EmbeddingModels
from hybrid_rag.src.utils.logutils import Logger


class VectorStoreManager:
    def __init__(
        self,
        zillinz_cloud_uri: str,
        zillinz_cloud_api_key: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the VectorStoreManager with required configuration.

        Args:
            zilliz_cloud_uri (str): URI for the Zilliz Cloud service.
            zilliz_cloud_api_key (str): API key for accessing Zilliz Cloud.

        Returns:
            None: This method initializes the VectorStoreManager instance and doesn't return a value.
        """
        self.logger = logger if logger else Logger().get_logger()
        self.zillinz_cloud_uri = zillinz_cloud_uri
        self.__zillinz_cloud_api_key = zillinz_cloud_api_key
        self._connect()  # Establish connection once at initialization

    def _connect(self) -> None:
        """
        Establish connection to Zilliz Cloud using the provided URI and API key.

        This method ensures that the required connection is established before interacting with Milvus.
        """
        connections.connect(
            uri=self.zillinz_cloud_uri, token=self.__zillinz_cloud_api_key
        )

    def load_collection(self, collection_name: str) -> Collection:
        """
        Load a Milvus collection by name.

        Args:
            collection_name (str): Name of the collection to be loaded.

        Returns:
            MilvusCollection: The loaded collection object.

        Raises:
            Exception: If loading the collection fails.
        """
        try:
            milvus_collection = Collection(name=collection_name)
            milvus_collection.load()
            self.logger.info(f"Collection {collection_name} loaded successfully.")
            return milvus_collection
        except Exception as e:
            self.logger.error(f"Failed to load collection {collection_name}: {str(e)}")
            raise

    def drop_collection(self, collection_name: str) -> None:
        """
        Drop a Milvus collection by name.

        Args:
            collection_name (str): Name of the collection to be dropped.

        Raises:
            Exception: If dropping the collection fails.
        """
        try:
            utility.drop_collection(collection_name)
            self.logger.info(f"Collection {collection_name} dropped successfully.")
        except Exception as e:
            self.logger.error(f"Failed to drop collection {collection_name}: {str(e)}")
            raise

    def initialise_vector_store(
        self,
        vector_field: str,
        search_params: dict,
        dense_embedding_model: str,
        collection_name: str,
    ) -> Milvus:
        """
        Initialize a vector store with the specified parameters.

        Args:
            vector_field (str): Name of the vector field in the collection schema.
            search_params (dict): Search parameters for the vector store (e.g., search method, distance metric).
            dense_embedding_model (str): Path or name of the dense embedding model for generating embeddings.
            collection_name (str): Name of the collection where the vector store will be initialized.

        Returns:
            vector_store: An initialized vector store object (type depends on the implementation of the vector store).

        Raises:
            Exception: If initialization fails due to invalid parameters, connectivity issues, or other initialization problems.
        """
        try:
            embedding_model = EmbeddingModels(dense_embedding_model)
            embeddings = embedding_model.retrieval_embedding_model()
            vector_store = Milvus(
                embeddings,
                connection_args={
                    "uri": self.zillinz_cloud_uri,
                    "token": self.__zillinz_cloud_api_key,
                    "secure": True,
                },
                collection_name=collection_name,
                search_params=search_params,
                vector_field=vector_field,
            )
            self.logger.info(
                f"Vector store for collection {collection_name} initialized successfully."
            )
            return vector_store
        except Exception as e:
            self.logger.error(
                f"Failed to initialize vector store for collection {collection_name}: {str(e)}"
            )
            raise
