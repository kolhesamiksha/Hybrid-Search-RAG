from typing import Optional
from pymilvus import Collection, AnnSearchRequest, RRFRanker, connections, utility
from hybrid_rag.src.models.retriever_model.models import EmbeddingModels
from langchain_community.vectorstores import Milvus
from hybrid_rag.src.utils.logutils import Logger

class VectorStoreManager:
    def __init__(self, zilliz_cloud_uri:str, zilliz_cloud_api_key:str, logger:Optional[Logger]=None):
        """
        Initialize the VectorStoreManager with required configuration.

        :param zilliz_cloud_uri: URI for the Zilliz Cloud service.
        :param zilliz_cloud_api_key: API key for accessing Zilliz Cloud.

        """
        self.logger = logger if logger else Logger().get_logger()
        self.zilliz_cloud_uri = zilliz_cloud_uri
        self.__zilliz_cloud_api_key = zilliz_cloud_api_key

    def _connect(self):
        """
        Establish connection to Zilliz Cloud using the provided URI and API key.

        This method ensures that the required connection is established before interacting with Milvus.
        """
        connections.connect(
            uri=self.zilliz_cloud_uri,
            token=self.__zilliz_cloud_api_key
        )

    def load_collection(self, collection_name):
        """
        Load a Milvus collection by name.

        :param collection_name: Name of the collection to be loaded.
        :return: The loaded collection object.
        :raises Exception: If loading the collection fails.
        """
        try:
            self._connect()
            milvus_collection = Collection(name=collection_name)
            milvus_collection.load()
            self.logger.info(f"Collection {collection_name} loaded successfully.")
            return milvus_collection
        except Exception as e:
            self.logger.error(f"Failed to load collection {collection_name}: {str(e)}")
            raise

    def drop_collection(self, collection_name):
        """
        Drop a Milvus collection by name.

        :param collection_name: Name of the collection to be dropped.
        :raises Exception: If dropping the collection fails.
        """
        try:
            self._connect()
            utility.drop_collection(collection_name)
            self.logger.info(f"Collection {collection_name} dropped successfully.")
        except Exception as e:
            self.logger.error(f"Failed to drop collection {collection_name}: {str(e)}")
            raise

    def initialise_vector_store(self, vector_field, search_params, dense_embedding_model, collection_name):
        """
        Initialize a vector store with the specified parameters.

        :param vector_field: Name of the vector field in the collection schema.
        :param search_params: Search parameters for the vector store (e.g., search method, distance metric).
        :param dense_embedding_model: Path or name of the dense embedding model for generating embeddings.
        :param collection_name: Name of the collection where the vector store will be initialized.
        :return: An initialized vector store object.
        :raises Exception: If initialization fails.
        """
        try:
            self._connect()
            embedding_model = EmbeddingModels(dense_embedding_model)
            embeddings = embedding_model.retrieval_embedding_model()
            vector_store = Milvus(
                embeddings,
                connection_args={"uri": self.zilliz_cloud_uri, 'token': self.__zilliz_cloud_api_key, 'secure': True},
                collection_name=collection_name,
                search_params=search_params,
                vector_field=vector_field
            )
            self.logger.info(f"Vector store for collection {collection_name} initialized successfully.")
            return vector_store
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store for collection {collection_name}: {str(e)}")
            raise
