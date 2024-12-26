from pymilvus import Collection, AnnSearchRequest, RRFRanker, connections, utility
from hybrid_rag.src.models.retriever_model.models import EmbeddingModels
from langchain_community.vectorstores import Milvus
from hybrid_rag.src.utils.logutils import Logger

logger = Logger().get_logger()

class VectorStoreManager:
    def __init__(self, zilliz_cloud_uri, zilliz_cloud_api_key):
        self.zilliz_cloud_uri = zilliz_cloud_uri
        self.__zilliz_cloud_api_key = zilliz_cloud_api_key

    def _connect(self):
        """Establish connection to Zilliz Cloud."""
        connections.connect(
            uri=self.zilliz_cloud_uri,
            token=self.__zilliz_cloud_api_key
        )

    def load_collection(self, collection_name):
        """Load a Milvus collection by name."""
        try:
            self._connect()
            milvus_collection = Collection(name=collection_name)
            milvus_collection.load()
            logger.info(f"Collection {collection_name} loaded successfully.")
            return milvus_collection
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {str(e)}")
            raise

    def drop_collection(self, collection_name):
        """Drop a Milvus collection by name."""
        try:
            self._connect()
            utility.drop_collection(collection_name)
            logger.info(f"Collection {collection_name} dropped successfully.")
        except Exception as e:
            logger.error(f"Failed to drop collection {collection_name}: {str(e)}")
            raise

    def initialise_vector_store(self, vector_field, search_params, dense_embedding_model, collection_name):
        """Initialize a vector store with the specified parameters."""
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
            logger.info(f"Vector store for collection {collection_name} initialized successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store for collection {collection_name}: {str(e)}")
            raise
