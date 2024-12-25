from hybrid_rag.src.models.retriever_model.models import retrieval_embedding_model
from hybrid_rag.src.utils.get_insert_mongo_data import format_creds_mongo
from hybrid_rag.src.utils.utils import decrypt_pass
from langchain_community.vectorstores import Milvus

creds_mongo = format_creds_mongo()

DENSE_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-en"
ZILLIZ_CLOUD_URI = creds_mongo['ZILLIZ_CLOUD_URI']
ZILLIZ_CLOUD_API_KEY = decrypt_pass(creds_mongo['ZILLIZ_CLOUD_API_KEY'])
COLLECTION_NAME = creds_mongo['COLLECTION_NAME']

def initialise_vector_store(vector_field:str, search_params:dict):
    embeddings = retrieval_embedding_model(DENSE_EMBEDDING_MODEL)
    vector_store = Milvus(
           embeddings,
           connection_args={"uri": ZILLIZ_CLOUD_URI, 'token': ZILLIZ_CLOUD_API_KEY, 'secure': True},
           collection_name = COLLECTION_NAME, ## custom collection name 
           search_params = search_params,
            vector_field = vector_field
        )
    return vector_store