from langchain_groq import ChatGroq
from hybrid_rag.src.utils.get_insert_mongo_data import format_creds_mongo
from hybrid_rag.src.utils.utils import decrypt_pass

LLM_MODEL_NAME = "llama-3.1-70b-versatile"


creds_mongo = format_creds_mongo()

GROQ_API_KEY = decrypt_pass(creds_mongo['GROQ_API_KEY'])

def initialise_llm_model(llm_model):
    llm_model = ChatGroq(model=LLM_MODEL_NAME,api_key = GROQ_API_KEY, temperature=0.0, max_retries=2)
    return llm_model