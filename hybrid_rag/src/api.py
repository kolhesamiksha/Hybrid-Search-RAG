# Query Expansion modules
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import traceback
from datetime import datetime

from langchain_core.documents import Document

# LECL chain modules
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage

# FastAPI modules
from fastapi import APIRouter, Response

# schema
from hybrid_rag.src.schema import ResponseSchema

#Logutils
from hybrid_rag.src.utils.logutils import Logger

from hybrid_rag.src.prompts.prompt import QUESTION_MODERATION_PROMPT, SupportPromptGenerator
from hybrid_rag.src.advance_rag.self_query_retriever import SelfQueryRetrieval
from hybrid_rag.src.advance_rag.query_expander import CustomQueryExpander
from hybrid_rag.src.advance_rag.reranker import DocumentReranker
from hybrid_rag.src.advance_rag.hybrid_search import MilvusHybridSearch
from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from hybrid_rag.src.utils.decrypter import AESDecryptor
from hybrid_rag.src.utils.get_mongo_data import MongoCredentialManager
from hybrid_rag.src.moderation.question_moderator import QuestionModerator
#TODO:add Logger & exceptions

# st.set_option('global.cache.persist', True)
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = Logger().get_logger()

rag_router = APIRouter()

#dataclass essentials from User
LLM_MODEL_NAME = "llama-3.1-70b-versatile"
DENSE_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-en"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
NO_HISTORY = 2
langchain=True
DENSE_SEARCH_PARAMS = {
    "index_type": "IVF_SQ8",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

SPARSE_SEARCH_PARAMS = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
    }
# Defone your Question Here
QUESTION = "What is Generative AI?"

CONNECTION_STRING = "mongodb+srv://kolhesamiksha25:yRxcIbqRBJORdwFm@cluster0.p3zf3zw.mongodb.net/"
MONGO_COLLECTION_NAME= "Hybrid-search-rag"
DB_NAME = "credentials"

TEMPERATURE = 0.0
TOP_P = 0.3
FREQUENCY_PENALTY = 1.0

HYBRID_SEARCH_TOPK = 6
RERANK_TOPK = 3

mongoCredManager = MongoCredentialManager(CONNECTION_STRING, MONGO_COLLECTION_NAME, DB_NAME)
aesDecryptor = AESDecryptor()

creds_mongo = mongoCredManager._format_creds()
GROQ_API_KEY = aesDecryptor.get_plain_text(creds_mongo['GROQ_API_KEY'])
ZILLIZ_CLOUD_URI = creds_mongo['ZILLIZ_CLOUD_URI']    
ZILLIZ_CLOUD_API_KEY = aesDecryptor.get_plain_text(creds_mongo['ZILLIZ_CLOUD_API_KEY'])
COLLECTION_NAME= creds_mongo['COLLECTION_NAME']


##Create Instances of all classes
supportPromptGenerator = SupportPromptGenerator()
#selfQueryRetrieval = SelfQueryRetrieval(LLM_MODEL_NAME, DENSE_SEARCH_PARAMS, DENSE_EMBEDDING_MODEL, ZILLIZ_CLOUD_URI, ZILLIZ_CLOUD_API_KEY, COLLECTION_NAME, GROQ_API_KEY) 
customQueryExpander = CustomQueryExpander(LLM_MODEL_NAME, DENSE_SEARCH_PARAMS, DENSE_EMBEDDING_MODEL, ZILLIZ_CLOUD_URI, ZILLIZ_CLOUD_API_KEY, COLLECTION_NAME, GROQ_API_KEY)
documentReranker = DocumentReranker(DENSE_EMBEDDING_MODEL, ZILLIZ_CLOUD_URI, ZILLIZ_CLOUD_API_KEY, DENSE_SEARCH_PARAMS) 
milvusHybridSearch = MilvusHybridSearch(COLLECTION_NAME, ZILLIZ_CLOUD_URI, ZILLIZ_CLOUD_API_KEY, SPARSE_EMBEDDING_MODEL, DENSE_EMBEDDING_MODEL, SPARSE_SEARCH_PARAMS, DENSE_SEARCH_PARAMS)
llmModelInitializer = LLMModelInitializer(LLM_MODEL_NAME, GROQ_API_KEY, TEMPERATURE, TOP_P, FREQUENCY_PENALTY)
questionModerator = QuestionModerator(LLM_MODEL_NAME, GROQ_API_KEY)

def format_document(doc: Document) -> str:
        prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
        if 'source_link' in doc.metadata.keys():
            prompt += PromptTemplate(input_variables=["source_link"], template="\n[Source: {source_link}]")
        base_info = {"page_content": doc.page_content, **doc.metadata}
        missing_metadata = set(prompt.input_variables).difference(base_info)
        if len(missing_metadata) > 0:
            required_metadata = [
                iv for iv in prompt.input_variables if iv != "page_content"
            ]
            raise ValueError(
                f"Document prompt requires documents to have metadata variables: "
                f"{required_metadata}. Received document with missing metadata: "
                f"{list(missing_metadata)}."
            )
        
        return prompt.format(**base_info)

def format_docs(docs):
    return "\n\n".join(format_document(doc) for doc in docs)

def format_result(result:AIMessage):
    response = result.content
    response_metadata = result.response_metadata
    return response, response_metadata

def advance_rag_chatbot(question, history):
    st_time = time.time()
    try:
        content_type = questionModerator.detect(question, QUESTION_MODERATION_PROMPT)
        content_type = content_type.dict()
        print(f"CONTENT TYPE: {content_type}")
        if content_type['content']=="IRRELEVANT-QUESTION":
            end_time = time.time() - st_time
            response = "Detected harmful content in the Question, Please Rephrase your question and Provide meaningful Question."
            return (response, end_time, [])
        else:
            expanded_queries = customQueryExpander.expand_query(question)
            #self_query, metadata_filters = selfQueryRetrieval.retrieve_query(question)
            combined_results = []
            for query in expanded_queries:
                output = milvusHybridSearch.hybrid_search(question, HYBRID_SEARCH_TOPK) 
                combined_results.extend(output)
            combined_results = combined_results[:3]
            #reranked_docs = documentReranker.rerank_docs(question, combined_results, RERANK_TOPK)
            formatted_context = format_docs(combined_results)
            response = chatbot(question, formatted_context, history)
            end_time = time.time() - st_time
            return (response, end_time, combined_results)
    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        end_time = time.time() - st_time
        return ("ERROR", end_time, [])
    
def chatbot(question, formatted_context, retrieved_history):
    history = []
    if retrieved_history:
        if len(retrieved_history)>=NO_HISTORY:
            history = retrieved_history[-NO_HISTORY:]
        else:
            history = retrieved_history

    llm_model = llmModelInitializer.initialise_llm_model()
    prompt = supportPromptGenerator.generate_prompt() ##

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = (
        {
            "LLAMA3_ASSISTANT_TAG":RunnablePassthrough(),
            "LLAMA3_USER_TAG":RunnablePassthrough(),
            "LLAMA3_SYSTEM_TAG":RunnablePassthrough(),
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),
            "MASTER_PROMPT": RunnablePassthrough()
        }
        | prompt
        | llm_model
    )
    try:
        with get_openai_callback() as cb:
            print("Before Chain")
            response = chain.invoke({"context":formatted_context,"chat_history":history, "question": question, "MASTER_PROMPT": supportPromptGenerator.MASTER_PROMPT, "LLAMA3_ASSISTANT_TAG":supportPromptGenerator.LLAMA3_ASSISTANT_TAG, "LLAMA3_USER_TAG":supportPromptGenerator.LLAMA3_USER_TAG, "LLAMA3_SYSTEM_TAG":supportPromptGenerator.LLAMA3_SYSTEM_TAG},{"callbacks": [cb]})
            print("After Chain")
            result, token_usage = format_result(response)
            # total_cost = calculate_cost(token_usage)
            return (result, token_usage)
    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        return str(e)

# @rag_router.post("/predict")
# async def pred(response: Response, elements: ResponseSchema):
#     prediction = advance_rag_chatbot(elements.query,elements.history)
#     return prediction

if __name__ == "__main__":
    query = ""
    history = []
    prediction = advance_rag_chatbot(query, history)
    print(prediction)

