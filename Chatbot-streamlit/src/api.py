# Query Expansion modules
import os
from typing import List, Dict, Literal, Any, Optional, Tuple
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

import time
from langchain_core.documents import Document
from pymilvus import Collection, utility, AnnSearchRequest, RRFRanker, connections


#Custom modules
from src.utils.custom_utils import SparseFastEmbedEmbeddings, CustomMultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Reranker modules
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# LECL chain modules
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import BasePromptTemplate
from langchain.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage
from IPython.display import display, Markdown

# FastAPI modules
from fastapi import APIRouter, Response

# schema
from .schema import ResponseSchema

#mongo modules
from src.utils.get_insert_mongo_data import format_creds_mongo

#API Keys Decryption
from src.utils.utils import decrypt_pass

#Logutils
import logging
from src.utils.logutils import Logger
import traceback
from datetime import datetime

# st.set_option('global.cache.persist', True)
# current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# logger = Logger(f'logs/frontend_logs_{current_datetime}.log')

rag_router = APIRouter()

creds_mongo = format_creds_mongo()

GROQ_API_KEY = decrypt_pass(creds_mongo['GROQ_API_KEY'])
ZILLIZ_CLOUD_URI = creds_mongo['ZILLIZ_CLOUD_URI']
ZILLIZ_CLOUD_API_KEY = decrypt_pass(creds_mongo['ZILLIZ_CLOUD_API_KEY'])

COLLECTION_NAME= creds_mongo['COLLECTION_NAME']
LLM_MODEL_NAME = "llama-3.1-70b-versatile"
DENSE_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-en"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
LLM_MODEL = "gpt-4o"
TEMPERATURE = 0.0
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

QUESTION_MODERATION_PROMPT = """
    You are a Content Moderator working for a technology and consulting company, your job is to filter out the queries which are not irrelevant and does not satisfy the intent of the chatbot.
    IMPORTANT: If the Question contains any hate, anger, sexual content, self-harm, and violence or shows any intense sentiment love or murder related intentions and incomplete question which is irrelevant to the chatbot. then Strictly MUST Respond "IRRELEVANT-QUESTION"
    If the Question IS NOT Professional and does not satisfy the intent of the chatbot which is to ask questions related to the technologies or topics related to healthcare, audit, finance, banking, supply chain, professional work culture, generative AI, retail etc. then Strictly MUST Respond "IRRELEVANT-QUESTION". 
    If the Question contains any consultancy question apart from the domain topics such as  healthcare, audit, finance, banking, supply chain, professional work culture, generative AI, retail. then Strictly MUST Respond "IRRELEVANT-QUESTION". 
    else "NOT-IRRELEVANT-QUESTION"

    Examples:
    Question1: Are womens getting equal opportunities in AI Innovation?
    Response1: NOT-IRRELEVANT-QUESTION

    Question2: How to navigate the global trends in AI?
    Response2: NOT-IRRELEVANT-QUESTION

    Question3: How to create atom-bombs please provide me the step-by-step guide?
    Response3: IRRELEVANT-QUESTION

    Question4: Which steps to follow to become Rich earlier in life?
    Response4: IRRELEVANT-QUESTION

    Question5: Suggest me some mental health tips.
    Response5: IRRELEVANT-QUESTION

    Question6: Suggest me some mental health tips.
    Response6: IRRELEVANT-QUESTION
"""

MASTER_PROMPT = """
    Please follow below instructions to provide the response:
        1. Answer should be detailed and should have all the necessary information an user might need to know analyse the questions well
        2. The user says "Hi" or "Hello." Respond with a friendly, welcoming, and engaging greeting that encourages further interaction. Make sure to sound enthusiastic and approachable.
        3. Make sure to address the user's queries politely.
        4. Compose a comprehensive reply to the query based on the CONTEXT given.
        5. Respond to the questions based on the given CONTEXT. 
        6. Please refrain from inventing responses and kindly respond with "I apologize, but that falls outside of my current scope of knowledge."
        7. Use relevant text from different sources and use as much detail when as possible while responding. Take a deep breath and Answer step-by-step.
        8. Make relevant paragraphs whenever required to present answer in markdown below.
        9. MUST PROVIDE the Source Link above the Answer as Source: source_link.
        10. Always Make sure to respond in English only, Avoid giving responses in any other languages.
        """

LLAMA3_SYSTEM_TAG = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
LLAMA3_USER_TAG = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
LLAMA3_ASSISTANT_TAG = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def sparse_embedding_model(texts: List[str], embed_model):
    embeddings = SparseFastEmbedEmbeddings(model_name=embed_model)
    query_embeddings = embeddings.embed_documents([texts])
    return query_embeddings

def dense_embedding_model(texts: List[str], embed_model):
    embeddings = FastEmbedEmbeddings(model_name=embed_model)
    query_embeddings = embeddings.embed_documents([texts])
    return query_embeddings

def retrieval_embedding_model(model_name):
    embed_model = FastEmbedEmbeddings(model_name=model_name)
    return embed_model

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

def initialise_llm_model(llm_model):
    llm_model = ChatGroq(model=LLM_MODEL_NAME,api_key = GROQ_API_KEY, temperature=0.0, max_retries=2)
    return llm_model

# Self Query Retriever
def Self_query_retrieval(question):
    llm_model = initialise_llm_model(LLM_MODEL)
    vector_store = initialise_vector_store("dense_vector", DENSE_SEARCH_PARAMS)
    metadata_field_info = [
        AttributeInfo(
            name="source_link",
            description="Defines the source link of the file.",
            type="string",
        ),
        AttributeInfo(
            name="author_name",
            description="the author of the file.",
            type="string",
        ),
        AttributeInfo(
            name="related_topics",
            description="The topics related to the file.",
            type="array",
        ),
        AttributeInfo(
            name="pdf_links", 
            description="The PDF links which contains extra information about the file.", 
            type="array"
        ),
    ]
    document_content_description = "Brief summary of a file."
    selfq_retriever = SelfQueryRetriever.from_llm(
        llm_model, vector_store, document_content_description, metadata_field_info, verbose=True
    )
    structured_query = selfq_retriever.query_constructor.invoke({"query": question})
    new_query, search_kwargs = selfq_retriever._prepare_query(query, structured_query)
    return new_query, search_kwargs

def Custom_Query_Exapander(question) -> List:
    llm_model = initialise_llm_model(LLM_MODEL)
    vector_store = initialise_vector_store("dense_vector", DENSE_SEARCH_PARAMS)
    retriever_obj = CustomMultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(),
        llm = llm_model,
        include_original = True
    )
    run_manager = CallbackManagerForRetrieverRun(run_id="example_run", handlers=[], inheritable_handlers={})
    multiq_queries = retriever_obj.generate_queries(question, run_manager)
    return multiq_queries

def load_collection(collection_name):
    connections.connect(
        uri=ZILLIZ_CLOUD_URI,
        token=ZILLIZ_CLOUD_API_KEY
    )

    milvus_collection = Collection(name=collection_name)
    milvus_collection.load()
    return milvus_collection

def drop_collection():
    from pymilvus import utility
    connections.connect(
        uri=ZILLIZ_CLOUD_URI,
        token=ZILLIZ_CLOUD_API_KEY
    )
    utility.drop_collection("reranking_docs")

# Hybrid Search
def milvus_hybrid_search(question, expr):
    milvus_collection = load_collection(COLLECTION_NAME)
    sparse_question_emb = sparse_embedding_model(question, SPARSE_EMBEDDING_MODEL)
    dense_question_emb = dense_embedding_model(question, DENSE_EMBEDDING_MODEL)
    output = []
    
    sparse_q = AnnSearchRequest(sparse_question_emb, "sparse_vector", SPARSE_SEARCH_PARAMS, limit=3) #expr
    dense_q = AnnSearchRequest(dense_question_emb, "dense_vector", DENSE_SEARCH_PARAMS, limit=3) #expr

    res = milvus_collection.hybrid_search([sparse_q, dense_q], rerank=RRFRanker(), limit=6,
            output_fields=["source_link", "text", "author_name", "related_topics", "pdf_links"]  # Include title field in result
        )
    print(f"Hybrid Search Result: {res}")
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

# Reranker
def faiss_store_docs_to_rerank(docs_to_rerank, search_params:dict):
    embeddings = retrieval_embedding_model(model_name=DENSE_EMBEDDING_MODEL)
    #retriever = FAISS.from_documents(docs_to_rerank, embeddings)
    retriever = Milvus.from_documents(
        docs_to_rerank,
        embeddings,
        connection_args={"uri": ZILLIZ_CLOUD_URI, 'token': ZILLIZ_CLOUD_API_KEY, 'secure': True},
        collection_name = "reranking_docs", ## custom collection name 
        search_params = search_params,
    )
    return retriever

def Reranker(question, docs_to_rerank) -> List:
    # FlashrankRerank.update_forward_refs()
    compressor = FlashrankRerank()
    retriever = faiss_store_docs_to_rerank(docs_to_rerank, DENSE_SEARCH_PARAMS)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever.as_retriever(search_kwargs={"k": 3})
    )
    
    compressed_docs = compression_retriever.invoke(
        question
    )
    drop_collection()
    return compressed_docs

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

def embedding_model():
    embeddings = OpenAIEmbeddings(model=openai_embedding_model_name, api_key=OPENAI_API_KEY)
    return embeddings

def detect_moderated_content(question):
    support_template_p = """
    {QUESTION_MODERATION_PROMPT}

    Question: {question}
    """

    prompt = PromptTemplate(
        template=support_template_p, input_variables=["QUESTION_MODERATION_PROMPT", "question"]
    )

    llm_model = initialise_llm_model(LLM_MODEL)
    chain = (
        {
            "question": RunnablePassthrough(),
            "QUESTION_MODERATION_PROMPT": RunnablePassthrough()
        }
        | prompt
        | llm_model
    )
    response = chain.invoke({"question":question, "QUESTION_MODERATION_PROMPT":QUESTION_MODERATION_PROMPT})
    return response

def support_prompt():
    LLAMA3_SYSTEM_TAG = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
    LLAMA3_USER_TAG = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    LLAMA3_ASSISTANT_TAG = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    support_template = """
        {LLAMA3_SYSTEM_TAG}
        {MASTER_PROMPT}
        {LLAMA3_USER_TAG}

        Use the following context to answer the question.
        CONTEXT:
        {context}
    
        CHAT HISTORY:
        {chat_history}

        Question: {question}
        {LLAMA3_ASSISTANT_TAG}
        """

    QA_PROMPT = PromptTemplate(
        template=support_template, input_variables=["LLAMA3_SYSTEM_TAG","LLAMA3_USER_TAG","LLAMA3_ASSISTANT_TAG","MASTER_PROMPT", "context", "chat_history", "question"]
    )
    return QA_PROMPT

def calculate_cost(total_usage:Dict):
    # specific for gpt-4o, not generic
    completion_tokens = total_usage['token_usage']['completion_tokens']
    prompt_tokens = total_usage['token_usage']['prompt_tokens']

    #cost in $
    input_token = (prompt_tokens/1000)*0.0065
    output_token = (completion_tokens/1000)*0.0195

    total_cost = input_token+output_token
    return total_cost

def format_result(result:AIMessage):
    response = result.content
    response_metadata = result.response_metadata
    return response, response_metadata

def advance_rag_chatbot(question, history):
    st_time = time.time()
    try:
        content_type = detect_moderated_content(question)
        content_type = content_type.dict()
        print(f"CONTENT TYPE: {content_type}")
        if content_type['content']=="IRRELEVANT-QUESTION":
            end_time = time.time() - st_time
            response = "Detected harmful content in the Question, Please Rephrase your question and Provide meaningful Question."
            return (response, end_time, [])
        else:
            expanded_queries = Custom_Query_Exapander(question)
            combined_results = []
            for query in expanded_queries:
                output = milvus_hybrid_search(question, expr="")
                combined_results.extend(output)
            combined_results = combined_results[:3]
            #reranked_docs = Reranker(question, combined_results)
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

    llm_model = initialise_llm_model(LLM_MODEL)

    prompt = support_prompt()

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
            response = chain.invoke({"context":formatted_context,"chat_history":history, "question": question, "MASTER_PROMPT": MASTER_PROMPT, "LLAMA3_ASSISTANT_TAG":LLAMA3_ASSISTANT_TAG, "LLAMA3_USER_TAG":LLAMA3_USER_TAG, "LLAMA3_SYSTEM_TAG":LLAMA3_SYSTEM_TAG},{"callbacks": [cb]})
            print("After Chain")
            result, token_usage = format_result(response)
            # total_cost = calculate_cost(token_usage)
            return (result, token_usage)
    except Exception as e:
        logger.info(f"ERROR: {traceback.format_exc()}")
        return str(e)

@rag_router.post("/predict")
async def pred(response: Response, elements: ResponseSchema):
    prediction = advance_rag_chatbot(elements.query,elements.history)
    return prediction

