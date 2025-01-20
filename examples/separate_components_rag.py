from hybrid_rag.src.models import LLMModelInitializer
from hybrid_rag.src.vectordb import VectorStoreManager
from hybrid_rag.src.advance_rag import MilvusHybridSearch
from hybrid_rag.src.advance_rag import SelfQueryRetrieval
from hybrid_rag.src.advance_rag import DocumentReranker
from hybrid_rag.src.advance_rag import CustomQueryExpander
from hybrid_rag.src.prompts import SupportPromptGenerator
from hybrid_rag.src.moderation import QuestionModerator
from hybrid_rag.src.utils import DocumentFormatter
from hybrid_rag.src.evaluate import RAGAEvaluator
from hybrid_rag.src.utils import calculate_cost_openai, calculate_cost_groq_llama31, save_history_to_s3
from hybrid_rag.src.utils import Logger
from hybrid_rag.src.config import Config
from dotenv import load_dotenv
import asyncio

import time
from typing import List, Dict, Optional, Tuple, Any

from langchain.callbacks import get_openai_callback
from langchain_core.runnables import (
    RunnablePassthrough,
)

#load_dotenv(dotenv_path=".env.example")
logger = Logger().get_logger()

#LLM Configs
llm_model_name = "llama-3.1-8b-instant"
provider_base_url = ""
groq_api_key = ""
temperature = 0.3
top_p = 0.1
frequency_penalty = 1.0

#Embedding params
sparse_embedding_model = "Qdrant/bm42-all-minilm-l6-v2-attentions"
dense_embedding_model = "jinaai/jina-embeddings-v2-base-en"
dense_search_params = {
    'index_type':'IVF_SQ8',
    'metric_type': 'L2',
    'params': {'nlist': 128}
}
sparse_search_params = {
    'index_type':'SPARSE_INVERTED_INDEX',
    'metric_type':'IP'
}

metadata_attributes = [
    {
        "name": "source_link",
        "description": "Defines the source link of the file.",
        "type": "string"
    },
    {
        "name": "author_name",
        "description": "The author of the file.",
        "type": "string"
    },
    {
        "name": "related_topics",
        "description": "The topics related to the file.",
        "type": "array"
    },
    {
        "name": "pdf_links",
        "description": "The PDF links which contain extra information about the file.",
        "type": "array"
    },
]

document_info = "ey company docs contains audit, tax, ai & supply chain domains"

#VectorDB params
collection_name = ""
zillinz_cloud_uri = ""
zillinz_cloud_api_key = ""
top_k = 4

#Reranking params
rerank_topk=3
dense_topk=3
sparse_topk=3

##AWS CONFIGS:
s3_bucket = ""
s3_key = ""
aws_access_key_id = ""
aws_secret_access_key = ""

#Prompt params
master_prompt = """Please follow below instructions to provide the response:
        1. Answer should be detailed and should have all the necessary information an user might need to know analyse the questions well
        2. The user says Hi or Hello. Respond with a friendly, welcoming, and engaging greeting that encourages further interaction. Make sure to sound enthusiastic and approachable
        3. Make sure to address the user's queries politely.
        4. Compose a comprehensive reply to the query based on the CONTEXT given.
        5. Respond to the questions based on the given CONTEXT.
        6. Please refrain from inventing responses and kindly respond with I apologize, but that falls outside of my current scope of knowledge.
        7. Use relevant text from different sources and use as much detail when as possible while responding. Take a deep breath and Answer step-by-step.
        8. Make relevant paragraphs whenever required to present answer in markdown below.
        9. MUST PROVIDE the Source Link above the Answer as Source: source_link.
        10. Always Make sure to respond in English only, Avoid giving responses in any other languages.
"""

question_moderation_prompt = """You are a Content Moderator working for a technology and consulting company, your job is to filter out the queries which are not irrelevant and does not satisfy the intent of the chatbot.
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

#Note: If your model has user_tags or system_tags, please do mention like below inside SupportPromptGenerator, 
# If your model is llama then it automatically, captures correct tags (set inside the code:)
# But, If your model has some specific tags then it's better to write.. improvises the performance!! As most of the generation performance can be improvdes using prompts.

llama3_user_tag = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
llama3_system_tag = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
llama3_assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# Initialize other components with necessary parameters
support_prompt_generator = SupportPromptGenerator(
    llm_model_name=llm_model_name,
    master_prompt=master_prompt,
    llama3_user_tag=llama3_user_tag,
    llama3_system_tag=llama3_system_tag,
    llama3_assistant_tag=llama3_assistant_tag,
)

model = LLMModelInitializer(
    llm_model_name=llm_model_name, 
    provider_base_url=provider_base_url, 
    groq_api_key=groq_api_key, 
    temperature=temperature, 
    top_p=top_p, 
    frequency_penalty=frequency_penalty, 
    logger=logger
)

vectordb = VectorStoreManager(
    zillinz_cloud_uri=zillinz_cloud_uri, 
    zillinz_cloud_api_key=zillinz_cloud_api_key, 
    logger=logger
)

milvus_hybrid_search = MilvusHybridSearch(
    collection_name=collection_name,
    sparse_embedding_model=sparse_embedding_model,
    dense_embedding_model=dense_embedding_model,
    sparse_search_params=sparse_search_params,
    dense_search_params=dense_search_params,
    vectorDbInstance=vectordb,
    logger=logger
)

self_query_retrieval = SelfQueryRetrieval(
    collection_name=collection_name,
    dense_search_params=dense_search_params,
    dense_embedding_model=dense_embedding_model,
    metadata_attributes=metadata_attributes,
    document_info=document_info,
    llmModelInstance=model,
    vectorDbInstance=vectordb,
    logger=logger
)

custom_query_expander = CustomQueryExpander(
    collection_name=collection_name,
    dense_search_params=dense_search_params,
    dense_embedding_model=dense_embedding_model,
    llmModelInstance=model,
    vectorDbInstance=vectordb,
    logger=logger
)

document_reranker = DocumentReranker(
    dense_embedding_model=dense_embedding_model,
    zillinz_cloud_uri=zillinz_cloud_uri,
    zillinz_cloud_api_key=zillinz_cloud_api_key,
    dense_search_params=dense_search_params,
    vectorDbInstance=vectordb,
    logger=logger
)

question_moderator = QuestionModerator(
    llmModelInstance=model,
    logger=logger
)

document_formatter = DocumentFormatter()

rag_evaluator = RAGAEvaluator(
    dense_embedding_model=dense_embedding_model,
    llmModelInstance=model,
    logger=logger
)

def generate_chatbot_response(question: str, formatted_context: str, retrieved_history: List[str]) -> Tuple[str, dict]:
    llm_model = model.initialise_llm_model()
    prompt = support_prompt_generator.generate_prompt()
    chain = (
        {
            "LLAMA3_ASSISTANT_TAG": RunnablePassthrough(),
            "LLAMA3_USER_TAG": RunnablePassthrough(),
            "LLAMA3_SYSTEM_TAG": RunnablePassthrough(),
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),
            "MASTER_PROMPT": RunnablePassthrough(),
        } 
        | prompt 
        | llm_model
    )
    try:
        with get_openai_callback() as cb:
            response = chain.invoke(
                {
                    "context": formatted_context,
                    "chat_history": retrieved_history,
                    "question": question,
                    "MASTER_PROMPT": support_prompt_generator.MASTER_PROMPT,               #you can pass your own tags as well, instead leave empty=""
                    "LLAMA3_ASSISTANT_TAG": support_prompt_generator.LLAMA3_ASSISTANT_TAG,
                    "LLAMA3_USER_TAG": support_prompt_generator.LLAMA3_USER_TAG,
                    "LLAMA3_SYSTEM_TAG": support_prompt_generator.LLAMA3_SYSTEM_TAG,
                },
                {"callbacks": [cb]},
            )
        result = document_formatter.format_result(response)
        return result[0], result[1]
    except Exception as e:
        return f"Error: {str(e)}", {}

def is_coroutine_check(result):
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # For running in an active event loop
            return asyncio.run_coroutine_threadsafe(result, loop).result()
        else:
            return asyncio.run(result)
    return result

# Process the question in a custom flow
def process_question(question: str, history: List[str]) -> Tuple[str, float, List[Any], dict, float]:
    start_time = time.time()

    # Step 1: Detect the content type of the question (content moderation)
    content_type = question_moderator.detect(question, question_moderation_prompt=question_moderation_prompt)
    content_type = is_coroutine_check(content_type)
    content_type = content_type.dict()
    
    if content_type["content"] == "IRRELEVANT-QUESTION":
        end_time = time.time() - start_time
        response = "Detected harmful content in the Question, Please Rephrase your question and Provide meaningful Question."
        return response, end_time, [], {}, 0.0

    # Step 2: Expand the query and retrieve results
    expanded_queries = custom_query_expander.expand_query(question)
    expanded_queries = is_coroutine_check(expanded_queries)
    print(expanded_queries)
    self_query, metadata_filters = self_query_retrieval.retrieve_query(question)
    self_query = is_coroutine_check(self_query)
    print(self_query)
    expanded_queries.append(self_query)

    combined_results = []
    results_to_store = []
    for query in expanded_queries:
        output = milvus_hybrid_search.hybrid_search(query, search_limit=top_k, dense_search_limit=dense_topk, sparse_search_limit=sparse_topk)
        output = is_coroutine_check(output)
        combined_results.extend(output)
        results_to_store.append({"question": query, "output": output})

    # Step 3: Rerank the results (optional)
    reranked_docs = document_reranker.rerank_docs(question, docs_to_rerank=combined_results, rerank_topk=rerank_topk)
    reranked_docs = is_coroutine_check(reranked_docs)
    # Step 4: Format the final context
    formatted_context = document_formatter.format_docs(docs=reranked_docs)
    # Step 5: Generate the chatbot response using the LLM model
    response = generate_chatbot_response(question, formatted_context, history)
    # Step 6: Evaluate the results (optional)
    evaluated_results = {}
    # if True:  # Replace with your evaluation condition
    #     evaluated_results = rag_evaluator.evaluate_rag([question], [response], combined_results)

    end_time = time.time() - start_time
    total_cost = calculate_cost_groq_llama31(response[1])  # Replace with your cost calculation logic

    #response is stringm end_time is float, combined_results is dict, evaluated_results is dict, total_cost is float
    return response, end_time, combined_results, evaluated_results, total_cost

if __name__=="__main__":
    question = "tell me about supply chain"
    history = []
    result = process_question(question, history)
    print(result[0])
    
    #store the results to AWS
    save_history_to_s3(question, result, s3_bucket, s3_key, aws_access_key_id, aws_secret_access_key)


