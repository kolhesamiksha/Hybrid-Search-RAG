"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""
import json
import logging
import asyncio

import os
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import numpy as np

import mlflow.pyfunc
import pandas as pd
import psutil
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec
from mlflow.types import Schema

#for memory consumption report & CPU optimization report
from memory_profiler import profile
import cProfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# LECL chain modules
from langchain.callbacks import get_openai_callback
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)

from hybrid_rag.src.advance_rag import (
    CustomQueryExpander,
    DocumentReranker,
    MilvusHybridSearch,
    SelfQueryRetrieval,
)
from hybrid_rag.src.config import Config
from hybrid_rag.src.evaluate import RAGAEvaluator
from hybrid_rag.src.models import LLMModelInitializer
from hybrid_rag.src.moderation import QuestionModerator
from hybrid_rag.src.prompts.prompt import (
    SupportPromptGenerator,
)
from hybrid_rag.src.prompts.followup_prompt import (
    FollowupPromptGenerator,
)

from hybrid_rag.src.utils import (
    DocumentFormatter,
    Logger,
)
from hybrid_rag.src.vectordb import VectorStoreManager
from hybrid_rag.src.custom_mlflow.predict import RAGChatbotModel
from hybrid_rag.src.utils.utils import calculate_cost_groq_llama31
from hybrid_rag.src.utils.followup_questions import FollowupQGeneration

# TODO:add Logger & exceptions
warnings.filterwarnings("ignore")


class RAGChatbot:
    """
    A class that encapsulates the functionality of a Retrieval-Augmented Generation (RAG) chatbot.
    The chatbot can process questions, handle content moderation, perform hybrid search, and interact with a language model.
    """

    def __init__(
        self, config: Config, logger: Optional[logging.Logger], **kwargs: dict
    ) -> None:
        """
        Initializes the RAGChatbot instance with a given configuration.

        Args:
            config (Config): The configuration object containing all necessary parameters.
        """
        self.config = config
        self.logger = logger if logger else Logger().get_logger()
        self.kwargs = kwargs

        subprocess.run("chcp 65001", shell=True, check=True)
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["MLFLOW_LOG_LEVEL"]="DEBUG"

        self.mlflow_tracking_uri = self.config.MLFLOW_TRACKING_URI
        self.mlflow_experiment = self.config.MLFLOW_EXPERIMENT_NAME
        self.mlflow_run_name = self.config.MLFLOW_RUN_NAME
        self._setup_mlflow()
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.langchain.autolog(log_traces=True)
        
        
        # Initialize dependencies using the provided self.config
        self.supportPromptGenerator = SupportPromptGenerator(
            self.config.LLM_MODEL_NAME,
            self.config.MASTER_PROMPT,
            self.config.MODEL_SPECIFIC_PROMPT_USER_TAG,
            self.config.MODEL_SPECIFIC_PROMPT_SYSTEM_TAG,
            self.config.MODEL_SPECIFIC_PROMPT_ASSISTANT_TAG,
            self.logger,
        )
        self.followupPromptGenerator = FollowupPromptGenerator(
            self.config.FOLLOWUP_TEMPLATE,
            self.logger
        )

        self.llmModelInitializer = LLMModelInitializer(
            self.config.LLM_MODEL_NAME,
            self.config.PROVIDER_BASE_URL,
            self.config.LLM_API_KEY,
            self.config.TEMPERATURE,
            self.config.TOP_P,
            self.config.FREQUENCY_PENALTY,
            self.logger,
        )
        self.vectorDBInitializer = VectorStoreManager(
            self.config.ZILLIZ_CLOUD_URI,
            self.config.ZILLIZ_CLOUD_API_KEY,
            self.logger,
        )
        self.milvusHybridSearch = MilvusHybridSearch(
            self.config.COLLECTION_NAME,
            self.config.SPARSE_EMBEDDING_MODEL,
            self.config.DENSE_EMBEDDING_MODEL,
            self.config.SPARSE_SEARCH_PARAMS,
            self.config.DENSE_SEARCH_PARAMS,
            self.config.SELF_RAG_METADATA_ATTRIBUTES,
            self.vectorDBInitializer,
            self.logger,
        )
        self.selfQueryRetrieval = SelfQueryRetrieval(
            self.config.COLLECTION_NAME,
            self.config.DENSE_SEARCH_PARAMS,
            self.config.DENSE_EMBEDDING_MODEL,
            self.config.SELF_RAG_METADATA_ATTRIBUTES,
            self.config.SELF_RAG_DOCUMENTS_BRIEF_SUMMARY,
            self.llmModelInitializer,
            self.vectorDBInitializer,
            self.logger,
        )
        self.customQueryExpander = CustomQueryExpander(
            self.config.COLLECTION_NAME,
            self.config.DENSE_SEARCH_PARAMS,
            self.config.DENSE_EMBEDDING_MODEL,
            self.llmModelInitializer,
            self.vectorDBInitializer,
            self.logger,
        )
        self.documentReranker = DocumentReranker(
            self.config.DENSE_EMBEDDING_MODEL,
            self.config.ZILLIZ_CLOUD_URI,
            self.config.ZILLIZ_CLOUD_API_KEY,
            self.config.DENSE_SEARCH_PARAMS,
            self.vectorDBInitializer,
            self.logger,
        )
        self.questionModerator = QuestionModerator(
            self.llmModelInitializer,
            self.logger,
        )
        self.docFormatter = DocumentFormatter()
        self.ragaEvaluator = RAGAEvaluator(
            self.config.DENSE_EMBEDDING_MODEL,
            self.llmModelInitializer,
            self.logger,
        )
        self.followupqgenerator = FollowupQGeneration(
            self.llmModelInitializer,
            self.followupPromptGenerator,
            self.logger
        )

        if isinstance(os.getenv("DENSE_SEARCH_PARAMS"), dict):
            for key, value in os.getenv("DENSE_SEARCH_PARAMS").items():
                mlflow.log_param(key, value)
                self.logger.info(
                    f"Logged {len(os.getenv('DENSE_SEARCH_PARAMS'))} parameters to MLflow."
                )
            else:
                self.logger.info("dense_search_params is not a dictionary.")

        if isinstance(os.getenv("SPARSE_SEARCH_PARAMS"), dict):
            for key, value in os.getenv("SPARSE_SEARCH_PARAMS").items():
                mlflow.log_param(key, value)
                self.logger.info(
                    f"Logged {len(os.getenv('SPARSE_SEARCH_PARAMS'))} parameters to MLflow."
                )
            else:
                self.logger.info("spase_search_params is not a dictionary.")
        #mlflow.langchain.autolog(log_models=True,log_input_examples=True)

    def _setup_mlflow(self):
        mlflow_command = [
            "mlflow",
            "server",
            "--backend-store-uri",
            "sqlite:///mlflow.db",  # Use SQLite database
            "--default-artifact-root",
            "./mlruns",  # Artifact storage path
            "--host",
            "0.0.0.0",  # Bind to all network interfaces
            "--port",
            "5000",  # Port to run the server on
        ]

        try:
            # Start the MLflow server
            self.logger.info("Starting MLflow server...")
            subprocess.Popen(mlflow_command)
            self.logger.info("MLflow server started on http://localhost:5000")
        except Exception as e:
            self.logger.error(f"Failed to start MLflow server: {e}")

    def _get_system_metrics(self):
        """
        Collects system metrics such as CPU usage, memory usage, and disk usage.

        Returns:
            dict: A dictionary containing system metrics.
        """
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "used": psutil.virtual_memory().used,
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "free": psutil.disk_usage("/").free,
                "percent": psutil.disk_usage("/").percent,
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_received": psutil.net_io_counters().bytes_recv,
            },
        }
        return metrics

    async def _chatbot(
        self, question: str, formatted_context: str, retrieved_history: List[str]
    ) -> Tuple[str, dict]:
        """
        Generates a response to a question using a pre-defined prompt and a language model. The context and history are provided to the model.

        Args:
            question (str): The question asked by the user.
            formatted_context (str): The formatted context from the document retrieval system.
            retrieved_history (List[str]): The conversation history.

        Returns:
            Tuple[str, int]:
                - The response generated by the language model.
                - The token usage for the response.
        """
        history = []
        if not retrieved_history:
            if len(retrieved_history) >= self.config.NO_HISTORY:
                history = retrieved_history[-self.config.NO_HISTORY :]
            else:
                history = retrieved_history

        llm_model = self.llmModelInitializer.initialise_llm_model()
        prompt = self.supportPromptGenerator.generate_prompt()

        # Define the chain of operations including the language model.
        # chain = (
        #     {
        #         "LLAMA3_ASSISTANT_TAG": RunnablePassthrough(),
        #         "LLAMA3_USER_TAG": RunnablePassthrough(),
        #         "LLAMA3_SYSTEM_TAG": RunnablePassthrough(),
        #         "context": RunnablePassthrough(),
        #         "question": RunnablePassthrough(),
        #         "chat_history": RunnablePassthrough(),
        #         "MASTER_PROMPT": RunnablePassthrough(),
        #     }
        #     | prompt
        #     | llm_model
        # )

        ########### Rnnable Parallel Chaining reduces the chaining time by 20% ###############
        chain = (
            RunnableParallel(
                {
                    "LLAMA3_ASSISTANT_TAG": RunnablePassthrough(),
                    "LLAMA3_USER_TAG": RunnablePassthrough(),
                    "LLAMA3_SYSTEM_TAG": RunnablePassthrough(),
                    "context": RunnablePassthrough(),
                    "question": RunnablePassthrough(),
                    "chat_history": RunnablePassthrough(),
                    "MASTER_PROMPT": RunnablePassthrough(),  # Ensure it's a `Runnable`
                }
            )
            | prompt
            | llm_model
        )
        self.logger.info("Successfully Initlalised the Chain.")
        try:
            # Use the OpenAI callback to monitor API usage.
            with get_openai_callback() as cb:
                response = await chain.ainvoke(
                    {
                        "context": formatted_context,
                        "chat_history": history,
                        "question": question,
                        "MASTER_PROMPT": self.supportPromptGenerator.MASTER_PROMPT,
                        "LLAMA3_ASSISTANT_TAG": self.supportPromptGenerator.LLAMA3_ASSISTANT_TAG,
                        "LLAMA3_USER_TAG": self.supportPromptGenerator.LLAMA3_USER_TAG,
                        "LLAMA3_SYSTEM_TAG": self.supportPromptGenerator.LLAMA3_SYSTEM_TAG,
                    },
                    {"callbacks": [cb]},
                )

                self.logger.info(f"Successfully Generated the Response: {response}")
                # Format the result and return the response.
                result = self.docFormatter.format_result(response)
                self.logger.info(f"Token Usage for the Question: {result[1]}")
                # total_cost = calculate_cost(token_usage)
                return result[0], result[1]

        except Exception as e:
            self.logger.info(f"ERROR: {traceback.format_exc()}")
            return str(e), {}

    def _generate_reranked_to_mlflow(self, docs):
        for doc in docs:
            yield self._convert_documents_to_dicts([doc])[0]

    def _generate_results_to_store(self, queries, outputs):
        for query, output in zip(queries, outputs):
            yield {
                "question": query,
                "output": self._convert_documents_to_dicts(output)
            }

    def _convert_documents_to_dicts(self, combined_results: list) -> list:
        """
        Converts a list of Document objects into a list of dictionaries.

        Args:
            combined_results (list): A list containing Document objects.

        Returns:
            list: A list of dictionaries with the format:
                {
                    "content": "<page_content>",
                    "metadata": {
                        "field1": "value1",
                        "field2": "value2",
                        ...
                    }
                }
        """
        converted_results = []
        for document in combined_results:
            if hasattr(document, "page_content") and hasattr(document, "metadata"):
                converted_results.append(
                    {"content": document.page_content, "metadata": document.metadata}
                )
            else:
                # Handle non-Document objects gracefully
                self.logger.info(f"Skipping non-Document object: {document}")
        return converted_results

    def convert_to_serializable(self,obj):
        if isinstance(obj, np.float32):
            return float(obj)  # Convert numpy.float32 to Python float
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        return obj

    def _add_json_results(self, retrieved_results: list, file_name: str) -> None:
        temp_folder = "dummy_to_delete"
        os.makedirs(temp_folder, exist_ok=True)
        serializable_results = self.convert_to_serializable(retrieved_results)
        try:
            # Open the file in write mode and dump the results
            with open(f"dummy_to_delete/{file_name}", "w", encoding="utf-8") as json_file:
                json.dump(serializable_results, json_file, indent=4)
            self.logger.info(f"Successfully wrote combined results to {file_name}")
        except Exception as e:
            self.logger.error(
                f"Failed to write combined results to {file_name}: {str(e)}"
            )
            raise

    def _garbage_collector(self, folder_name: str) -> None:
        try:
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
            else:
                self.logger.info(f"The directory '{folder_name}' does not exist.")
        except Exception as e:
            self.logger.error(f"An error occurred while deleting the directory: {e}")

    #@profile  #memory-profiler for montoring code memory consumption!!
    async def _advance_rag_chatbot_async(
        self, question: str, history: List[str]
    ) -> Tuple[str, float, List[Any], dict, float, List[str]]:
        """
        Processes a question through the chatbot pipeline, including content moderation, query expansion, document retrieval, and response generation.

        Args:
            question (str): The question to process.
            history (List[str]): A list of previous conversation history.

        Returns:
            Tuple[str, float, List[Any]]:
                - A response string from the chatbot.
                - The time taken to process the request.
                - A list of combined search results.
        """
        st_time = time.time()
        if mlflow.active_run() is not None:
            mlflow.end_run()

        with mlflow.start_run(run_name=self.mlflow_run_name):
            #batch logging approach to remove I/O overhead and searialisation cost
            ###############
            """
                Batching parameters into a single mlflow.log_params call:

                - Reduces the number of network/DB interactions.
                - Serializes and processes all parameters in one go, which is faster and more memory-efficient
            """
            ######################
            mlflow.log_params({
                "LLM_MODEL_NAME": os.getenv("LLM_MODEL_NAME"),
                "DENSE_EMBEDDING_MODEL": os.getenv("DENSE_EMBEDDING_MODEL"),
                "SPARSE_EMBEDDING_MODEL": os.getenv("SPARSE_EMBEDDING_MODEL"),
                "TOP_K": os.getenv("HYBRID_SEARCH_TOPK"),
                "TEMPERATURE": os.getenv("TEMPERATURE"),
                "TOP_P": os.getenv("TOP_P"),
                "FREQUENCY_PENALTY": os.getenv("FREQUENCY_PENALTY"),
                "user_question": question,
            })
            try:
                # Detect the content type of the question using the moderator.
                with mlflow.start_span(name="advance_rag_chatbot") as main_span:
                    with mlflow.start_span(name="question_moderation") as mod_span:
                        content_type = await self.questionModerator.detect_async(
                            question, self.config.QUESTION_MODERATION_PROMPT
                        )
                        self.logger.info(
                            f"Question Moderation Response:{content_type.dict()['content']}"
                        )
                        mod_span.set_attributes({"moderation_response": content_type.dict()["content"]})
                        
                        # If the question is irrelevant, return an error message.
                        if content_type.dict()["content"] == "IRRELEVANT-QUESTION":
                            end_time = time.time() - st_time
                            response = "Detected harmful content in the Question, Please Rephrase your question and Provide meaningful Question."
                            self.logger.info("IRRELEVANT-QUESTION")
                            return (response, end_time, [], {}, 0.0, [])
                    
                    with mlflow.start_span(name="query_expansion") as query_span:
                        # Expand the query and retrieve results.

                        tasks = [
                            asyncio.create_task(self.selfQueryRetrieval.retrieve_query_async(question)),
                            asyncio.create_task(self.customQueryExpander.expand_query_async(question)),
                        ]
                        self_query, expanded_queries = await asyncio.gather(*tasks)
                        metadata_filters = self_query[1]  # Assuming self_query is a tuple
                        metadata_filters_str = json.dumps(metadata_filters)
                        mlflow.log_param("self_query_metadata_filters", metadata_filters_str)
                        
                        self.logger.info(f"METADATA FILTERS: {metadata_filters}")

                        expanded_queries.append(self_query[0])
                        self.logger.info(f"Expanded Queries are: {expanded_queries}")
                        self.logger.info(
                            f"Self Query Generated: {self_query}, Metadata Filters: {metadata_filters}"
                        )
                        query_span.set_attributes({
                            "self_query": self_query[0],
                            "metadata_filters": metadata_filters,
                            "expanded_queries": expanded_queries,
                        })
                    
                    with mlflow.start_span(name="hybrid_search") as search_span:
                        queries = (query for query in expanded_queries)
                        outputs = await asyncio.gather(*[
                            self.milvusHybridSearch.hybrid_search_async(query, metadata_filters, self.config.HYBRID_SEARCH_TOPK, self.config.DENSE_SEARCH_LIMIT, self.config.SPARSE_SEARCH_LIMIT)
                            for query in queries
                        ])

                        #Use generator to avoid loading 48 combined documents inside a list with a vast amount of text as page content-> Memory Optimisation
                        results_to_store = (
                            result for result in self._generate_results_to_store(expanded_queries, outputs)
                        )

                        for result in results_to_store:
                            # Add to MLflow or perform other operations
                            self._add_json_results([result], "retrieved_questions_results.json")
                        
                        mlflow.log_artifact("dummy_to_delete/retrieved_questions_results.json")

                        #Access Combined Results Context
                        combined_results = []
                        for _, output in zip(expanded_queries, outputs):
                            combined_results.extend(output)

                        search_span.set_attributes({"retrieved_docs_count": len(combined_results)})
                        mlflow.log_param("retrieved_docs", len(combined_results))
                        self.logger.info(f"Combine results: {combined_results}")
                        self.logger.info(f"Type of Combined results: {type(combined_results)}")
                        del results_to_store

                    with mlflow.start_span(name="context_reranking") as rerank_span:
                        if self.config.IS_RERANK:
                            # Rank and format the documents.
                            reranked_task = asyncio.create_task(self.documentReranker.rerank_docs_async(
                                question, combined_results, self.config.RERANK_TOPK
                            ))
                            reranked_docs = await reranked_task
                        else:
                            reranked_docs = combined_results[-self.config.RERANK_TOPK :]

                        self.logger.info(f"Reranked Documents: {reranked_docs}")
                        self.logger.info(f"Type of RERANKED DOCS: {type(reranked_docs)}")
                        self.logger.info(f"TOTAL RERANKED DOCS: {len(reranked_docs)}")

                        reranker_to_mlflow = (
                            result for result in self._generate_reranked_to_mlflow(reranked_docs)
                        )

                        for result in reranker_to_mlflow:
                            # Add to MLflow or perform other operations
                            self._add_json_results([result], "reranked_results.json")
                        mlflow.log_artifact("dummy_to_delete/reranked_results.json")

                        #Storing results to disk lower down the memory usage but might increase the latency, kept holding in memory.
                        formatted_context = self.docFormatter.format_docs(reranked_docs)
                        self.logger.info(f"Formatted_cntext: {formatted_context}")

                        #mlflow cause issues while serealising the data.
                        formatted_context = formatted_context.replace("\u2192", "")
                        rerank_span.set_attributes({
                            "reranked_docs_count": len(reranked_docs),
                            "formatted_context": formatted_context,
                        })
                        self.logger.info(f"Formatted Context: {formatted_context}")
                        del reranker_to_mlflow

                    with mlflow.start_span(name="evaluate_response") as evaluate_span:
                        # Generate the chatbot response.
                        response, token_usage = await self._chatbot(
                            question, formatted_context, history
                        )
                        print(token_usage)
                        response = response.encode('utf-8', 'ignore').decode('utf-8')
                        evaluate_span.set_attributes({
                            "Chatbot Response": response,
                            "Token Usage": token_usage
                            })

                    with mlflow.start_span(name="followup_questions") as followup_span:
                        if self.config.IS_FOLLOWUP:
                            try:
                                #Generate Followup Questions
                                followup_questions = await self.followupqgenerator.generate_followups(question, reranked_docs, response)
                            except Exception as e:
                                followup_questions = []
                                self.logger.error(f"Error While Generating Followup Questions! {str(e)} TRACEBACK: {traceback.format_exc()}")
                        else:
                            self.logger.info("Followup Question generation is not allowed")
                            followup_questions = []
                        
                        followup_span.set_attributes({
                                "followup_questions": followup_questions
                            })
                        mlflow.log_param("followup_questions", followup_questions)
                        del reranked_docs

                    with mlflow.start_span(name="ragas_evaluation") as ragas_span:
                        if self.config.IS_EVALUATE:
                            evaluated_results = await self.ragaEvaluator.evaluate_rag_async(
                                [question], [response], combined_results
                            )
                        else:
                            evaluated_results = ()
                        #evaluated_results = evaluated_results.encode('utf-8', 'ignore').decode('utf-8')
                        # ragas_span.set_attributes({
                        #     "evaluated_results": evaluated_results,
                        # })

                self.logger.info(f"Evaluation Result: {evaluated_results}")
                if token_usage:
                    total_cost = calculate_cost_groq_llama31(
                        token_usage,
                        self.config.INPUT_TOKENS_PER_MILLION_COST,
                        self.config.INPUT_TOKENS_PER_MILLION_COST,
                    )
                else:
                    total_cost = 0.0
                
                system_metrics = self._get_system_metrics()
                mlflow.log_metrics({
                    "cpu_percent": system_metrics["cpu_percent"],
                    "memory_used_percent": system_metrics["memory"]["percent"],
                    "disk_used_percent": system_metrics["disk"]["percent"],
                    "network_bytes_sent": system_metrics["network"]["bytes_sent"],
                    "network_bytes_received": system_metrics["network"]["bytes_received"],
                    "total_response_time": time.time() - st_time,
                    "completion_tokens": token_usage["token_usage"].get("completion_tokens", 0),
                    "prompt_tokens": token_usage["token_usage"].get("prompt_tokens", 0),
                    "prompt_time": token_usage["token_usage"].get("prompt_time", 0),
                    "completion_time": token_usage["token_usage"].get("completion_time", 0),
                    "queue_time": token_usage["token_usage"].get("queue_time", 0),
                    "total_tokens": token_usage["token_usage"].get("total_tokens", 0),
                    "total_chain_time": token_usage["token_usage"].get("total_time", 0),
                    "total_cost": total_cost,
                })
                mlflow.log_param("response", response)
                if evaluated_results:
                    if len(evaluated_results) > 1:
                        mlflow.log_metrics({
                            "answer_relevancy": getattr(evaluated_results[0], "answer_relevancy", 0),
                            "faithfulness": getattr(evaluated_results[0], "faithfulness", 0),
                            "context_precision": getattr(evaluated_results[1], "context_precision", 0),
                        })
                else:
                    self.logger.warning(
                        "Evaluated results tuple is empty. Skipping MLflow logging."
                    )
                self._garbage_collector("dummy_to_delete")

                #TODO: Streaming to response can be applied
                return (
                    response,
                    time.time() - st_time,
                    combined_results,
                    evaluated_results,
                    total_cost,
                    followup_questions,
                )
            except Exception:
                self._garbage_collector("dummy_to_delete")
                self.logger.error(f"ERROR: {traceback.format_exc()}")
                total_cost = 0.0
                return ("ERROR", time.time() - st_time, [], {}, 0.0)
        
        self.logger.info(f"Total_Time_required: {time.time()-st_time}")
        mlflow.end_run()
    
    def advance_rag_chatbot(self, question, history):
        # profiler = cProfile.Profile()
        # profiler.enable()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # In an existing event loop, schedule the coroutine as a task and wait for it
            future = asyncio.run_coroutine_threadsafe(
                self._advance_rag_chatbot_async(question, history), loop
            )
            result = future.result() 
        else:
            # Otherwise, run a new event loop
            result = asyncio.run(self._advance_rag_chatbot_async(question, history))
        
        # profiler.disable()
        # profiler.print_stats(sort="time")
        return result