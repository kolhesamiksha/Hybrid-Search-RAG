"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import json
import logging
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

import mlflow.pyfunc
import pandas as pd
import psutil
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec
from mlflow.types import Schema

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# LECL chain modules
from langchain.callbacks import get_openai_callback
from langchain_core.runnables import (
    RunnablePassthrough,
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
from hybrid_rag.src.utils import (
    DocumentFormatter,
    Logger,
)
from hybrid_rag.src.vectordb import VectorStoreManager
from hybrid_rag.src.custom_mlflow.predict import RAGChatbotModel
from hybrid_rag.src.utils.utils import calculate_cost_groq_llama31

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

        # Initialize dependencies using the provided self.config
        self.supportPromptGenerator = SupportPromptGenerator(
            self.config.LLM_MODEL_NAME,
            self.config.MASTER_PROMPT,
            self.config.MODEL_SPECIFIC_PROMPT_USER_TAG,
            self.config.MODEL_SPECIFIC_PROMPT_SYSTEM_TAG,
            self.config.MODEL_SPECIFIC_PROMPT_ASSISTANT_TAG,
            self.logger,
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
            self.vectorDBInitializer,
            self.logger,
        )
        self.selfQueryRetrieval = SelfQueryRetrieval(
            self.config.COLLECTION_NAME,
            self.config.DENSE_SEARCH_PARAMS,
            self.config.DENSE_EMBEDDING_MODEL,
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
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
        self.mlflow_tracking_uri = self.config.MLFLOW_TRACKING_URI
        self.mlflow_experiment = self.config.MLFLOW_EXPERIMENT_NAME
        self.mlflow_run_name = self.config.MLFLOW_RUN_NAME
        self._setup_mlflow()
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.log_param("LLM_MODEL_NAME", os.getenv("LLM_MODEL_NAME"))
        mlflow.log_param("DENSE_EMBEDDING_MODEL", os.getenv("DENSE_EMBEDDING_MODEL"))
        mlflow.log_param("SPARSE_EMBEDDING_MODEL", os.getenv("SPARSE_EMBEDDING_MODEL"))
        mlflow.log_param("TOP_K", os.getenv("HYBRID_SEARCH_TOPK"))
        mlflow.log_param("TEMPERATURE", os.getenv("TEMPERATURE"))
        mlflow.log_param("TOP_P", os.getenv("TOP_P"))
        mlflow.log_param("FREQUENCY_PENALTY", os.getenv("FREQUENCY_PENALTY"))
        mlflow.langchain.autolog(log_traces=True)

        if isinstance(os.getenv("DENSE_SEARCH_PARAMS"), dict):
            for key, value in os.getenv("DENSE_SEARCH_PARAMS").items():
                mlflow.log_param(key, value)
                print(
                    f"Logged {len(os.getenv('DENSE_SEARCH_PARAMS'))} parameters to MLflow."
                )
            else:
                print("dense_search_params is not a dictionary.")

        if isinstance(os.getenv("SPARSE_SEARCH_PARAMS"), dict):
            for key, value in os.getenv("SPARSE_SEARCH_PARAMS").items():
                mlflow.log_param(key, value)
                print(
                    f"Logged {len(os.getenv('SPARSE_SEARCH_PARAMS'))} parameters to MLflow."
                )
            else:
                print("spase_search_params is not a dictionary.")
        # mlflow.langchain.autolog(log_models=True,log_input_examples=True)

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
            print("Starting MLflow server...")
            subprocess.Popen(mlflow_command)
            print("MLflow server started on http://localhost:5000")
        except Exception as e:
            print(f"Failed to start MLflow server: {e}")

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

    def _chatbot(
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
        self.logger.info("Successfully Initlaised the Chain.")
        try:
            # Use the OpenAI callback to monitor API usage.
            with get_openai_callback() as cb:
                response = chain.invoke(
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
                print(f"Skipping non-Document object: {document}")
        return converted_results

    def _add_json_results(self, retrieved_results: list, file_name: str) -> None:
        temp_folder = "dummy_to_delete"
        os.makedirs(temp_folder, exist_ok=True)

        try:
            # Open the file in write mode and dump the results
            with open(f"dummy_to_delete/{file_name}", "w") as json_file:
                json.dump(retrieved_results, json_file, indent=4)
            self.logger.info(f"Successfully wrote combined results to {file_name}")
        except Exception as e:
            self.logger.error(
                f"Failed to write combined results to {file_name}: {str(e)}"
            )
            raise

    def _garbage_collector(self, folder_name: str) -> None:
        try:
            if os.path.exists(folder_name):
                os.rmdir(folder_name)
            else:
                print(f"The directory '{folder_name}' does not exist.")
        except Exception as e:
            print(f"An error occurred while deleting the directory: {e}")

    def advance_rag_chatbot(
        self, question: str, history: List[str]
    ) -> Tuple[str, float, List[Any], dict, float]:
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
        if mlflow.active_run() is not None:
            mlflow.end_run()

        with mlflow.start_run(run_name=self.mlflow_run_name):
            st_time = time.time()
            mlflow.log_param("user_question", question)
            try:
                # Detect the content type of the question using the moderator.
                content_type = self.questionModerator.detect(
                    question, self.config.QUESTION_MODERATION_PROMPT
                )
                content_type = content_type.dict()
                self.logger.info(
                    f"Question Moderation Response:{content_type['content']}"
                )
                # If the question is irrelevant, return an error message.
                if content_type["content"] == "IRRELEVANT-QUESTION":
                    end_time = time.time() - st_time
                    response = "Detected harmful content in the Question, Please Rephrase your question and Provide meaningful Question."
                    self.logger.info("IRRELEVANT-QUESTION")
                    total_cost = 0.0
                    return (response, end_time, [], {}, total_cost)

                # Expand the query and retrieve results.
                expanded_queries = self.customQueryExpander.expand_query(question)
                self_query, metadata_filters = self.selfQueryRetrieval.retrieve_query(
                    question
                )
                metadata_filters_str = json.dumps(metadata_filters)
                mlflow.log_param("self_query_metadata_filters", metadata_filters_str)
                expanded_queries.append(self_query)
                self.logger.info(f"Expanded Queries are: {expanded_queries}")
                self.logger.info(
                    f"Self Query Generated: {self_query}, Metadata Filters: {metadata_filters}"
                )
                combined_results = []
                results_to_store = []
                for query in expanded_queries:
                    output = self.milvusHybridSearch.hybrid_search(
                        query, self.config.HYBRID_SEARCH_TOPK
                    )
                    output_to_mlflow = self._convert_documents_to_dicts(output)
                    combined_results.extend(output)
                    results_to_store.append(
                        {"question": query, "output": output_to_mlflow}
                    )

                self._add_json_results(
                    results_to_store, "retrieved_questions_results.json"
                )
                mlflow.log_artifact("dummy_to_delete/retrieved_questions_results.json")

                mlflow.log_param("retrieved_docs", len(combined_results))
                self.logger.info(f"Combine results: {combined_results}")
                self.logger.info(f"Type of Combined results: {type(combined_results)}")
                if self.config.IS_RERANK:
                    # Rank and format the documents.
                    reranked_docs = self.documentReranker.rerank_docs(
                        question, combined_results, self.config.RERANK_TOPK
                    )
                    self.logger.info(f"TYPE OF RERANKED_DOCS1: {type(reranked_docs)}")
                else:
                    reranked_docs = combined_results[-self.config.RERANK_TOPK :]
                    self.logger.info(f"TYPE OF RERANKED_DOCS2: {type(reranked_docs)}")

                self.logger.info(f"TOTAL RERANKED DOCS: {len(reranked_docs)}")
                reranker_to_mlflow = self._convert_documents_to_dicts(reranked_docs)
                self._add_json_results(reranker_to_mlflow, "reranked_results.json")
                mlflow.log_artifact("dummy_to_delete/reranked_results.json")
                formatted_context = self.docFormatter.format_docs(reranked_docs)

                # Generate the chatbot response.
                response, token_usage = self._chatbot(
                    question, formatted_context, history
                )
                if self.config.IS_EVALUATE:
                    evaluated_results = self.ragaEvaluator.evaluate_rag(
                        [question], [response], combined_results
                    )
                else:
                    evaluated_results = ()

                if token_usage:
                    total_cost = calculate_cost_groq_llama31(
                        token_usage,
                        self.config.INPUT_TOKENS_PER_MILLION_COST,
                        self.config.INPUT_TOKENS_PER_MILLION_COST,
                    )
                else:
                    total_cost = 0.0
                system_metrics = self._get_system_metrics()
                mlflow.log_metric("cpu_percent", system_metrics["cpu_percent"])
                mlflow.log_metric(
                    "memory_used_percent", system_metrics["memory"]["percent"]
                )
                mlflow.log_metric(
                    "disk_used_percent", system_metrics["disk"]["percent"]
                )
                mlflow.log_metric(
                    "network_bytes_sent", system_metrics["network"]["bytes_sent"]
                )
                mlflow.log_metric(
                    "network_bytes_received",
                    system_metrics["network"]["bytes_received"],
                )
                total_time = time.time() - st_time

                mlflow.log_metric("total_response_time", total_time)
                mlflow.log_metric(
                    "completion_tokens",
                    token_usage["token_usage"].get("completion_tokens", 0),
                )
                mlflow.log_metric(
                    "prompt_tokens", token_usage["token_usage"].get("prompt_tokens", 0)
                )
                mlflow.log_metric(
                    "prompt_time", token_usage["token_usage"].get("prompt_time", 0)
                )
                mlflow.log_metric(
                    "completion_time",
                    token_usage["token_usage"].get("completion_time", 0),
                )
                mlflow.log_metric(
                    "queue_time", token_usage["token_usage"].get("queue_time", 0)
                )
                mlflow.log_metric(
                    "total_tokens", token_usage["token_usage"].get("total_tokens", 0)
                )
                mlflow.log_metric(
                    "total_chain_time", token_usage["token_usage"].get("total_time", 0)
                )
                mlflow.log_metric("total_cost", total_cost)
                mlflow.log_param("response", response)
                if evaluated_results:
                    if len(evaluated_results) > 1:
                        mlflow.log_metric(
                            "answer_relevancy",
                            evaluated_results[0].get("answer_relevancy", 0),
                        )
                        mlflow.log_metric(
                            "faithfulness", evaluated_results[0].get("faithfulness", 0)
                        )
                        mlflow.log_metric("context_precision", evaluated_results[1])
                else:
                    self.logger.warning(
                        "Evaluated results tuple is empty. Skipping MLflow logging."
                    )

                self._garbage_collector("dummy_to_delete")
                # self.log_model(question,history)
                return (
                    response,
                    total_time,
                    combined_results,
                    evaluated_results,
                    total_cost,
                )
            except Exception:
                self._garbage_collector("dummy_to_delete")
                print(f"ERROR: {traceback.format_exc()}")
                end_time = time.time() - st_time
                total_cost = 0.0
                return ("ERROR", end_time, [], {}, total_cost)
        mlflow.end_run()

    # # Custom Mlflow Model Logging
    # def log_model(self, question, history):
    #     """
    #     This method is called during model loading. It initializes the chatbot instance.
    #     """
    #     # Define input/output schema
    #     input_schema = Schema([
    #         ColSpec("string", "question"),
    #         ColSpec("string", "history")
    #     ])
    #     output_schema = Schema([
    #         ColSpec("string")
    #     ])
    #     signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    #     # Log parameters
    #     mlflow.log_param("model_type", "RAGChatbot")
    #     mlflow.log_param("config", self.config.__dict__)

    #     model_input = {"question": pd.Series(question), "history": history}
    #     model_input["question"] = model_input["question"].tolist()
    #     # Log input/output example
    #     # input_example = pd.DataFrame([{
    #     #     "question": question,
    #     #     "history": history
    #     # }])
    #     self.requirements = [
    #         'cloudpickle==3.1.0',
    #         'mlflow==2.19.0',
    #         'cryptography==44.0.0',
    #         'dnspython==1.16.0',
    #         'jaraco-classes==3.4.0',
    #         'jaraco-collections==5.1.0',
    #         'lark==1.2.2',
    #         'numpy==1.26.4',
    #         'pandas==2.2.3',
    #         'platformdirs==4.3.6',
    #         'psutil==6.1.1',
    #         'pyarrow==18.1.0',
    #         'rich==13.9.4',
    #         'tornado==6.4.2',
    #         'langchain-community==0.2.13',
    #         'langchain==0.2.12',
    #         'fastembed==0.3.2',
    #         'langchain-core==0.2.43',
    #         'langchain-openai==0.1.25',
    #         'langchain-groq==0.1.10',
    #         'openai==1.58.1',
    #         'tiktoken==0.8.0',
    #         'pymilvus==2.5.1',
    #         'faiss-cpu==1.9.0.post1',
    #         'fastapi==0.115.6',
    #         'uvicorn==0.34.0',
    #         'pycryptodome==3.21.0',
    #         'ragas==0.2.9',
    #         'datasets==3.2.0',
    #         'flashrank==0.2.9',
    #         'PyGithub==2.5.0',
    #         'lark==1.2.2',
    #         'ipython==8.31.0',
    #         'pymongo==3.11',  # pymongo with extras
    #         'python-dotenv==1.0.1',
    #     ]
    #     dotenv_path = "./.env.example"
    #     artifact_dir = "model_artifacts"
    #     if os.path.isfile(artifact_dir):
    #         os.remove(artifact_dir)

    #     os.makedirs(artifact_dir, exist_ok=True)

    #     # Copy the .env.example file to the artifact directory
    #     shutil.copy(dotenv_path, os.path.join(artifact_dir, ".env.example"))

    #     rag_model = RAGChatbotModel()
    #     # Log the model as a PyFunc
    #     mlflow.pyfunc.log_model(
    #         artifact_path="rag_chatbot_model_1",
    #         python_model=rag_model,
    #         signature=signature,
    #         input_example=model_input,
    #         artifacts={"env_file": f"{artifact_dir}/.env.example"},
    #         pip_requirements=self.requirements
    #     )
