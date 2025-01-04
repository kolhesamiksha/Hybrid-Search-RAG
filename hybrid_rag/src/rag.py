"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import os
import sys
import time
import traceback
import warnings
import logging

import mlflow
import subprocess

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

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

# TODO:add Logger & exceptions
warnings.filterwarnings("ignore")


class RAGChatbot:
    """
    A class that encapsulates the functionality of a Retrieval-Augmented Generation (RAG) chatbot.
    The chatbot can process questions, handle content moderation, perform hybrid search, and interact with a language model.
    """

    def __init__(self, config: Config, logger: Optional[logging.Logger], **kwargs:dict) -> None:
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
        self.mlflow_tracking_uri = self.config.MLFLOW_TRACKING_URI
        self.mlflow_experiment = self.config.MLFLOW_EXPERIMENT_NAME
        self._setup_mlflow()
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.langchain.autolog(log_models=True,log_input_examples=True)

    def _setup_mlflow(self):
        mlflow_command = [
            "mlflow",
            "server",
            "--backend-store-uri", "sqlite:///mlflow.db",  # Use SQLite database
            "--default-artifact-root", "./mlruns",        # Artifact storage path
            "--host", "0.0.0.0",                          # Bind to all network interfaces
            "--port", "5000"                              # Port to run the server on
        ]

        try:
            # Start the MLflow server
            print("Starting MLflow server...")
            subprocess.Popen(mlflow_command)
            print("MLflow server started on http://localhost:5000")
        except Exception as e:
            print(f"Failed to start MLflow server: {e}")

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
        if retrieved_history:
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

    def advance_rag_chatbot(
        self, question: str, history: List[str]
    ) -> Tuple[str, float, List[Any], dict, dict]:
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
        try:
            # Detect the content type of the question using the moderator.
            content_type = self.questionModerator.detect(
                question, self.config.QUESTION_MODERATION_PROMPT
            )
            content_type = content_type.dict()
            self.logger.info(f"Question Moderation Response:{content_type['content']}")
            # If the question is irrelevant, return an error message.
            if content_type["content"] == "IRRELEVANT-QUESTION":
                end_time = time.time() - st_time
                response = "Detected harmful content in the Question, Please Rephrase your question and Provide meaningful Question."
                self.logger.info("IRRELEVANT-QUESTION")
                return (response, end_time, [], {}, {})

            # Expand the query and retrieve results.
            expanded_queries = self.customQueryExpander.expand_query(question)
            self_query, metadata_filters = self.selfQueryRetrieval.retrieve_query(
                question
            )
            expanded_queries.append(self_query)
            self.logger.info(f"Expanded Queries are: {expanded_queries}")
            self.logger.info(
                f"Self Query Generated: {self_query}, Metadata Filters: {metadata_filters}"
            )
            combined_results = []
            for query in expanded_queries:
                output = self.milvusHybridSearch.hybrid_search(
                    query, self.config.HYBRID_SEARCH_TOPK
                )
                combined_results.extend(output)
            self.logger.info(f"Combine results: {combined_results}")
            self.logger.info(f"Type of Combined results: {type(combined_results)}")
            if self.config.IS_RERANK:
                # Rank and format the documents.
                reranked_docs = self.documentReranker.rerank_docs(
                    question, combined_results, self.config.RERANK_TOPK
                )
                self.logger.info(f"TYPE OF RERANKED_DOCS1: {type(reranked_docs)}")
            else:
                reranked_docs = combined_results[-self.config.RERANK_TOPK:]
                self.logger.info(f"TYPE OF RERANKED_DOCS2: {type(reranked_docs)}")
            formatted_context = self.docFormatter.format_docs(reranked_docs)

            # Generate the chatbot response.
            response, token_usage = self._chatbot(question, formatted_context, history)

            if self.config.IS_EVALUATE:
                evaluated_results = self.ragaEvaluator.evaluate_rag(
                    [question], [response], combined_results
                )
            else:
                evaluated_results = ()
            total_time = time.time() - st_time
            return (
                response,
                total_time,
                combined_results,
                evaluated_results,
                token_usage,
            )

        except Exception:
            print(f"ERROR: {traceback.format_exc()}")
            end_time = time.time() - st_time
            return ("ERROR", end_time, [], {}, {})
