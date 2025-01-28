"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""
import asyncio
import logging
import traceback
from typing import List
from typing import Optional
from typing import Tuple

from datasets import Dataset
from datasets import Sequence
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy
from ragas.metrics import faithfulness
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import LLMContextRecall
from ragas.metrics import ResponseRelevancy

from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from hybrid_rag.src.utils.logutils import Logger

from functools import lru_cache
import cachetools
# from ragas.metrics.critique import harmfulness, correctness


class RAGAEvaluator:
    """
    A class for evaluating Retrieval-Augmented Generation (RAG) models using metrics from the RAGAS framework.

    Attributes:
        dense_embedding_model (str): Name of the dense embedding model.
        llmModelInstance (LLMModelInitializer): An initialized LLM model instance.
        logger (logging.Logger): Logger instance for logging information and errors.
    """
    def __init__(
        self,
        dense_embedding_model: str,
        llmModelInstance: LLMModelInitializer,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the RAG Evaluator class with required configurations.

        Args:
            dense_embedding_model (str): Name of the dense embedding model.
            llmModelInstance (LLMModelInitializer): An instance of LLMModelInitializer.
            logger (Optional[logging.Logger]): Logger instance for logging. Defaults to None.
        """

        self.logger = logger if logger else Logger().get_logger()
        self.llmModelInstance = llmModelInstance
        self.llmModel_initializer = llmModelInstance.initialise_llm_model()
        self.langchainLLMWrapper = self._get_llm_model()
        self.dense_embedding_model = dense_embedding_model
        self.batch_size = 2

    @lru_cache(maxsize=2)  # Cache the embedding model
    def _get_embedding_model(self) -> FastEmbedEmbeddings:
        """
        Load and cache the dense embedding model.

        Returns:
            FastEmbedEmbeddings: The initialized embedding model.
        """
        self.logger.info(f"Loading dense embedding model: {self.dense_embedding_model}")
        return FastEmbedEmbeddings(model_name=self.dense_embedding_model)
    
    @lru_cache(maxsize=1)
    def _get_llm_model(self) -> LangchainLLMWrapper:
        self.logger.info("Loading LLM model...")
        return LangchainLLMWrapper(self.llmModel_initializer)
    
    def _validate_column_dtypes(self, ds: Dataset) -> str:
        """
        Validate dataset column types for compatibility with the RAGAS framework.

        Args:
            ds (Dataset): The dataset to validate.

        Returns:
            str: 'PASS' if validation succeeds, 'FAIL' otherwise.
        """
        try:
            # Validate string columns
            for column_name in ["question", "answer", "ground_truth"]:
                if column_name in ds.features:
                    # Directly check dtype for Value fields
                    if ds.features[column_name].dtype != "string":
                        self.logger.error(
                            f'Dataset feature "{column_name}" should be of type string, '
                            f"but got {ds.features[column_name].dtype}"
                        )
                        return "FAIL"

            # Validate Sequence[string] columns
            for column_name in ["contexts"]:
                if column_name in ds.features:
                    feature = ds.features[column_name]
                    # Check if it's a Sequence with a nested dtype
                    if not (
                        isinstance(feature, Sequence)
                        and feature.feature.dtype == "string"
                    ):
                        self.logger.error(
                            f'Dataset feature "{column_name}" should be of type Sequence[string], '
                            f'but got {type(feature)} with feature type {getattr(feature, "feature", None)}'
                        )
                        return "FAIL"

            # If all checks pass
            return "PASS"
        except Exception:
            self.logger.error(
                f"Failed to validate dataset columns: {traceback.format_exc()}"
            )
            return "FAIL"

    async def context_precision_without_reference(
        self, input: str, answer: str, context: List[str]
    ):
        """
        Calculate context precision without reference.

        Args:
            input (str): User input.
            answer (str): LLM response.
            context (List[str]): Retrieved contexts.

        Returns:
            float: Context precision score.
        """
        sample = SingleTurnSample(
            user_input=input,
            response=answer,
            retrieved_contexts=context,
        )

        context_precision = LLMContextPrecisionWithoutReference(
            llm=self.langchainLLMWrapper
        )
        scorer = await context_precision.single_turn_ascore(sample)
        return scorer

    async def answer_relevancy(self, input: str, answer: str, context: List[str]):
        """
        Calculate answer relevancy.

        Args:
            input (str): User input.
            answer (str): LLM response.
            context (List[str]): Retrieved contexts.

        Returns:
            float: Answer relevancy score.
        """ 
        sample = SingleTurnSample(
            user_input=input,
            response=answer,
            retrieved_contexts=context,
        )

        answer_relevancy = ResponseRelevancy(llm=self.langchainLLMWrapper)
        scorer = await answer_relevancy.single_turn_ascore(sample)
        return scorer

    def _prepare_context_for_ragas(self, documents: List[Document]) -> List[List[str]]:
        """
        Prepare document contexts for RAGAS evaluation.

        Args:
            documents (List[Document]): List of documents.

        Returns:
            List[List[str]]: Nested list of document contents.
        """
        result = []
        for doc in documents:
            result.append(doc.page_content)
        return [result]

    async def evaluate_rag_async(
        self, question: List[str], answer: List[str], context: List[Document]
    ) -> Tuple[dict, float]:
        """
        Evaluate RAG performance asynchronously.

        Args:
            question (List[str]): Questions.
            answer (List[str]): Answers.
            context (List[Document]): Retrieved contexts.

        Returns:
            Tuple[dict, float]: Evaluation results and context precision score.
        """
        try:
            # Create dataset
            contexts = self._prepare_context_for_ragas(context)
            data = {"question": question, "answer": answer, "contexts": contexts}
            rag_dataset = Dataset.from_dict(data)

            # Validate dataset columns
            status = self._validate_column_dtypes(rag_dataset)
            if status != "PASS":
                raise ValueError("Dataset validation failed")

            self.logger.info("Successfully validated the dataset for RAG evaluation.")

            evaluation_embeddings = self._get_embedding_model()
            # Perform evaluation
            result = evaluate(
                rag_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                ],  # context_utilization, harmfulness, correctness
                llm=self.langchainLLMWrapper,
                embeddings=evaluation_embeddings,
                batch_size=self.batch_size,
            )
            self.logger.info(f"type of result: {result}")
            # result['context_precision'] = context_precision
            # result['answer_relevancy'] = answer_relevancy
            self.logger.info(
                "Successfully evaluated the questions, answers, and context."
            )
            context_precision=0.0
            return (result, context_precision)

        except Exception as e:
            error = str(e)
            self.logger.error(
                f"ERROR : TRACEBACK: {traceback.format_exc()}"
            )
            raise
    
    def evaluate_rag(self, question: List[str], answer: List[str], context: List[Document]) -> Tuple[dict, float]:
        """
        Synchronous Evaluate RAG performance.

        Args:
            question (List[str]): Questions.
            answer (List[str]): Answers.
            context (List[Document]): Retrieved contexts.

        Returns:
            Tuple[dict, float]: Evaluation results and context precision score.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already running an event loop, use run_coroutine_threadsafe
                self.logger.info("Running in an existing event loop.")
                future = asyncio.run_coroutine_threadsafe(
                    self.evaluate_rag_async(question, answer, context), loop
                )
                return future.result()
            else:
                # If no event loop is running, create a new one
                self.logger.info("Starting a new event loop.")
                return asyncio.run(self.evaluate_rag_async(question, answer, context))
        except Exception as e:
            self.logger.error(f"Error in async_rag: {e}")
            raise
