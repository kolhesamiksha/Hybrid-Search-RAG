"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import traceback
from typing import List
from typing import Optional
from typing import Tuple
import logging
import asyncio
from langchain_openai import ChatOpenAI

from datasets import Dataset
from datasets import Sequence
from hybrid_rag.src.utils.logutils import Logger
from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics import LLMContextRecall
from ragas.metrics import ResponseRelevancy

# from ragas.metrics.critique import harmfulness, correctness


class RAGAEvaluator:
    def __init__(
        self,
        dense_embedding_model: str,
        llmModelInstance: LLMModelInitializer,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the RAG Evaluator class with required configuration

        :param llm_model_name: LLM model name
        :param openai_api_base: RAGAS support ChatOpenAI API compatibility , provide openai_api_base e.g https://api.openai.com/v1
        :param groq_api_key: groq api key for ChatOpenAI
        :param dense_embedding_model: dense embedding model param for embeddings
        """

        self.logger = logger if logger else Logger().get_logger()
        self.llmModelInstance = llmModelInstance
        self.llmModel_initializer = self.llmModelInstance.initialise_llm_model()
        self.langchainLLMWrapper = LangchainLLMWrapper(self.llmModel_initializer)
        self.dense_embedding_model = dense_embedding_model

    def _validate_column_dtypes(self, ds: Dataset) -> str:
        """Validate the dataset's column types against expected RAGAS framework."""
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

    async def context_precision_without_reference(self, input:str, answer:str, context:List[str]):
        sample = SingleTurnSample(
            user_input=input,
            response=answer,
            retrieved_contexts=context, 
        )

        context_precision = LLMContextPrecisionWithoutReference(llm=self.langchainLLMWrapper)
        scorer = await context_precision.single_turn_ascore(sample)
        return scorer

    async def answer_relevancy(self, input:str, answer:str, context:List[str]):
        sample = SingleTurnSample(
            user_input=input,
            response=answer,
            retrieved_contexts=context, 
        )

        answer_relevancy = ResponseRelevancy(llm=self.langchainLLMWrapper)
        scorer = await answer_relevancy.single_turn_ascore(sample)
        return scorer

    def _prepare_context_for_ragas(self, documents: List[Document]) -> List[List[str]]:
        result = []
        for doc in documents:
            result.append(doc.page_content)
        return [result]

    def evaluate_rag(
        self, question: List[str], answer: List[str], context: List[Document]
    ) -> Tuple[dict, float]:
        """
        Evaluate the RAG model with given questions, answers, and context

        :param question: Question as List[str]
        :param answer: answer as List[str]
        :param context: context as List[List[str]]
        :return Dictionary with matrices and their values
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

            evaluation_embeddings = FastEmbedEmbeddings(
                model_name=self.dense_embedding_model
            )

            try:
                context_precision = asyncio.run(self.context_precision_without_reference(question[0], answer[0], contexts[0]))
                self.logger.info(f"Successfully Caluclated the Context Precision Metrics SCORE: {context_precision}")
            except Exception as e:
                error = str(e)
                self.logger.error("Failed to Calculate Context precision Metrics")
                raise

            # try:
            #     answer_relevancy = asyncio.run(self.answer_relevancy(question[0], answer[0], contexts[0]))
            #     self.logger.info(f"Successfully Caluclated the Answer Relevancy Metrics SCORE: {answer_relevancy}")
            # except Exception as e:
            #     error = str(e)
            #     self.logger.error("Failed to Calculate Answer Relevancy Metrics")
            #     raise
            
            # Perform evaluation
            result = evaluate(
                rag_dataset,
                metrics=[faithfulness,answer_relevancy],  # context_utilization, harmfulness, correctness
                llm=self.langchainLLMWrapper,
                embeddings=evaluation_embeddings,
            )
            self.logger.info(f"type of result: {result}")
            #result['context_precision'] = context_precision
            # result['answer_relevancy'] = answer_relevancy
            self.logger.info(
                "Successfully evaluated the questions, answers, and context."
            )
            return (result, context_precision)

        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to evaluate RAG: {error} -> TRACEBACK: {traceback.format_exc()}"
            )
            raise
