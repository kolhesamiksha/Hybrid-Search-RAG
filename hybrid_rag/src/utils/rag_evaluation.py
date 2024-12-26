from typing import List
import traceback

from ragas.metrics import faithfulness
#from ragas.metrics.critique import harmfulness, correctness
from langchain_openai import ChatOpenAI
from ragas import evaluate
from langchain_core.documents import Document
from datasets import Dataset, Sequence
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from hybrid_rag.src.utils.logutils import Logger
logger = Logger().get_logger()

class RAGEvaluator:
    def __init__(self, llm_model_name: str, openai_api_base: str, groq_api_key: str, dense_embedding_model: str):
        """Initialize the RAG Evaluator class with required configuration"""
        self.llm_model_name = llm_model_name
        self.openai_api_base = openai_api_base
        self.__groq_api_key = groq_api_key
        self.dense_embedding_model = dense_embedding_model

    def _validate_column_dtypes(self, ds: Dataset) -> str:
        """Validate the dataset's column types against expected RAGAS framework"""
        try:
            for column_name in ["question", "answer", "ground_truth"]:
                if column_name in ds.features:
                    if ds.features[column_name].feature.dtype != "string":
                        raise ValueError(
                            f'Dataset feature "{column_name}" should be of type string'
                        )
                        return "FAIL"
            
            for column_name in ["contexts"]:
                if column_name in ds.features:
                    if not (
                        isinstance(ds.features[column_name], Sequence)
                        and ds.features[column_name].feature.dtype == "string"
                    ):
                        raise ValueError(
                            f'Dataset feature "{column_name}" should be of type Sequence[string], got {type(ds.features[column_name])}'
                        )
            
            return "PASS"
        except Exception as e:
            logger.error(f"Failed to validate dataset columns: {traceback.format_exc()}")
            return "FAIL"

    def evaluate_rag(self, question: List[str], answer: List[str], context: List[List[str]]) -> dict:
        """Evaluate the RAG model with given questions, answers, and context"""
        try:
            # Create dataset
            data = {
                "question": question,
                "answer": answer,
                "contexts": context
            }
            rag_dataset = Dataset.from_dict(data)

            # Validate dataset columns
            status = self._validate_column_dtypes(rag_dataset)
            if status != "PASS":
                raise ValueError("Dataset validation failed")
            
            logger.info("Successfully validated the dataset for RAG evaluation.")

            # Initialize LLM and embeddings
            llm_chat = ChatOpenAI(
                model=self.llm_model_name,
                openai_api_base=self.openai_api_base,
                openai_api_key=self.__groq_api_key
            )

            evaluation_chat_model = LangchainLLMWrapper(llm_chat)
            evaluation_embeddings = FastEmbedEmbeddings(model_name=self.dense_embedding_model)

            # Perform evaluation
            result = evaluate(
                rag_dataset,
                metrics=[faithfulness, context_utilization, harmfulness, correctness],
                llm=evaluation_chat_model,
                embeddings=evaluation_embeddings
            )
            
            logger.info("Successfully evaluated the questions, answers, and context.")
            return result

        except Exception as e:
            error = str(e)
            logger.error(f"Failed to evaluate RAG: {error} -> TRACEBACK: {traceback.format_exc()}")
            raise
