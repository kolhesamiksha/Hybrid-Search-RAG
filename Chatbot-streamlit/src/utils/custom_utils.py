# SparseEmbeddiing Class modules
from pydantic import BaseModel, Field, model_validator, validator
from langchain_core.embeddings import Embeddings

# Custom Query Expansion Class modules
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Optional, Tuple, Literal, Any, Dict
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import BasePromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import re

class SparseFastEmbedEmbeddings(BaseModel, Embeddings):
    """Qdrant FastEmbedding models.
    FastEmbed is a lightweight, fast, Python library built for embedding generation.

    To use this class, you must install the `fastembed` Python package.

    `pip install fastembed`
    Example:
        from langchain_community.embeddings import FastEmbedEmbeddings
        fastembed = FastEmbedEmbeddings()
    """

    model_name: str = "BAAI/bge-small-en-v1.5"
    """Name of the FastEmbedding model to use
    Defaults to "BAAI/bge-small-en-v1.5"
    Find the list of supported models at
    https://qdrant.github.io/fastembed/examples/Supported_Models/
    """

    cache_dir: Optional[str] = Field(default=None)
    """The path to the cache directory.
    Defaults to `local_cache` in the parent directory
    """

    threads: Optional[int] = Field(default=None)
    """The number of threads single onnxruntime session can use.
    Defaults to None
    """

    doc_embed_type: str = "default"
    """Type of embedding to use for documents
    The available options are: "default" and "passage"
    """

    model: Any = Field(default=None, exclude=True)  # Renamed to 'model' and marked as private

    class Config:
        """Configuration for this pydantic object."""
        extra = 'forbid'

    @model_validator(mode='before')
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that FastEmbed has been installed."""
        return values

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the FastEmbed model."""
        try:
            # >= v0.2.0
            from fastembed import SparseTextEmbedding

            self.model = SparseTextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
            )
        except ImportError as ie:
            try:
                # < v0.2.0
                from fastembed.embedding import FlagEmbedding

                self.model = FlagEmbedding(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    threads=self.threads,
                )
            except ImportError:
                raise ImportError(
                    "Could not import 'fastembed' Python package. "
                    "Please install it with `pip install fastembed`."
                ) from ie

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents using FastEmbed.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings: List[np.ndarray]
        if self.doc_embed_type == "passage":
            embeddings = self.model.passage_embed(texts)
        else:
            embeddings = self.model.embed(texts)
        return [
            {int(idx): float(val) for idx, val in zip(embed.indices, embed.values)}
            for embed in embeddings
        ]

    def embed_query(self, text: str) -> List[float]:
        """Generate query embeddings using FastEmbed.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        query_embeddings: np.ndarray = next(self.model.query_embed(text))
        return [{int(idx): float(val) for idx, val in zip(query_embeddings.indices, query_embeddings.values)}]

# Custom Query Expansion Class
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        cleaned_lines = [re.sub(r'^\d+\.\s*', '', line) for line in lines]
        return cleaned_lines

class CustomMultiQueryRetriever(MultiQueryRetriever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    MULTI_QUERY_PROMPT = PromptTemplate(
        input_variables = ["question"],
        template="""You are an AI language model assistant. Your task is to break down question if possible to retrieve
       relevant documents from a vector database. By generating multiple break down questions of user question, your goal is to 
       help the user overcome some of the limitations of distance-based similarity search. Provide these multi step questions
       separated by newlines. Only break down question if poosible otherwise return the original question.\n Original Question: {question}
        """ 
    )

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: Any,
        prompt: BasePromptTemplate = MULTI_QUERY_PROMPT,
        include_original: bool = True,
    ) -> "CustomMultiQueryRetriever":

        output_parser = LineListOutputParser()
        llm_chain = prompt | llm | output_parser

        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            include_original=include_original
        )