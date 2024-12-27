import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict

@dataclass
class Config:
    """Configuration class to store all environment variables and constants"""

    # Language Model Configuration
    LLM_MODEL_NAME: str = field(default_factory=lambda: _get_env_var("LLM_MODEL_NAME", default_value="llama-3.1-70b-versatile"))
    DENSE_EMBEDDING_MODEL: str = field(default_factory=lambda: _get_env_var("DENSE_EMBEDDING_MODEL", default_value="jinaai/jina-embeddings-v2-base-en"))
    SPARSE_EMBEDDING_MODEL: str = field(default_factory=lambda: _get_env_var("SPARSE_EMBEDDING_MODEL", default_value="Qdrant/bm42-all-minilm-l6-v2-attentions"))

    GROQ_API_KEY:str = field(default_factory=lambda: _get_env_var("GROQ_API_KEY"))
    OPENAI_API_BASE: str = field(default_factory=lambda: _get_env_var("OPENAI_API_BASE"))
    # Chat History Configuration
    NO_HISTORY: int = field(default_factory=lambda: int(_get_env_var("CHAT_HISTORY", default_value=2, required=False)))

    # LangChain Usage Flag
    LANGCHAIN: bool = field(default_factory=lambda: _get_env_var("LANGCHAIN", default_value="True", required=False).lower() in ("true", "1", "yes"))

    # Dense Search Parameters
    DENSE_SEARCH_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
            "index_type": _get_env_var("DENSE_SEARCH_INDEX_TYPE", default_value="IVF_SQ8", required=False),
            "metric_type": _get_env_var("DENSE_SEARCH_METRIC_TYPE", default_value="L2", required=False),
            "params": {"nlist": int(_get_env_var("DENSE_SEARCH_NLIST", default_value=128, required=False))},
        }
    )

    # Sparse Search Parameters
    SPARSE_SEARCH_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
            "index_type": _get_env_var("SPARSE_SEARCH_INDEX_TYPE", default_value="SPARSE_INVERTED_INDEX", required=False),
            "metric_type": _get_env_var("SPARSE_SEARCH_METRIC_TYPE", default_value="IP", required=False),
        }
    )

    # Default Question
    QUESTION: str = field(default_factory=lambda: _get_env_var("QUESTION", default_value="What is Generative AI?"))

    # MongoDB Configuration
    CONNECTION_STRING: str = field(default_factory=lambda: _get_env_var("CONNECTION_STRING"))
    MONGO_COLLECTION_NAME: str = field(default_factory=lambda: _get_env_var("MONGO_COLLECTION_NAME"))  #Hybrid-search-rag
    DB_NAME: str = field(default_factory=lambda: _get_env_var("DB_NAME")) #credentials

    # Model Configuration Parameters
    TEMPERATURE: float = field(default_factory=lambda: float(_get_env_var("TEMPERATURE", default_value=0.0, required=False)))
    TOP_P: float = field(default_factory=lambda: float(_get_env_var("TOP_P", default_value=0.3, required=False)))
    FREQUENCY_PENALTY: float = field(default_factory=lambda: float(_get_env_var("FREQUENCY_PENALTY", default_value=1.0, required=False)))

    #Zillinz Milvus Credentials
    ZILLIZ_CLOUD_URI:str = field(default_factory=lambda: _get_env_var("ZILLIZ_CLOUD_URI"))
    ZILLIZ_CLOUD_API_KEY:str = field(default_factory=lambda: _get_env_var("ZILLIZ_CLOUD_API_KEY"))
    COLLECTION_NAME: str = field(default_factory=lambda: _get_env_var("COLLECTION_NAME"))
    
    # Hybrid Search Parameters
    HYBRID_SEARCH_TOPK: int = field(
        default_factory=lambda: int(_get_env_var("HYBRID_SEARCH_TOPK", default_value=6, required=False))
    )
    RERANK_TOPK: int = field(
        default_factory=lambda: int(_get_env_var("RERANK_TOPK", default_value=3, required=False))
    )

def _get_env_var(var_name: str, default_value: Any = "", required: bool = True) -> Any:
    """
    Retrieve an environment variable with optional requirement.

    Args:
        var_name (str): Name of the environment variable
        required (bool, optional): Whether the variable is required. Defaults to True.

    Returns:
        str: Value of the environment variable

    Raises:
        ValueError: If the variable is required but not set
    """
    value = os.getenv(var_name, default_value)
    if not value and required:
        raise ValueError(f"Required environment variable '{var_name}' is not set. Please make sure to export it.")
    if not value:
        print(f"Warning: Environment variable '{var_name}' is not set. Using default value: {default_value}")

    return value or ""
