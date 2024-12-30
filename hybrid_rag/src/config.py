"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict


@dataclass
class Config:
    """Configuration class to store all environment variables and constants"""

    # Language Model Configuration
    LLM_MODEL_NAME: str = field(
        default_factory=lambda: _get_env_var(
            "LLM_MODEL_NAME", default_value="llama-3.1-70b-versatile"
        )
    )
    DENSE_EMBEDDING_MODEL: str = field(
        default_factory=lambda: _get_env_var(
            "DENSE_EMBEDDING_MODEL", default_value="jinaai/jina-embeddings-v2-base-en"
        )
    )
    SPARSE_EMBEDDING_MODEL: str = field(
        default_factory=lambda: _get_env_var(
            "SPARSE_EMBEDDING_MODEL",
            default_value="Qdrant/bm42-all-minilm-l6-v2-attentions",
        )
    )

    GROQ_API_KEY: str = field(default_factory=lambda: _get_env_var("GROQ_API_KEY"))
    OPENAI_API_BASE: str = field(
        default_factory=lambda: _get_env_var("OPENAI_API_BASE")
    )
    # Chat History Configuration
    NO_HISTORY: int = field(
        default_factory=lambda: int(
            _get_env_var("CHAT_HISTORY", default_value=2, required=False)
        )
    )
    MASTER_PROMPT: str = field(
        default_factory=lambda: _get_env_var(
            "MASTER_PROMPT",
            default_value="""Please follow below instructions to provide the response:
            1. Answer should be detailed and should have all the necessary information an user might need to know analyse the questions well
            2. The user says "Hi" or "Hello." Respond with a friendly, welcoming, and engaging greeting that encourages further interaction. Make sure to sound enthusiastic and approachable.
            3. Make sure to address the user's queries politely.
            4. Compose a comprehensive reply to the query based on the CONTEXT given.
            5. Respond to the questions based on the given CONTEXT.
            6. Please refrain from inventing responses and kindly respond with "I apologize, but that falls outside of my current scope of knowledge."
            7. Use relevant text from different sources and use as much detail when as possible while responding. Take a deep breath and Answer step-by-step.
            8. Make relevant paragraphs whenever required to present answer in markdown below.
            9. MUST PROVIDE the Source Link above the Answer as Source: source_link.
            10. Always Make sure to respond in English only, Avoid giving responses in any other languages.""",
            required=False
        )
    )

    LLAMA3_SYSTEM_TAG: str = field(
        default_factory=lambda: _get_env_var("LLAMA3_SYSTEM_TAG", default_value="", required=False),
    )
    LLAMA3_USER_TAG: str = field(
        default_factory=lambda: _get_env_var("LLAMA3_USER_TAG", default_value="", required=False)
    )
    LLAMA3_ASSISTANT_TAG: str = field(
        default_factory=lambda: _get_env_var("LLAMA3_ASSISTANT_TAG", default_value="", required=False)
    )

    QUESTION_MODERATION_PROMPT: str = field(
        default_factory=lambda: _get_env_var(
            "QUESTION_MODERATION_PROMPT",
            default_value="""You are a Content Moderator working for a technology and consulting company, your job is to filter out the queries which are not irrelevant and does not satisfy the intent of the chatbot.
    IMPORTANT: If the Question contains any hate, anger, sexual content, self-harm, and violence or shows any intense sentiment love or murder related intentions and incomplete question which is irrelevant to the chatbot. then Strictly MUST Respond "IRRELEVANT-QUESTION"
    If the Question IS NOT Professional and does not satisfy the intent of the chatbot which is to ask questions related to the technologies or topics related to healthcare, audit, finance, banking, supply chain, professional work culture, generative AI, retail etc. then Strictly MUST Respond "IRRELEVANT-QUESTION".
    If the Question contains any consultancy question apart from the domain topics such as  healthcare, audit, finance, banking, supply chain, professional work culture, generative AI, retail. then Strictly MUST Respond "IRRELEVANT-QUESTION".
    else "NOT-IRRELEVANT-QUESTION"

    Examples:
    Question1: Are women getting equal opportunities in AI Innovation?
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
    Response6: IRRELEVANT-QUESTION""",
    required=False
        )
    )

    IS_EVALUATE: bool = field(
        default_factory=lambda: _get_env_var(
            "IS_EVALUATE", default_value="True", required=False
        ).lower()
        in ("true", "1", "yes")
    )

    # As the rarnk is more heavy operation
    IS_RERANK: bool = field(
        default_factory=lambda: _get_env_var(
            "IS_RERANK", default_value="True", required=False
        ).lower()
        in ("true", "1", "yes")
    )

    IS_GUARDRAIL: bool = field(
        default_factory=lambda: _get_env_var(
            "IS_GUARDRAIL", default_value="True", required=False
        ).lower()
        in ("true", "1", "yes")
    )

    IS_FOLLOWUP: bool = field(
        default_factory=lambda: _get_env_var(
            "IS_FOLLOWUP", default_value="True", required=False
        ).lower()
        in ("true", "1", "yes")
    )

    # LangChain Usage Flag
    LANGCHAIN: bool = field(
        default_factory=lambda: _get_env_var(
            "LANGCHAIN", default_value="True", required=False
        ).lower()
        in ("true", "1", "yes")
    )

    # Dense Search Parameters
    DENSE_SEARCH_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
            "index_type": _get_env_var(
                "DENSE_SEARCH_INDEX_TYPE", default_value="IVF_SQ8", required=False
            ),
            "metric_type": _get_env_var(
                "DENSE_SEARCH_METRIC_TYPE", default_value="L2", required=False
            ),
            "params": {
                "nlist": int(
                    _get_env_var(
                        "DENSE_SEARCH_NLIST", default_value=128, required=False
                    )
                )
            },
        }
    )

    # Sparse Search Parameters
    SPARSE_SEARCH_PARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
            "index_type": _get_env_var(
                "SPARSE_SEARCH_INDEX_TYPE",
                default_value="SPARSE_INVERTED_INDEX",
                required=False,
            ),
            "metric_type": _get_env_var(
                "SPARSE_SEARCH_METRIC_TYPE", default_value="IP", required=False
            ),
        }
    )

    # Default Question
    QUESTION: str = field(
        default_factory=lambda: _get_env_var(
            "QUESTION", default_value="What is Generative AI?"
        )
    )

    # MongoDB Configuration
    CONNECTION_STRING: str = field(
        default_factory=lambda: _get_env_var("CONNECTION_STRING")
    )
    MONGO_COLLECTION_NAME: str = field(
        default_factory=lambda: _get_env_var("MONGO_COLLECTION_NAME")
    )  # Hybrid-search-rag
    DB_NAME: str = field(default_factory=lambda: _get_env_var("DB_NAME"))  # credentials

    # Model Configuration Parameters
    TEMPERATURE: float = field(
        default_factory=lambda: float(
            _get_env_var("TEMPERATURE", default_value=0.0, required=False)
        )
    )
    TOP_P: float = field(
        default_factory=lambda: float(
            _get_env_var("TOP_P", default_value=0.3, required=False)
        )
    )
    FREQUENCY_PENALTY: float = field(
        default_factory=lambda: float(
            _get_env_var("FREQUENCY_PENALTY", default_value=1.0, required=False)
        )
    )

    # Zillinz Milvus Credentials
    ZILLIZ_CLOUD_URI: str = field(
        default_factory=lambda: _get_env_var("ZILLIZ_CLOUD_URI")
    )
    ZILLIZ_CLOUD_API_KEY: str = field(
        default_factory=lambda: _get_env_var("ZILLIZ_CLOUD_API_KEY")
    )
    COLLECTION_NAME: str = field(
        default_factory=lambda: _get_env_var("COLLECTION_NAME")
    )

    # Hybrid Search Parameters
    HYBRID_SEARCH_TOPK: int = field(
        default_factory=lambda: int(
            _get_env_var("HYBRID_SEARCH_TOPK", default_value=6, required=False)
        )
    )
    RERANK_TOPK: int = field(
        default_factory=lambda: int(
            _get_env_var("RERANK_TOPK", default_value=3, required=False)
        )
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
        raise ValueError(
            f"Required environment variable '{var_name}' is not set. Please make sure to export it."
        )
    if not value:
        print(
            f"Warning: Environment variable '{var_name}' is not set. Using default value: {default_value}"
        )

    return value or ""
