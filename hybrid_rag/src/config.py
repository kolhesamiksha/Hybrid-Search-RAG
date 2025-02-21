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
from typing import List
from typing import Union


@dataclass
class Config:
    """Configuration class to store all environment variables and constants"""

    # Language Model Configuration
    LLM_MODEL_NAME: str = field(
        default_factory=lambda: _get_env_var(
            "LLM_MODEL_NAME", default_value="llama-3.1-70b-versatile"
        )
    )
    INPUT_TOKENS_PER_MILLION_COST: float = field(
        default_factory=lambda: _get_env_var(
            "INPUT_TOKENS_PER_MILLION_COST", default_value=0.0025
        )
    )
    OUTPUT_TOKENS_PER_MILLION_COST: float = field(
        default_factory=lambda: _get_env_var(
            "OUTPUT_TOKENS_PER_MILLION_COST", default_value=0.0064
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

    LLM_API_KEY: str = field(default_factory=lambda: _get_env_var("LLM_API_KEY"))
    PROVIDER_BASE_URL: str = field(
        default_factory=lambda: _get_env_var("PROVIDER_BASE_URL")
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
            required=False,
        )
    )

    MODEL_SPECIFIC_PROMPT_SYSTEM_TAG: str = field(
        default_factory=lambda: _get_env_var(
            "MODEL_SPECIFIC_PROMPT_SYSTEM_TAG", default_value="", required=False
        ),
    )
    MODEL_SPECIFIC_PROMPT_USER_TAG: str = field(
        default_factory=lambda: _get_env_var(
            "MODEL_SPECIFIC_PROMPT_USER_TAG", default_value="", required=False
        )
    )
    MODEL_SPECIFIC_PROMPT_ASSISTANT_TAG: str = field(
        default_factory=lambda: _get_env_var(
            "MODEL_SPECIFIC_PROMPT_ASSISTANT_TAG", default_value="", required=False
        )
    )

    MLFLOW_TRACKING_URI: str = field(
        default_factory=lambda: _get_env_var(
            "MLFLOW_TRACKING_URI", default_value="", required=False
        )
    )

    MLFLOW_EXPERIMENT_NAME: str = field(
        default_factory=lambda: _get_env_var(
            "MLFLOW_EXPERIMENT_NAME", default_value="hybrid_rag_exp", required=False
        )
    )
    MLFLOW_RUN_NAME: str = field(
        default_factory=lambda: _get_env_var(
            "MLFLOW_RUN_NAME", default_value="rag_chatbot_sam", required=False
        )
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
            required=False,
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

    DENSE_SEARCH_LIMIT:int = field(
        default_factory=lambda: int(
            _get_env_var("DENSE_SEARCH_LIMIT", default_value=3, required=False)
        )
    )
    SPARSE_SEARCH_LIMIT:int = field(
        default_factory=lambda: int(
            _get_env_var("SPARSE_SEARCH_LIMIT", default_value=3, required=False)
        )
    )
    
    NO_OF_QUESTIONS_TO_GENERATE: int = field(
        default_factory=lambda: int(
            _get_env_var("NO_OF_QUESTIONS_TO_GENERATE", default_value=5, required=False)
        )
    )

    SELF_RAG_METADATA_ATTRIBUTES: List[Dict[str, str]] = field(
        default_factory=lambda:[
            {
                "METADATA_ATTRIBUTE1_NAME": _get_env_var("METADATA_ATTRIBUTE1_NAME",default_value="",required=False),
                "METADATA_ATTRIBUTE1_DESCRIPTION": _get_env_var("METADATA_ATTRIBUTE1_DESCRIPTION",default_value="",required=False),
                "METADATA_ATTRIBUTE1_TYPE": _get_env_var("METADATA_ATTRIBUTE1_TYPE",default_value="",required=False),
            },
            {
                "METADATA_ATTRIBUTE2_NAME": _get_env_var("METADATA_ATTRIBUTE2_NAME",default_value="",required=False),
                "METADATA_ATTRIBUTE2_DESCRIPTION": _get_env_var("METADATA_ATTRIBUTE2_DESCRIPTION",default_value="",required=False),
                "METADATA_ATTRIBUTE2_TYPE": _get_env_var("METADATA_ATTRIBUTE2_TYPE",default_value="",required=False),
            },
            {
                "METADATA_ATTRIBUTE3_NAME": _get_env_var("METADATA_ATTRIBUTE3_NAME",default_value="",required=False),
                "METADATA_ATTRIBUTE3_DESCRIPTION": _get_env_var("METADATA_ATTRIBUTE3_DESCRIPTION",default_value="",required=False),
                "METADATA_ATTRIBUTE3_TYPE": _get_env_var("METADATA_ATTRIBUTE3_TYPE",default_value="",required=False),
            },
            {
                "METADATA_ATTRIBUTE4_NAME": _get_env_var("METADATA_ATTRIBUTE4_NAME",default_value="",required=False),
                "METADATA_ATTRIBUTE4_DESCRIPTION": _get_env_var("METADATA_ATTRIBUTE4_DESCRIPTION",default_value="",required=False),
                "METADATA_ATTRIBUTE4_TYPE": _get_env_var("METADATA_ATTRIBUTE4_TYPE",default_value="",required=False),
            },
            {
                "METADATA_ATTRIBUTE5_NAME": _get_env_var("METADATA_ATTRIBUTE5_NAME",default_value="",required=False),
                "METADATA_ATTRIBUTE5_DESCRIPTION": _get_env_var("METADATA_ATTRIBUTE5_DESCRIPTION",default_value="",required=False),
                "METADATA_ATTRIBUTE5_TYPE": _get_env_var("METADATA_ATTRIBUTE5_TYPE",default_value="",required=False),
            }
        ])
    
    MLFLOW_ASR_EXPERIMENT_NAME: str = field(
        default_factory=lambda: _get_env_var("MLFLOW_ASR_EXPERIMENT_NAME", default_value="asr_experiment", required=False)
    )
    MLFLOW_ASR_RUN_NAME: str = field(
        default_factory=lambda: _get_env_var("MLFLOW_ASR_RUN_NAME", default_value="asr_run", required=False)
    )

    SELF_RAG_DOCUMENTS_BRIEF_SUMMARY: str = field(
        default_factory=lambda: _get_env_var("SELF_RAG_DOCUMENTS_BRIEF_SUMMARY", default_value="ey company docs contains audit, tax, ai & supply chain domains", required=False), 
    )

    RERANK_TOPK: int = field(
        default_factory=lambda: int(
            _get_env_var("RERANK_TOPK", default_value=3, required=False)
        )
    )
    FOLLOWUP_TEMPLATE: str = field(
        default_factory=lambda:_get_env_var("FOLLOWUP_TEMPLATE", default_value="Generate 5 followup questions based on the question, context of the question and response for more comprehensive and relatable to the question.", required=False)
    )
    IS_ASR_LOCAL: bool = field(
        default_factory=lambda: _get_env_var(
            "IS_ASR_LOCAL", default_value="True", required=False
        ).lower()
        in ("true", "1", "yes")
    )

    ASR_LOCAL_MODEL_NAME: str = field(
        default_factory=lambda: _get_env_var("ASR_LOCAL_MODEL_NAME", default_value="ai4bharat/indicwav2vec-hindi", required=False)
    )

    IS_ASR_HUGGING_FACE: bool = field(
        default_factory=lambda: _get_env_var(
            "IS_ASR_HUGGING_FACE", default_value="True", required=False
        ).lower()
        in ("true", "1", "yes")
    )

    HUGGING_FACE_TOKEN: str = field(
        default_factory=lambda: _get_env_var("HUGGING_FACE_TOKEN", default_value="", required=False)
    )

    ASR_HG_MODEL_NAME: str = field(
        default_factory=lambda: _get_env_var("ASR_HG_MODEL_NAME", default_value="openai/whisper-small", required=False)
    )
    HUGGING_FACE_ENDPOINT: str = field(
        default_factory=lambda: _get_env_var("HUGGING_FACE_ENDPOINT", default_value="", required=False)
    )

    IS_ASR_MLFLOW: bool = field(
        default_factory=lambda: _get_env_var(
            "IS_ASR_MLFLOW", default_value="True", required=False
        ).lower()
        in ("true", "1", "yes")
    )

    MLFLOW_ASR_MODEL_NAME: str = field(
        default_factory=lambda: _get_env_var("MLFLOW_ASR_MODEL_NAME", default_value="vasista22/whisper-hindi-small", required=False)
    )

    METADATA_FILTERATION_FIELD: str = field(
        default_factory=lambda: _get_env_var("METADATA_FILTERATION_FIELD", default_value="topic", required=True)
    )

    MILVUS_ITERATOR_BATCH_SIZE: int = field(
        default_factory=lambda: _get_env_var("MILVUS_ITERATOR_BATCH_SIZE", default_value=50, required=False)
    )

    ITERATIVE_SEARCH_LIMIT: int = field(
        default_factory=lambda: _get_env_var("ITERATIVE_SEARCH_LIMIT", default_value=10000, required=False)
    )

    MAP_PROMPT: str = field(
        default_factory=lambda: _get_env_var("MAP_PROMPT", default_value="The following is a set of documents below \nBased on this list of docs, please identify the main themes.", required=False)
    )

    REDUCE_PROMPT: str = field(
        default_factory=lambda: _get_env_var("REDUCE_PROMPT", default_value="The following is set of summaries provided below \nTake these and distill it into a final, consolidated summary of the main themes.", required=False)
    )

    META_FILTERING_PROMPT: str = field(
        default_factory=lambda: _get_env_var("METADATA_FILTER_EXTRACTOR_FROM_QUESTION_PROMPT", default_value="topic", required=False)
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
