# Query Expansion modules
import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datetime import datetime
from typing import List
from typing import Tuple
from typing import Any

# FastAPI modules
from fastapi import APIRouter, Response
from pydantic import BaseModel

from hybrid_rag.src.config import Config
from hybrid_rag.src.rag import RAGChatbot
from hybrid_rag.src.utils import Logger

# st.set_option('global.cache.persist', True)
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = Logger().get_logger()

rag_router = APIRouter()


class ResponseSchema(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []


@rag_router.post("/predict")
async def pred(
    response: Response, elements: ResponseSchema
) -> Tuple[str, float, List[Any], dict, dict]:
    load_dotenv()
    question = "heyy mannna"
    history = []
    logger = Logger().get_logger()
    config = Config()
    chatbot_instance = RAGChatbot(config, logger)
    prediction = chatbot_instance.advance_rag_chatbot(question, history)
    return prediction
