# Query Expansion modules
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import traceback
from datetime import datetime

# FastAPI modules
from fastapi import APIRouter, Response

from hybrid_rag.src.utils import ResponseSchema, Logger
from hybrid_rag.src.config import Config
from hybrid_rag.src.rag import RAGChatbot
from pydantic import BaseModel
from typing import List, Tuple

# st.set_option('global.cache.persist', True)
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = Logger().get_logger()

rag_router = APIRouter()

class ResponseSchema(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []

@rag_router.post("/predict")
async def pred(response: Response, elements: ResponseSchema):
    load_dotenv()
    question = "heyy mannna"
    history = []
    logger = Logger().get_logger()
    config = Config()
    chatbot_instance = RAGChatbot(config, logger)
    prediction  = chatbot_instance.advance_rag_chatbot(question, history)
    return prediction

