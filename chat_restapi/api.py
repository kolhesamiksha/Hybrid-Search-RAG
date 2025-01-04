# Create a .env.example file with all Env variables added

import os
os.system("pip install hybrid_rag-0.0.1-py3-none-any.whl")

from typing import List
from typing import Tuple

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi import Response
from hybrid_rag.src.config import Config
from hybrid_rag.src.rag import RAGChatbot
from hybrid_rag.src.utils import Logger
from pydantic import BaseModel

rag_router = APIRouter()


class ResponseSchema(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []

@rag_router.post("/chatbot")
async def pred(response: Response, elements: ResponseSchema):
    load_dotenv(dotenv_path=".env.example")

    question = "tell me about supply chain consulting"
    history = []
    logger = Logger().get_logger()
    config = Config()
    chatbot_instance = RAGChatbot(config, logger)
    prediction = chatbot_instance.advance_rag_chatbot(question, history)
    return prediction
