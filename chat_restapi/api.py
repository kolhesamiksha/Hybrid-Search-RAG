# to install the env variables from .env file
from typing import List
from typing import Tuple

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi import Response
from hybrid_rag.src.config import Config
from hybrid_rag.src.rag import RAGChatbot
from hybrid_rag.src.utils import Logger
from pydantic import BaseModel

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
rag_router = APIRouter()


class ResponseSchema(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []


@rag_router.post("/predict")
async def pred(response: Response, elements: ResponseSchema):
    load_dotenv()
    question = "tell me about supply chain consulting"
    history = []
    logger = Logger().get_logger()
    config = Config()
    chatbot_instance = RAGChatbot(config, logger)
    prediction = chatbot_instance.advance_rag_chatbot(question, history)
    return prediction
