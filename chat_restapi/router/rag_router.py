# Create a .env.example file with all Env variables added
import logging
import os
import subprocess

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response
import asyncio

from .limiter import limiter
from chat_restapi.schema import ResponseSchema
from hybrid_rag.src.config import Config
from hybrid_rag.src.rag import RAGChatbot
from hybrid_rag.src.utils import Logger

# Construct the absolute path to the .whl file
whl_path = "chat_restapi/hybrid_rag-0.1.0-py3-none-any.whl"
subprocess.run(["pip", "install", whl_path], check=True)

# initialize the router
rag_router = APIRouter()


## Add dependency injection to make code more modular and easy to debug
def get_logger():
    return Logger().get_logger()


def get_config():
    load_dotenv(dotenv_path=".env.example")
    return Config()


# to check the health of API, for validation or auth errors
@rag_router.get("/health")
async def health_check():
    return {"status": "Super healthy"}


# Your other routes
@rag_router.get("/")
async def root():
    return {"message": "Welcome to the Hybrid RAG Chatbot!! Developed by SamikshaKolhe"}


@rag_router.post("/chatbot")
@limiter.limit("10/second")
async def pred(
    response: Response,
    request: Request,
    payload: ResponseSchema,
    config: Config = Depends(get_config),
    logger: logging.Logger = Depends(get_logger),
):
    question = payload.query
    history = payload.history
    chatbot_instance = RAGChatbot(config, logger)
    prediction = asyncio.run(
        chatbot_instance._advance_rag_chatbot_async(question, history)
    )
    return prediction[0]
