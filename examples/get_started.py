from dotenv import load_dotenv

from hybrid_rag import Config
from hybrid_rag import RAGChatbot
from hybrid_rag.src.utils import Logger

def chatbot():
    load_dotenv(dotenv_path=".env.example")
    question = "heyy mannna"
    history = []
    logger = Logger().get_logger()
    config = Config()
    chatbot_instance = RAGChatbot(config, logger)     #this function will log parameters and traces in mlflow for your model experimentation.
    prediction = chatbot_instance.advance_rag_chatbot(question, history)
    return prediction


if __name__ == "__main__":
    res = chatbot()
    print(res)
