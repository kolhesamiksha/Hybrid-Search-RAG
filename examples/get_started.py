from hybrid_rag import RAGChatbot
from dotenv import load_dotenv

from hybrid_rag import Config
from hybrid_rag.src.utils import Logger

def chatbot():
    load_dotenv()
    question = "heyy mannna"
    history = []
    logger = Logger().get_logger()
    config = Config()
    chatbot_instance = RAGChatbot(config, logger)
    prediction = chatbot_instance.advance_rag_chatbot(question, history)
    return prediction

if __name__=="__main__":
    res = chatbot()
    print(res)