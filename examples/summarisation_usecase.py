from dotenv import load_dotenv
from hybrid_rag import Config
from hybrid_rag.src.utils import Logger
from hybrid_rag import Summarization


def summarization_chatbot():
    load_dotenv('.env.example')
    question = "summarise the insurance document how-can-you-prepare-for-tomorrows-climate-today"
    logger = Logger().get_logger()
    config = Config()
    summary = Summarization(config, logger)
    result = summary.summarize_chatbot(question)
    print(result)


if __name__ == "__main__":
    res = summarization_chatbot()
    print(res)