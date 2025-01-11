import json
import mlflow
import ast
import mlflow.pyfunc
import pandas as pd
from hybrid_rag.src.config import Config
from hybrid_rag.src.utils import Logger
from dotenv import load_dotenv

class RAGChatbotModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.logger = Logger().get_logger()
    
    def load_context(self, context):
        from hybrid_rag.src.rag import RAGChatbot
        dotenv_path = context.artifacts.get("env_file", None)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path)

        # Initialize the chatbot instance
        config = Config()
        logger = Logger().get_logger()
        self.chatbot_instance = RAGChatbot(config, logger)

    def predict(self, context, model_input):
        question = str(model_input['question'].iloc[0])
        history_str = str(model_input['history'].iloc[0])
        print(question)
        print(type(question))
        print(history_str)
        print(type(history_str))
        try:
            if history_str == "[]":
                history = []
            else:
                history = ast.literal_eval(history_str)
            print(history)
            print(type(history))    
        except Exception as e:
            self.logger.info("Exception in predict method")
        
        return self.chatbot_instance.advance_rag_chatbot(question, history)
