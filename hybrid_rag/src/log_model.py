"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""

import os
import json
import shutil
import mlflow
import logging
import traceback
from typing import Optional
import pandas as pd
import mlflow.pyfunc
import subprocess
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
from hybrid_rag.src.config import Config
from hybrid_rag.src.utils import Logger
from dotenv import load_dotenv
#from hybrid_rag.src.rag import RAGChatbot
from hybrid_rag.src.custom_mlflow import RAGChatbotModel
from hybrid_rag.src.utils.logutils import Logger

class CustomMlflowLogging:
    def __init__(self, config:Config, logger: Optional[logging.Logger] = None):
        """
        RAGChatbotModel is a custom MLflow Python model to interact with a RAG (Retrieval-Augmented Generation) chatbot.

        Methods:
            load_context(context: MLflowContext) -> None:
                Loads environment variables and initializes the RAGChatbot instance.

            predict(context: MLflowContext, model_input: pd.DataFrame) -> str:
                Processes the input DataFrame, extracts question and history, and returns the chatbot's response.
        
        Args:
            context (MLflowContext): The MLflow model context. Used to load environment variables and artifacts.
            model_input (pd.DataFrame): The input DataFrame containing the columns 'question' and 'history'.
            
        Returns:
            str: The response from the RAG chatbot, typically a string generated from the chatbot's response logic.

        Raises:
            ValueError: If the input DataFrame is missing required columns ('question' or 'history') or is empty.
        """
        self.config = config
        self.logger = logger if logger else Logger().get_logger()
        self._setup_mlflow()
        os.environ['MLFLOW_TRACKING_URI'] = self.config.MLFLOW_TRACKING_URI
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
       
        self.mlflow_tracking_uri = self.config.MLFLOW_TRACKING_URI
        self.mlflow_experiment = self.config.MLFLOW_EXPERIMENT_NAME
        self.mlflow_run_name = self.config.MLFLOW_RUN_NAME
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.langchain.autolog(log_traces=True)
        self.logger.info("Started MLFLOW Tracing..")
        self.requirements = [
            'python==3.11.5'
            'cloudpickle==3.1.0',
            'mlflow==2.19.0',
            'cryptography==44.0.0',
            'dnspython==1.16.0',
            'jaraco-classes==3.4.0',
            'jaraco-collections==5.1.0',
            'lark==1.2.2',
            'numpy==1.26.4',
            'pandas==2.2.3',
            'platformdirs==4.3.6',
            'psutil==6.1.1',
            'pyarrow==18.1.0',
            'rich==13.9.4',
            'tornado==6.4.2',
            'langchain-community==0.2.13',
            'langchain==0.2.12',
            'fastembed==0.3.2',
            'langchain-core==0.2.43',
            'langchain-openai==0.1.25',
            'langchain-groq==0.1.10',
            'openai==1.58.1',
            'tiktoken==0.8.0',
            'pymilvus==2.5.1',
            'faiss-cpu==1.9.0.post1',
            'fastapi==0.115.6',
            'uvicorn==0.34.0',
            'pycryptodome==3.21.0',
            'ragas==0.2.9',
            'datasets==3.2.0',
            'flashrank==0.2.9',
            'PyGithub==2.5.0',
            'lark==1.2.2',
            'ipython==8.31.0',
            'pymongo==3.11',
            'python-dotenv==1.0.1',
        ]

    def _setup_mlflow(self):
        mlflow_command = [
            "mlflow",
            "server",
            "--backend-store-uri", "sqlite:///mlflow.db",  # Use SQLite database
            "--default-artifact-root", "./mlruns",        # Artifact storage path
            "--host", "0.0.0.0",                          # Bind to all network interfaces
            "--port", "5000"                              # Port to run the server on
        ]

        try:
            # Start the MLflow server
            self.logger.info("Starting MLflow server...")
            subprocess.Popen(mlflow_command)
            self.logger.info("MLflow server started on http://localhost:5000")
        except Exception as e:
            self.logger.error(f"Failed to start MLflow server: {str(e)} TRACEBACK: {traceback.format_exc()}")

    def log_model(self, question:str, history:list):
        """
        This method logs the RAGChatbot model with associated input/output schema,
        environment file, and Python dependencies.
        """

        question = str(question)
        history = str(history)
        input_schema = Schema([
            ColSpec("string", "question"),
            ColSpec("string", "history")
        ])
        output_schema = Schema([
            ColSpec("string")
        ])

        #hist_json_str = json.dumps(hist_json)
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        self.logger.info("Successfully Generated the model Signature")
        model_input = {"question":question, "history": history}
        
        dotenv_path = "./.env.example"
        artifact_dir = "model_artifacts"
        if os.path.isfile(artifact_dir):
            os.remove(artifact_dir)
        
        os.makedirs(artifact_dir, exist_ok=True)
        shutil.copy(dotenv_path, os.path.join(artifact_dir, ".env.example"))

        if mlflow.active_run() is not None:
            mlflow.end_run()
        
        # Log model details using MLflow
        try:
            mlflow.log_param("model_type", "RAGChatbot")
            # Initialize the model instance
            rag_model = RAGChatbotModel()
            self.logger.info("Called the RAGChatbotModel custom mlflow model class")
            with mlflow.start_run(run_name=self.mlflow_run_name, nested=True) as run:

                self.run_id = run.info.run_id
                self.logger.info(f"Started MLflow run with ID: {self.run_id}")
                self.logger.info("Successfully Started the mlflow tracing and logging")
                
                # Log the model to MLflow as a PyFunc model
                mlflow.pyfunc.log_model(
                    artifact_path="rag_chatbot_artifacts",
                    python_model=rag_model,
                    signature=signature,
                    input_example=model_input,
                    artifacts={"env_file": f"{artifact_dir}/.env.example"},
                    pip_requirements=self.requirements
                )
            self.logger.info("Model logged successfully.")

        except Exception as e:
            self.logger.error(f"Error during model logging: {e}")
            raise
        mlflow.end_run()
        return self