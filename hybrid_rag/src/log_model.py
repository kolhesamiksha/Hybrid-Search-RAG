import os
import json
import shutil
import mlflow
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

class CustomMlflowLogging:
    def __init__(self, config:Config):
        # dotenv_path = './.env.example'
        # load_dotenv(dotenv_path=dotenv_path)
        # self.config = Config()
        # self.logger = Logger()
        self._setup_mlflow()
        os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:5000"
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        self.config = config
        self.mlflow_tracking_uri = self.config.MLFLOW_TRACKING_URI
        self.mlflow_experiment = self.config.MLFLOW_EXPERIMENT_NAME
        self.mlflow_run_name = self.config.MLFLOW_RUN_NAME
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.langchain.autolog(log_traces=True)
        
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
            print("Starting MLflow server...")
            subprocess.Popen(mlflow_command)
            print("MLflow server started on http://localhost:5000")
        except Exception as e:
            print(f"Failed to start MLflow server: {e}")

    def log_model(self, question, history):
        """
        This method logs the RAGChatbot model with associated input/output schema,
        environment file, and Python dependencies.
        """
        # question_lst = [question]
        # history_lst = [history]
        # model_input = pd.DataFrame({
        #     "question": question_lst,
        #     "history": history_lst
        # })
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
        model_input = {"question":question, "history": history}
        
        # Log input/output example
        # input_example = pd.DataFrame([{
        #     "question": question,
        #     "history": history
        # }])
        
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

            with mlflow.start_run(run_name=self.mlflow_run_name, nested=True):
            # Log the model to MLflow as a PyFunc model
                mlflow.pyfunc.log_model(
                    artifact_path="rag_chatbot_model_1",
                    python_model=rag_model,
                    signature=signature,
                    input_example=model_input,
                    artifacts={"env_file": f"{artifact_dir}/.env.example"},
                    pip_requirements=self.requirements
                )

            print("Model logged successfully.")

        except Exception as e:
            print(f"Error during model logging: {e}")
            raise