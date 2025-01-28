"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""

import os
import torch
import transformers
import accelerate
import traceback
import mlflow
import numpy as np
import pandas as pd
import logging
import subprocess
from hybrid_rag.src.utils import Logger
from hybrid_rag.src.config import Config

from hybrid_rag.src.custom_asr_mlflow import ASRLogging

from typing import Optional

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

from huggingface_hub import snapshot_download

# Define input example
class CustomASRMlflowLogging:
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
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
        self.logger = logger if logger else Logger().get_logger()
        self.config = config
        self._setup_mlflow()
        os.environ['MLFLOW_TRACKING_URI'] = self.config.MLFLOW_TRACKING_URI
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
       
        self.mlflow_tracking_uri = self.config.MLFLOW_TRACKING_URI
        self.mlflow_experiment = self.config.MLFLOW_ASR_EXPERIMENT_NAME
        self.mlflow_run_name = self.config.MLFLOW_ASR_RUN_NAME
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.langchain.autolog(log_traces=True)
        self.logger.info("Started MLFLOW Tracing..")
        self.requirements = [
            'python==3.11.5'
            'cloudpickle==3.1.0',
            'mlflow==2.19.0',
            'torch==2.5.1',
            'transformers==4.48.1',
            'accelerate==1.3.0',
            'tensorflow==2.18.0',
            'tf-keras==2.18.0',
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

    def log_model(self, file_path: str):
        input_example = pd.DataFrame({"prompt": [file_path]})
        #snapshot_location = snapshot_download(repo_id=repo_id)  #ai4bharat/indicwav2vec-hindi
        input_schema = Schema(
            [
                ColSpec(DataType.string, "prompt"),
            ]
        )
        output_schema = Schema([ColSpec(DataType.string, "candidates")])

        parameters = ParamSchema(
            [
                ParamSpec("temperature", DataType.float, np.float32(0.1), None),
                ParamSpec("max_tokens", DataType.integer, np.int32(1000), None),
            ]
        )

        # output_schema
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=parameters)
        torch_version = torch.__version__.split("+")[0]
        with mlflow.start_run(run_name=self.mlflow_run_name, nested=True) as run:
            try:
                self.run_id = run.info.run_id
                self.logger.info(f"Started MLflow run with ID: {self.run_id}")
                self.logger.info(f"MLFLOW ASR MODEL LOGGING NAME: {self.config.MLFLOW_ASR_MODEL_NAME}")
                mlflow.pyfunc.log_model(
                    "multilingual_asr_model",
                    python_model=ASRLogging(),
                    # NOTE: the artifacts dictionary mapping is critical! This dict is used by the load_context() method in ASRLogging() class.
                    #artifacts={"model_name": self.config.MLFLOW_ASR_MODEL_NAME},
                    pip_requirements=[
                        f"torch=={torch_version}",
                        f"transformers=={transformers.__version__}",
                        f"accelerate=={accelerate.__version__}",
                        "einops",
                        "sentencepiece",
                    ],
                    input_example=input_example,
                    signature=signature,
                )
                self.logger.error(f"Sucessfully log the ASR Model: {str(e)}")
            except Exception as e:
                self.logger.error(f"ERROR- Unable to log the model due to {str(e)}, TRACEBACK: {traceback.format_exc()}")
        mlflow.end_run()
        return self
