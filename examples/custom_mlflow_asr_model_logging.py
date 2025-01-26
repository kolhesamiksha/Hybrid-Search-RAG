import os
import mlflow

from dotenv import load_dotenv
import pandas as pd
from mlflow.models import validate_serving_input
from hybrid_rag import Config
from hybrid_rag.src.custom_asr_mlflow import CustomASRMlflowLogging
from hybrid_rag.src.utils import Logger

import subprocess

os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:5000"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

#optional
def setup_mlflow():
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

def load_and_predict(run_id: str, file_path: str):
    logged_model = run_id

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    if hasattr(loaded_model._model_impl.python_model, 'load_context'):
        context = None  # If necessary, set MLflow context here
        loaded_model._model_impl.python_model.load_context(context)

    data = {
        "prompt": [file_path] 
    }
    
    print(pd.DataFrame(data))
    pred = loaded_model.predict(pd.DataFrame(data))
    return pred

def log_asr_model(file_path: str):
    logger = Logger().get_logger()
    config = Config()
    print("Before Model Logging")
    model_logging = CustomASRMlflowLogging(config, logger)
    print("After Model Logging")
    model_logging.log_model(file_path)
    return model_logging
    # Now check your logged model inside http://localhost:5000. 

if __name__ == "__main__":
    #setup_mlflow()
    load_dotenv(dotenv_path=".env.example")
    file_path = "examples/audio_examples/question1.mp3" # harvard.wav
    model = log_asr_model(file_path)
    model_uri = f"runs:/{model.run_id}/multilingual_asr_model"
    print(f"MODEL URI: {model_uri}")
    prediction = load_and_predict(model_uri, file_path)
    print(prediction)
