import os
import mlflow

from dotenv import load_dotenv
import pandas as pd
from mlflow.models import validate_serving_input
from hybrid_rag import Config
from hybrid_rag import CustomMlflowLogging
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

def log_hybrid_rag_model():
    question = "tell me about sypply chain consulting"
    history = []
    logger = Logger().get_logger()
    config = Config()

    model_logging = CustomMlflowLogging(config, logger)
    model_logging.log_model(question,history)
    return model_logging
    # Now check your logged model inside http://localhost:5000. 

#optional
def validate_input_serving(run_id:str, question:str, history:list):
    model_uri = run_id

    # The model is logged with an input example. MLflow converts
    # it into the serving payload format for the deployed model endpoint,
    # and saves it to 'serving_input_payload.json'

    serving_payload = pd.DataFrame({
        "question": [f"{question}"],
        "history": [str(f"{history}")]  # Convert history to string representation
    })

    validate_serving_input(model_uri, serving_payload)

def load_and_predict(run_id: str, question:str, history:list):
    logged_model = run_id #'runs:/7960cb7cc77b47858a657ba73b6a52a2/rag_chatbot_model_1'

    # Load model as a PyFuncModel.
    mlflow.pyfunc.get_model_dependencies(logged_model)
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    data = {
        "question": [question],
        "history": [str(history)]
    }
    print(pd.DataFrame(data))
    pred = loaded_model.predict(pd.DataFrame(data))
    return pred

if __name__=="__main__":    
    #setup_mlflow()
    load_dotenv(dotenv_path=".env.example")
    model = log_hybrid_rag_model()
    model_uri = f"runs:/{model.run_id}/rag_chatbot_artifacts"
    question = "tell me about supply chain consulting"
    history = []
    #validate_input_serving(model_uri, question, history)
    prediction = load_and_predict(model_uri, question, history)
    print(prediction)

