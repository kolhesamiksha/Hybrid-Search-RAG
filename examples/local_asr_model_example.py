from hybrid_rag.src.utils import Logger
from hybrid_rag import Config
from hybrid_rag.src.models import LocalASRModel
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv(dotenv_path=".env.example")
    logger = Logger()
    config = Config()
    file_path = "harvard.wav"
    instance = LocalASRModel(config, logger)
    output = instance.transcribe(file_path)
    print(output)