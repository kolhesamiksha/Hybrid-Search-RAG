from hybrid_rag.src.models import HuggingFaceAudioModels
from hybrid_rag.src.utils import Logger
from hybrid_rag import Config
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env.example")
    file_path = "examples/audio_examples/question1.mp3" # harvard.wav
    logger = Logger()
    config = Config()
    instance_m = HuggingFaceAudioModels(config, logger)
    output = instance_m.hg_asr_models_by_token(file_path)
    print(output)