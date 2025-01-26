import torch
import traceback
from datasets import Dataset, Audio
from transformers import AutoModelForCTC, AutoProcessor
import torchaudio.functional as F
from hybrid_rag.src.utils import Logger
from hybrid_rag.src.config import Config

class LocalASRModel:
    """
    A class to handle local deployment and transcription using Indic ASR models.

    Attributes:
        model_name (str): The name of the Hugging Face model.
        file_path (str): The path to the audio file to be transcribed.
    """

    def __init__(self, config: Config, logger: Logger):
        """
        Initializes the IndicASRModel class.

        Args:
            model_name (str): The name of the Hugging Face ASR model.
            file_path (str): The path to the audio file to be transcribed.
        """
        self.config = config
        self.logger = logger if logger else Logger().get_logger()
        self.model_name = self.config.ASR_LOCAL_MODEL_NAME

    #indicwav2vec
    def transcribe(self, file_path: str) -> str:
        """
        Transcribes the input audio file using the specified ASR model.

        Returns:
            str: The transcribed text from the audio file.

        Raises:
            RuntimeError: If any error occurs during audio processing or model inference.
        """
        DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Load and resample audio file
            audio_dataset = Dataset.from_dict({"audio": [file_path]}).cast_column("audio", Audio())
            resampled_audio = F.resample(torch.tensor(audio_dataset[0]["audio"]["array"]), orig_freq=audio_dataset[0]["audio"]["sampling_rate"], new_freq=16000).numpy() #48000

            # Load model and processor
            model = AutoModelForCTC.from_pretrained(self.model_name).to(DEVICE_ID)
            processor = AutoProcessor.from_pretrained(self.model_name)

            # Prepare input values
            input_values = processor(resampled_audio, sampling_rate=16000, return_tensors="pt").input_values

            # Perform inference
            with torch.no_grad():
                logits = model(input_values.to(DEVICE_ID)).logits.cpu().numpy() 

            print(f"LOGITS SHAPE : {logits.shape}")
            # Decode predictions
            transcription = processor.batch_decode(logits)

            print(f"TRANSCRIPTION: {transcription[0]}")
            return transcription[0]

        except Exception as e:
            raise RuntimeError(f"Error during transcription: {traceback.format_exc(e)}")