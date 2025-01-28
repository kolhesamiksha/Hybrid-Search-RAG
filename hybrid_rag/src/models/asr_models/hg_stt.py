"""
Module Name: hybrid_search
Author: Samiksha Kolhe
Version: 0.1.0
"""
import traceback
import requests
import os
import mimetypes
from hybrid_rag.src.utils import Logger
from hybrid_rag.src.config import Config

class HuggingFaceAudioModels:
    """
    A class to interact with Hugging Face audio models for Automatic Speech Recognition (ASR).

    This class provides methods for sending audio files to Hugging Face's inference API either using a token-based model endpoint 
    or a specific service endpoint, and retrieving transcriptions.

    Attributes:
        model (str): The name of the Hugging Face model.
        file_path (str): The path to the audio file to be transcribed.
    """

    def __init__(self, config: Config, logger: Logger):
        """
        Initializes the HuggingFaceAudioModels class.

        Args:
            model_name (str): The name of the Hugging Face model.
            file_path (str): The path to the audio file to be transcribed.
        """
        self.config = config
        self.logger = logger if logger else Logger().get_logger()
        self.model = self.config.ASR_HG_MODEL_NAME
        self.token = self.config.HUGGING_FACE_TOKEN
        self.endpoint = self.config.HUGGING_FACE_ENDPOINT

    def _get_content_type(self, file_path: str) -> str:
        """
        Determines the MIME type of the audio file.

        Returns:
            str: The MIME type of the audio file.
        """
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            self.logger.info(f"Detected MIME type as {mime_type}")
            if not mime_type:
                raise ValueError(f"Unable to determine MIME type for file: {self.file_path}")
        except Exception as e:
            raise ValueError(f"Unable to determine MIME type for file: {self.file_path}")
        return mime_type

    def hg_asr_models_by_token(self, file_path: str) -> str:
        """
        Sends an audio file to the Hugging Face model inference API using a token.

        Args:
            token_name (str): The Hugging Face API token for authorization.

        Returns:
            str: The transcribed text from the audio file.

        Raises:
            Exception: If the API request fails or the response does not contain transcription text.
        """
        API_URL = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = headers = {"Authorization": f"Bearer {self.token}"}

        try:
            with open(file_path, "rb") as f:
                data = f.read()
            response = requests.post(API_URL, headers=headers, data=data)
            response.raise_for_status()
            transcription = response.json().get('text')

            try:
                response_json = response.json()
            except ValueError:
                raise RuntimeError("Failed to parse response as JSON")
            
            transcription = response_json.get('text')
            if transcription is None:
                raise ValueError("The response does not contain a 'text' field.")

            self.logger.info(f"Successfully Extracted the Transcript using Hugging Face Serverless API: {transcription}")
            return transcription

        except Exception as e:
            self.logger.error(f"ERROR During Transcript generation using HG Serveless API: TRACEBACK: {traceback.format_exc()}")
            raise RuntimeError(f"Error during API call: {e}")

    # NOTE: Only used this method if you deployed your models on hugging face
    def hg_asr_models_by_endpoint(self, file_path: str) -> str:
        """
        Sends an audio file to a specific service endpoint using a token.

        Args:
            service_endpoint (str): The service endpoint URL.
            token_name (str): The Hugging Face API token for authorization.

        Returns:
            str: The transcribed text from the audio file.

        Raises:
            Exception: If the API request fails or the response does not contain transcription text.
        """
        API_URL = self.endpoint
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": self._get_content_type(file_path)
        }

        try:
            with open(file_path, "rb") as f:
                data = f.read()

            response = requests.post(API_URL, headers=headers, data=data)
            response.raise_for_status()
            transcription = response.json().get('text')

            if transcription is None:
                raise ValueError("The response does not contain a 'text' field.")

            self.logger.info(f"Successfully Extracted the Transcript using Hugging Face Serverless API: {transcription}")
            return transcription

        except Exception as e:
            self.logger.error(f"ERROR During Transcript generation using HG Endpoint: TRACEBACK: {traceback.format_exc()}")
            raise RuntimeError(f"Error during API call: {e}")