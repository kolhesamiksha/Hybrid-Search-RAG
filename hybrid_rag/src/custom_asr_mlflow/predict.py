import os
import traceback
import torch
#from transformers import AutoModelForCTC, AutoProcessor
from transformers import pipeline
import mlflow
import numpy as np
import pandas as pd

class ASRLogging(mlflow.pyfunc.PythonModel):
    """
    Automatic Speech Recognition (ASR) Logging class using MLflow and Hugging Face Transformers.

    This class provides functionality for loading a pretrained ASR model, processing input audio, 
    and making predictions in a structured and reusable manner.
    
    Attributes:
        model (torch.nn.Module): The pretrained ASR model loaded from the specified snapshot.
        processor (transformers.AutoProcessor): The processor for preparing audio inputs.
    """

    def __init__(self):
        """
        Initializes the ASRLogging class with placeholders for the model and processor.
        """
        self.model = None
        self.processor = None

    def load_context(self, context):
        """
        Loads the pretrained ASR model and processor from the specified MLflow artifacts context.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow context containing artifact paths.

        Raises:
            ValueError: If the model ID or artifacts are not found.
        """
        #from huggingface_hub import snapshot_download
        try:
            model_name = os.getenv('MLFLOW_ASR_MODEL_NAME')
            self.DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
            #MODEL_ID = snapshot_download(repo_id="vasista22/whisper-hindi-small")
            self.transcribe = pipeline(task="automatic-speech-recognition", model=model_name, chunk_length_s=30, device=self.DEVICE_ID, token=os.getenv('HUGGING_FACE_TOKEN'))
            print(f"ASR pipeline initialized on {self.DEVICE_ID}.")
        except Exception as e:
            raise RuntimeError(f"Error loading model or processor: {e}")

    def _build_prompt(self, instruction):
        """
        Placeholder for building prompts, if needed for extended functionality.

        Args:
            instruction (str): Instruction or context for the ASR model.

        Returns:
            str: Processed prompt (currently not implemented).
        """
        # Extend this method to support custom prompts, if necessary.
        pass

    def predict(self, context, model_input, params=None):
        """
        Processes the input audio and generates predictions using the ASR model.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The MLflow context containing artifact paths.
            model_input (np.ndarray or list): Input audio data to be processed.
            params (dict, optional): Additional parameters for customization (not used).

        Returns:
            str: The predicted transcription for the input audio.

        Raises:
            ValueError: If the model or processor is not initialized.
        """
        if not hasattr(self, "transcribe") or self.transcribe is None:
            raise ValueError("Model or processor not loaded. Please call 'load_context' first.")

        try:
            audio = model_input['prompt'].iloc[0]
            print(f"Audio File Path: {audio}")
            
            # Process the input audio
            try:
                self.transcribe.model.config.forced_decoder_ids = self.transcribe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")
                # input_values = self.processor(model_input, return_tensors="pt", sampling_rate=16000).input_values
                
                # # Generate logits
                # with torch.no_grad():
                #     logits = self.model(input_values.to(self.DEVICE_ID)).logits.cpu()

                # # Decode the logits to transcription
                # predicted_ids = torch.argmax(logits, dim=-1)
                # transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                transcript = self.transcribe(audio)["text"]
                print(f"TRANSCRIPT: {transcript}")
                return transcript
            
            except Exception as e:
                raise RuntimeError(f"Error during transcription: {e}: TRACEBACK -> {traceback.format_exc()}")

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")

# Example usage:
# if __name__ == "__main__":
#     asr_logger = ASRLogging()
#     # Simulate MLflow context and input audio data
#     context = ...  # Define MLflow context with artifact path
#     model_input = ...  # Load or simulate audio input
#     asr_logger.load_context(context)
#     transcription = asr_logger.predict(context, model_input)
#     print(transcription)