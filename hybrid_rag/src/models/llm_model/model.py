"""
Module Name: hybrid_search.py
Author: Samiksha Kolhe
Version: 0.1.0
"""
import logging
import traceback
from typing import Optional

import mlflow
from functools import lru_cache
from langchain_openai import ChatOpenAI

from hybrid_rag.src.utils.logutils import Logger

logger = Logger().get_logger()


class LLMModelInitializer:
    def __init__(
        self,
        llm_model_name: str,
        provider_base_url: str,
        groq_api_key: str,
        temperature: Optional[float] = 0.3,
        top_p: Optional[float] = 0.1,
        frequency_penalty: Optional[float] = 1.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the LLMModelInitializer with the necessary parameters.

        :param llm_model_name: Name of the LLM model to initialize.
        :param groq_api_key: API key for accessing the ChatGroq service.
        :param temperature Optional: Sampling temperature for the model.
        :param top_p Optional: Top-p sampling value for the model.
        :param frequency_penalty Optional: Frequency penalty for the model.
        :param logger Optional: Logger
        """

        self.logger = logger if logger else Logger().get_logger()
        self.llm_model_name = llm_model_name
        self.__groq_api_key = groq_api_key  # Make API key private
        self.base_url = provider_base_url
        if temperature is not None:
            self.temperature = temperature
        else:
            self.temperature = 0.3
        if top_p is not None:
            self.top_p = top_p
        else:
            self.top_p = 0.1
        if frequency_penalty is not None:
            self.frequency_penalty = frequency_penalty
        else:
            self.frequency_penalty = 1.0
        self.llm_model = None

    @property
    def groq_api_key(self) -> str:
        return self.__groq_api_key

    @groq_api_key.setter
    def groq_api_key(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("groq_api_key must be a non-empty string.")
        self.__groq_api_key = value

    # Property for temperature
    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError("temperature must be a float in the range [0.0, 1.0].")
        self._temperature = value

    # Property for top_p
    @property
    def top_p(self) -> float:
        return self._top_p

    @top_p.setter
    def top_p(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError("top_p must be a float in the range [0.0, 1.0].")
        self._top_p = value

    # Property for frequency_penalty
    @property
    def frequency_penalty(self):
        return self._frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value: float) -> None:
        if not (-2.0 <= value <= 2.0):
            raise ValueError(
                "frequency_penalty must be a float in the range [-2.0, 2.0]."
            )
        self._frequency_penalty = value

    @lru_cache(maxsize=1)
    def initialise_llm_model(self) -> Optional[ChatOpenAI]:
        """
        Initializes and returns the LLM model.

        :return: An instance of the initialized LLM model.
        """
        try:
            self.llm_model = ChatOpenAI(
                model=self.llm_model_name,
                base_url=self.base_url,
                openai_api_key=self.__groq_api_key,  # Access private key
                temperature=self._temperature,
                top_p=self._top_p,
                frequency_penalty=self._frequency_penalty,
                max_retries=2,
            )
            self.logger.info("Successfully Initialized the LLM model")
        except Exception as e:
            error = str(e)
            self.logger.error(
                f"Failed to Initialize LLM model. Reason: {error} -> TRACEBACK: {traceback.format_exc()}"
            )
        return self.llm_model
