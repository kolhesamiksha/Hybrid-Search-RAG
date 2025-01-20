import asyncio
from typing import Optional
import logging

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from hybrid_rag.src.utils.logutils import Logger


class QuestionModerator:
    """
    A class to detect moderated content using a language model and a moderation prompt.
    """

    def __init__(
        self, llmModelInstance: LLMModelInitializer, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ModerationContentDetector with an LLM model.

        Args:
            llmModelInstance (Type[LLMModelInitializer]): LLMModel Instance must be of type LLMModelInitializer class
        """
        self.llmModelInstance = llmModelInstance
        self.logger = logger if logger else Logger().get_logger()
        llmModelInitializer = self.llmModelInstance
        self.llm_model_instance = llmModelInitializer.initialise_llm_model()

        self.prompt_template = """
        {QUESTION_MODERATION_PROMPT}

        Question: {question}
        """

    async def detect_async(self, question: str, question_moderation_prompt: str) -> str:
        """
            Detects moderated content based on the given question and moderation prompt.

            Args:
                question (str): The input question to be checked.
                question_moderation_prompt (str): The moderation prompt for the LLM.

            Returns:
                str: The response from the LLM after processing.
        """
        try:
            # Create the prompt
            prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["QUESTION_MODERATION_PROMPT", "question"],
            )

            # Create the processing chain
            chain = (
                {
                    "question": RunnablePassthrough(),
                    "QUESTION_MODERATION_PROMPT": RunnablePassthrough(),
                }
                | prompt
                | self.llm_model_instance
            )

            # Invoke the chain and return the response
            response = await chain.ainvoke(
                {
                    "question": question,
                    "QUESTION_MODERATION_PROMPT": question_moderation_prompt,
                }
            )
            self.logger.info("SUccessfully check the Moderation of Question")
            return response
        except Exception as e:
            self.logger.error(f"Error during moderation detection: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to detect moderated content: {str(e)}")

    def detect(self, question: str, question_moderation_prompt: str) -> str:
        """
            Synchronous wrapper for the async detect_async method.

            Args:
                question (str): The input question to be checked.
                question_moderation_prompt (str): The moderation prompt for the LLM.

            Returns:
                str: The response from the LLM after processing.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already running an event loop, use run_coroutine_threadsafe
                self.logger.info("Running in an existing event loop.")
                future = asyncio.run_coroutine_threadsafe(
                    self.detect_async(question, question_moderation_prompt), loop
                )
                return future.result()
            else:
                # If no event loop is running, create a new one
                self.logger.info("Starting a new event loop.")
                return asyncio.run(self.detect_async(question, question_moderation_prompt))
        except Exception as e:
            self.logger.error(f"Error in async_rag: {e}")
            raise 