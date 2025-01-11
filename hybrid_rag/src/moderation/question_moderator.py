from typing import Optional

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from hybrid_rag.src.models.llm_model.model import LLMModelInitializer
from hybrid_rag.src.utils.logutils import Logger


class QuestionModerator:
    """
    A class to detect moderated content using a language model and a moderation prompt.
    """

    def __init__(
        self, llmModelInstance: LLMModelInitializer, logger: Optional[Logger] = None
    ):
        """
        Initialize the ModerationContentDetector with an LLM model.

        :param llmModelInstance: LLMModel Instance must be of type LLMModelInitializer class
        """
        # self.llm_model_name = llm_model_name
        # self.__groq_api_key = groq_api_key
        self.llmModelInstance = llmModelInstance
        # llmModelInitializer = LLMModelInitializer(self.llm_model_name, self.__groq_api_key)

        llmModelInitializer = self.llmModelInstance
        self.llm_model_instance = llmModelInitializer.initialise_llm_model()

        self.prompt_template = """
        {QUESTION_MODERATION_PROMPT}

        Question: {question}
        """

    def detect(self, question: str, question_moderation_prompt: str) -> str:
        """
        Detects moderated content based on the given question and moderation prompt.

        :param question: The input question to be checked.
        :param question_moderation_prompt: The moderation prompt for the LLM.
        :return: The response from the LLM after processing.
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
            response = chain.invoke(
                {
                    "question": question,
                    "QUESTION_MODERATION_PROMPT": question_moderation_prompt,
                }
            )

            return response
        except Exception as e:
            raise RuntimeError(f"Failed to detect moderated content: {str(e)}")
