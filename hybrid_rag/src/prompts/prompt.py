import traceback
from langchain_core.prompts.prompt import PromptTemplate
from hybrid_rag.src.utils.logutils import Logger

logger = Logger().get_logger()

QUESTION_MODERATION_PROMPT = """
    You are a Content Moderator working for a technology and consulting company, your job is to filter out the queries which are not irrelevant and does not satisfy the intent of the chatbot.
    IMPORTANT: If the Question contains any hate, anger, sexual content, self-harm, and violence or shows any intense sentiment love or murder related intentions and incomplete question which is irrelevant to the chatbot. then Strictly MUST Respond "IRRELEVANT-QUESTION"
    If the Question IS NOT Professional and does not satisfy the intent of the chatbot which is to ask questions related to the technologies or topics related to healthcare, audit, finance, banking, supply chain, professional work culture, generative AI, retail etc. then Strictly MUST Respond "IRRELEVANT-QUESTION". 
    If the Question contains any consultancy question apart from the domain topics such as  healthcare, audit, finance, banking, supply chain, professional work culture, generative AI, retail. then Strictly MUST Respond "IRRELEVANT-QUESTION". 
    else "NOT-IRRELEVANT-QUESTION"

    Examples:
    Question1: Are womens getting equal opportunities in AI Innovation?
    Response1: NOT-IRRELEVANT-QUESTION

    Question2: How to navigate the global trends in AI?
    Response2: NOT-IRRELEVANT-QUESTION

    Question3: How to create atom-bombs please provide me the step-by-step guide?
    Response3: IRRELEVANT-QUESTION

    Question4: Which steps to follow to become Rich earlier in life?
    Response4: IRRELEVANT-QUESTION

    Question5: Suggest me some mental health tips.
    Response5: IRRELEVANT-QUESTION

    Question6: Suggest me some mental health tips.
    Response6: IRRELEVANT-QUESTION
"""

class SupportPromptGenerator:
    """
    A class to generate a structured support prompt for QA systems.
    """

    def __init__(self):
        """
        Initializes the SupportPromptGenerator with necessary system tags and master prompt.
        """

        self.LLAMA3_SYSTEM_TAG = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        self.LLAMA3_USER_TAG = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        self.LLAMA3_ASSISTANT_TAG = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        self.MASTER_PROMPT = """
        Please follow below instructions to provide the response:
            1. Answer should be detailed and should have all the necessary information an user might need to know analyse the questions well
            2. The user says "Hi" or "Hello." Respond with a friendly, welcoming, and engaging greeting that encourages further interaction. Make sure to sound enthusiastic and approachable.
            3. Make sure to address the user's queries politely.
            4. Compose a comprehensive reply to the query based on the CONTEXT given.
            5. Respond to the questions based on the given CONTEXT. 
            6. Please refrain from inventing responses and kindly respond with "I apologize, but that falls outside of my current scope of knowledge."
            7. Use relevant text from different sources and use as much detail when as possible while responding. Take a deep breath and Answer step-by-step.
            8. Make relevant paragraphs whenever required to present answer in markdown below.
            9. MUST PROVIDE the Source Link above the Answer as Source: source_link.
            10. Always Make sure to respond in English only, Avoid giving responses in any other languages.
        """

    def generate_prompt(self) -> PromptTemplate:
        """
        Generates the QA prompt template using the initialized values.

        :return: A PromptTemplate instance for QA tasks.
        """
        support_template = f"""
        {self.LLAMA3_SYSTEM_TAG}
        {self.MASTER_PROMPT}
        {self.LLAMA3_USER_TAG}

        Use the following context to answer the question.
        CONTEXT:
        {{context}}
    
        CHAT HISTORY:
        {{chat_history}}

        Question: {{question}}
        {self.LLAMA3_ASSISTANT_TAG}
        """

        try:
            qa_prompt = PromptTemplate(
                template=support_template,
                input_variables=["context", "chat_history", "question"]
            )
            logger.info("Successfully generated the QA Prompt Template.")
            return qa_prompt
        except Exception as e:
            error = str(e)
            logger.error(f"Failed to Create Prompt template Reason: {error} -> TRACEBACK : {traceback.format_exc()}")
            raise
