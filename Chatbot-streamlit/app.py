import streamlit as st
from streamlit_chat import message

import os
import io
import tempfile
import requests
from base64 import b64encode
from pydantic import BaseModel
from src.utils.logutils import Logger
from src.api import advance_rag_chatbot
from langchain_core.messages import HumanMessage, AIMessage
from github import Github
import pandas as pd
import base64
from typing import List

from src.utils.utils import save_history_to_github
from src.utils.utils import rag_evaluation

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = [] #["Hello! Ask me anything about 🤗"]

    if "past" not in st.session_state:
        st.session_state["past"] = [] #["Hi! How can I assist you👋"]
    
    if "df" not in st.session_state:
        st.session_state["df"] = []

def message_func(text, is_user=False):
    """
    This function is used to display the messages in the chatbot UI.

    Parameters:
    text (str): The text to be displayed.
    is_user (bool): Whether the message is from the user or the chatbot.
    """
    question_bg_color = "#f0f0f0"  # Faint grey color
    response_bg_color = "#f0f0f0"  # Faint grey color

    if is_user:
        avatar_url = "https://media.istockphoto.com/id/1184817738/vector/men-profile-icon-simple-design.jpg?s=612x612&w=0&k=20&c=d-mrLCbWvVYEbNNMN6jR_yhl_QBoqMd8j7obUsKjwIM="
        bg_color = question_bg_color
        alignment = "flex-end"
    else:
        avatar_url = "https://media.istockphoto.com/id/1184817738/vector/men-profile-icon-simple-design.jpg?s=612x612&w=0&k=20&c=d-mrLCbWvVYEbNNMN6jR_yhl_QBoqMd8j7obUsKjwIM="  # Provided logo link
        bg_color = response_bg_color
        alignment = "flex-start"
            # <div style="display: flex; align-items: center; margin-bottom: 20px;">
    st.write(
        f"""
        <div style="display: flex; align-items: flex-start; margin-bottom: 20px; justify-content: {alignment};">
            <div style="display: flex; align-items: center;">
                <img src="{avatar_url}" class="avatar" alt="avatar" style="width: 40px; height: 40px; border-radius: 50%; margin-right: 10px;" />
                <div style="background: {bg_color}; color: black; border-radius: 20px; padding: 10px; max-width: 75%; text-align: left;">
                    {text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def prepare_context(context) -> List[List[str]]:
    prepared_lst = []
    for doc in context:
        prepared_lst.append(doc.page_content)
    return [prepared_lst]

def conversation_chat(query, history):
    # logger.info(f"Query {query}")
    # logger.info(f"History {history}")
    print("Inside API")
    # data = {
    #     "query": query['query'],
    #     "history": history
    # }
    #response = requests.post("https://comparable-clarie-adsds-226b08fd.koyeb.app/predict", json=data)
    response = advance_rag_chatbot(query['query'], history)
    print(f"APP RESPONSE: {response}")
    if isinstance(response[0], str):
        metrices = {}
        pass
    else:
        print("Not harmful content")
        context = prepare_context(response[2])
        metrices = rag_evaluation([query['query']], [response[0][0]],context)
        save_history_to_github(query['query'], response)

    return response, metrices

def main():
    st.set_page_config(
        page_title="AI-Consultant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "About": """This is a Streamlit Chatbot Application designed to solve queries technology and business.
                """
        },
    )

    initialize_session_state()

    st.markdown("""
    <style>
        .top-bar {
            background-color: #FFD700; /* Example background color */
            padding: 30px; /* Example padding */
            text-align: center; /* Center the text */
        }

        .title-text {
            font-weight: bold; /* Make the text bold */
            font-size: 24px; /* Adjust the font size as needed */
            color: #333; /* Adjust the text color */
        }
        button {
            width: 80px;
            height: 40px;
            content: "Send url('{svg_base64}')";
            padding: 10px;
            background-color: #FFD700;
            color: black;
            border: 2px solid black;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            position: fixed;
            bottom: 3rem;
        }
        .stButton > button {
            width: 80px;
            height: 40px;
            background-color: #FFD700; /* Darker yellow color */
            color: black;
            border: 2px solid black;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .stTextInput>div>div>input {
            width: 85% !important;
            padding: 10px;
            background: #FFD700;
            border: 1px solid #ccc;
            border-radius: 5px;
            position: fixed;
            bottom: 3rem;
            height: 40px;
            font-weight: bold;
        }
        
        .main-container {
            background: #eeeeee;
            height: 100vh;
            overflow: hidden;
        }
        
        .github-link {
            position: absolute;
            right: 10px; /* Distance from the right edge of the top bar */
            top: 60%; /* Center vertically relative to the top bar */
            transform: translateY(-50%); /* Adjust vertical alignment */
            width: 50px; /* Adjust size as needed */
            height: 50px; /* Adjust size as needed */
        }

        .github-icon {
            border-radius: 50%; /* Makes the icon round */
            background: black; /* Background color for the circle */
            padding: 8px; /* Space between the circle and the icon */
        }

        .chat-container {
            background: #ffffff;
            height: 90vh;  /* Increased height */
            border-radius: 0.75rem;
            padding: 20px;  /* Increased padding */
            overflow-y: scroll;
            width: 100% !important;
            max-width: 1200px;  /* Added max-width for larger size */
            margin: 0 auto;  /* Centered the container */
            flex-direction: column-reverse;
        }

        .st-b7 {
            background-color: rgb(255 255 255 / 0%);
        }

        .st-b6 {
            border-bottom-color: rgb(255 255 255 / 0%);
        }

        .st-b5 {
            border-top-color: rgb(255 255 255 / 0%);
        }

        .st-b4 {
            border-right-color: rgb(255 255 255 / 0%);
        }

        .st-b3 {
            border-left-color: rgb(255 255 255 / 0%);
        }
        
    </style>
""", unsafe_allow_html=True)


    st.markdown("""
        <style>
        .logo-img {
            max-width: 150px; /* Adjust the maximum width as needed */
            height: auto;
            display: block; /* Ensures the image is centered and not inline */
            margin: auto; /* Centers the image horizontally */
        }
        </style>
        
        <div class="top-bar">
            <span class="title-text">AI Consultant</span>
            <a href="https://github.com/kolhesamiksha" class="github-link" target="_blank">
                <svg class="github-icon" xmlns="http://www.w3.org/2000/svg" width="45" height="45" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 22v-2.09a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.42 5.42 0 0 0 20 4.77 5.07 5.07 0 0 0 20.91 1S19.73.65 16 3a13.38 13.38 0 0 0-8 0C5.27.65 4.09 1 4.09 1A5.07 5.07 0 0 0 5 4.77 5.42 5.42 0 0 0 3.5 10.3c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 19.91V22"></path></svg>
            </a>
        </div>
        """, unsafe_allow_html=True
    )

    container = st.container()
    col1, col2 = st.columns([14, 1])
    with col1:
        user_input = st.text_input(
            "Question: ",
            placeholder="Enter the prompt here...",
            key="input",
            value=st.session_state.get("input", ""),
            label_visibility="hidden",
        )

    with col2:
        st.write("")
        st.write("")
        svg_image = """
        <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 1 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="tabler-icon tabler-icon-send"><path d="M10 14l11 -11"></path><path d="M21 3l-6.5 18a.55 .55 0 0 1 -1 0l-3.5 -7l-7 -3.5a.55 .55 0 0 1 0 -1l18 -6.5"></path></svg>
        """
        svg_base64 = "data:image/svg+xml;base64," + b64encode(svg_image.encode()).decode()

        # Check if the hidden button is clicked
        if st.button("Send", on_click=None):
            data = {
                'query': user_input
            }
            try:
                output, metrices  = conversation_chat(
                    data, st.session_state["history"]
                )
                if isinstance(output[0], str):
                    print("Inside isinstance str")
                    st.session_state["history"].append([user_input, output[0]])
                    st.session_state["past"].append(user_input)
                    st.session_state["generated"].append(output[0])
                else:
                    st.session_state["history"].append([user_input, output[0][0]])
                    # st.session_state["df"].append({"Question":user_input, "Answer":output[0][0], "Latency":output[1], "Total_Cost($)":output[0][1]})  #we can store this data to mongo or s3 for qa fine-tuning.
                    st.session_state["past"].append(user_input)
                    st.session_state["generated"].append(output[0][0])
            except Exception as e:
                st.session_state["generated"].append('There is some issue with API Key, Usage Limit exceeds for the Day!!')

    if st.session_state["generated"]:
        print(st.session_state["generated"])
        with container:
            for i in range(len(st.session_state["generated"])):
                with st.container():
                    message_func(st.session_state["past"][i], is_user=True)
                    if 'output' in locals() and 'metrices' in locals():
                        if isinstance(output[0], str):
                            message_func(
                                f'<strong>Latency:</strong> {output[1]}s<br>'
                                f'{st.session_state["generated"][i]}',
                                is_user=False
                            )
                        else:
                            message_func(
                                f'<strong>Latency:</strong> {output[1]}s<br>'
                                f'<strong>Faithfullness:</strong> {metrices["faithfulness"]}<br>'
                                f'<strong>Context_Utilization:</strong> {metrices["context_utilization"]}<br>'
                                f'<strong>Harmfulness:</strong> {metrices["harmfulness"]}<br>'
                                f'<strong>Correctness:</strong> {metrices["correctness"]}<br>'
                                f'<strong>Completion Tokens:</strong> {output[0][1]["token_usage"]["completion_tokens"]}<br>'
                                f'<strong>Prompt Tokens:</strong> {output[0][1]["token_usage"]["prompt_tokens"]}<br>'
                                f'{st.session_state["generated"][i]}',
                                is_user=False
                            )
                        #message_func(f"**Latency**:{output[1]}s\t\t\t**Total_Cost**: ${output[0][1]}\n{st.session_state['generated'][i]}")

if __name__ == "__main__":
    main()
