import streamlit as st
from streamlit_chat import message

import os
import tempfile
import requests
from base64 import b64encode
from pydantic import BaseModel
from src.utils.logutils import Logger

from langchain_core.messages import HumanMessage, AIMessage

# logger = Logger()

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
        avatar_url = "https://www.nvidia.com/content/nvidiaGDC/us/en_US/about-nvidia/legal-info/logo-brand-usage/_jcr_content/root/responsivegrid/nv_container_392921705/nv_container/nv_image.coreimg.100.850.png/1703060329053/nvidia-logo-vert.png"  # Provided logo link
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
        unsafe_allow_html=True)

def conversation_chat(query, history):
    # logger.info(f"Query {query}")
    # logger.info(f"History {history}")
    print("Inside API")
    data = {
        "query": query['query'],
        "history": history
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    print(response.json())
    # history.append((query, result[0][0])) #history.append(HumanMessage(content=query)), history.append(AIMessage(content=result))
    return response.json()

# def conversation_chat(query, history):
#     logger.info(f"Query {query}")
#     logger.info(f"History {history}")
#     result = advance_rag_chatbot(query['query'], history, logger)
#     # history.append((query, result[0][0])) #history.append(HumanMessage(content=query)), history.append(AIMessage(content=result))
#     return result

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
            background-color: #f8f8f8; /* Example background color */
            padding: 10px; /* Example padding */
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
            background-color: #e6ffe6;
            color: black;
            border: 2px solid black;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            position: fixed;
            bottom: 3rem;
        }
        .stTextInput>div>div>input {
            width: 85% !important;
            padding: 10px;
            background: #e6ffe6;
            border: 1px solid #ccc;
            border-radius: 5px;
            position: fixed;
            bottom: 3rem;
            height: 40px;
        }
        
        .main-container {
            background: #eeeeee;
            height: 100vh;
            overflow: hidden;
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
            output  = conversation_chat(
                data, st.session_state["history"]
            )
            st.session_state["history"].append([user_input, output[0][0]])
            st.session_state["df"].append({"Question":user_input, "Answer":output[0][0], "Latency":output[1], "Total_Cost($)":output[0][1]})  #we can store this data to mongo or s3 for qa fine-tuning.
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output[0][0])

    if st.session_state["generated"]:
        print(st.session_state["generated"])
        with container:
            for i in range(len(st.session_state["generated"])):
                with st.container():
                    message_func(st.session_state["past"][i], is_user=True)
                    if 'output' in locals():
                        message_func(f"**Latency**:{output[1]}s\t\t\t**Total_Cost**: ${output[0][1]}\n{st.session_state['generated'][i]}")

if __name__ == "__main__":
    main()