import os
from base64 import b64encode
from typing import List

import streamlit as st
from dotenv import load_dotenv
from hybrid_rag.src.config import Config
from hybrid_rag.src.rag import RAGChatbot
from hybrid_rag.src.utils import Logger
from hybrid_rag.src.utils.utils import save_history_to_github

load_dotenv()
logger = Logger().get_logger()
config = Config()

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = []  # ["Hello! Ask me anything about 🤗"]

    if "past" not in st.session_state:
        st.session_state["past"] = []  # ["Hi! How can I assist you👋"]

    if "df" not in st.session_state:
        st.session_state["df"] = []

    if "username" not in st.session_state:
        st.session_state["name"] = "abc"  # Default admin username

    if "persona" not in st.session_state:
        st.session_state["persona"] = "Chatbot Assistant"  # Default password


def login_section():
    # Display login form
    username = st.text_input(
        "username", key="username", placeholder="Enter your name here"
    )
    persona = st.text_input(
        "Perosna",
        key="persona",
        placeholder="Enter your persona for your chatbot personalization",
    )
    if st.button("Save"):
        if username and persona:  # Example condition
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["persona"] = persona
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")
    else:
        pass


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
        unsafe_allow_html=True,
    )


def render_menu():
    # Custom CSS for menu button and options
    custom_css = """
    <style>
    /* Styling for Streamlit buttons to look like menu options */
    div.stButton > button {
        background-color: #f0f0f0;
        color: black;
        border-radius: 5px;
        width: 20%;
        margin: -5px 0;
        border: 1px solid black;
    }
    div.stButton > button:hover {
        background-color: yellow;
        color: black;
    }

    /* Styling for the main menu toggle button */
    .main-menu-button {
        background-color: #f0f0f0;
        color: black;
        border-radius: 5px;
        width: 10%;
        padding: 8px 0;
        font-size: 16px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    .main-menu-button:hover {
        background-color: yellow;
        color: black;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Main menu button
    if "show_menu" not in st.session_state:
        st.session_state.show_menu = False
    if "clicked_option" not in st.session_state:
        st.session_state.clicked_option = ""

    # Display main menu button
    main_menu = st.button("☰", key="main_menu")
    if main_menu:
        st.session_state.show_menu = not st.session_state.show_menu

    # Display submenu when the menu is clicked
    if st.session_state.show_menu:
        with st.container():
            st.markdown('<div class="expandable-menu">', unsafe_allow_html=True)
            if st.button("User Guide"):
                st.session_state.clicked_option = "User Guide"
            if st.button("Monitoring"):
                st.session_state.clicked_option = "Monitoring"
            if st.button("Chatbot"):
                st.session_state.clicked_option = "Chatbot"
            st.markdown("</div>", unsafe_allow_html=True)

    # Display content based on the selected option
    if st.session_state.clicked_option == "User Guide":
        st.title("User Guide")
        st.write("Welcome to the User Guide...")
    elif st.session_state.clicked_option == "Monitoring":
        st.title("Monitoring")
        st.write("Monitoring content here...")
    elif st.session_state.clicked_option == "Chatbot":
        st.title("Chatbot")
        st.write("Chatbot content here...")


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
    # response = requests.post("https://comparable-clarie-adsds-226b08fd.koyeb.app/predict", json=data)

    # Optional
    chatbot_instance = RAGChatbot(config, logger)
    response = chatbot_instance.advance_rag_chatbot(query["query"], history)
    print(
        f"APP RESPONSE: {response}"
    )  # response:str,total_time:float,combined_results:list,evaluated_results:dict,token_usage:dict
    if isinstance(response[0], str):
        metrices = {}
        pass
    else:
        print("Not harmful content")
        metrices = response[3]

        if (
            os.getenv("GITHUB_TOKEN")
            and os.getenv("GITHUB_REPO_NAME")
            and os.getenv("CHATFILE_PATH")
        ):
            save_history_to_github(
                query["query"],
                response,
                os.getenv("GITHUB_TOKEN"),
                os.getenv("GITHUB_REPO_NAME"),
                os.getenv("CHATFILE_PATH"),
            )
        else:
            pass
    return response


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
    # st.session_state["logged_in"] = False
    if not st.session_state["logged_in"]:
        login_section()  # Ask the user to login
        # if st.session_state["logged_in"]:  # If logged in, rerun to hide the login screen
        # st.experimental_rerun()
    else:
        if st.session_state["logged_in"]:
            # st.markdown(custom_css, unsafe_allow_html=True)
            if "show_menu" not in st.session_state:
                st.session_state.show_menu = False  # Initially, the menu is hidden
            if "clicked_option" not in st.session_state:
                st.session_state.clicked_option = ""  # No option clicked initially

            st.markdown(
                """
                <style>
                .logo-img {
                    max-width: 150px; /* Adjust the maximum width as needed */
                    height: auto;
                    display: block; /* Ensures the image is centered and not inline */
                    margin: auto; /* Centers the image horizontally */
                }
                </style>

                <div class="top-bar">
                    <button class="menu-button">☰</button>
                    <span class="title-text">AI Consultant</span>
                    <a href="https://github.com/kolhesamiksha" class="github-link" target="_blank">
                        <svg class="github-icon" xmlns="http://www.w3.org/2000/svg" width="45" height="45" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 22v-2.09a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.42 5.42 0 0 0 20 4.77 5.07 5.07 0 0 0 20.91 1S19.73.65 16 3a13.38 13.38 0 0 0-8 0C5.27.65 4.09 1 4.09 1A5.07 5.07 0 0 0 5 4.77 5.42 5.42 0 0 0 3.5 10.3c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 19.91V22"></path></svg>
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
            <style>
                .top-bar {
                    background-color: #FFD700; /* Example background color */
                    padding: 15px; /* Example padding */
                    text-align: center; /* Center the text */
                    position: relative;
                }
                .menu-button {
                    position: absolute; /* Positioning relative to the parent container */
                    top: 10px; /* Distance from the top */
                    left: 10px; /* Distance from the left */
                    background-color: white; /* Button background color */
                    color: black; /* Button text color */
                    border: 2px solid black; /* Border style */
                    border-radius: 5px; /* Rounded corners */
                    padding: 10px 20px; /* Button padding */
                    font-size: 16px; /* Font size */
                    cursor: pointer; /* Change cursor on hover */
                    font-weight: bold;
                }
                .menu-button:hover {
                    background-color: #FFD700; /* Change to yellow on hover */
                    color: black; /* Keep text color black */
                }
                div.stButton > button:hover {
                    background-color: yellow; /* Yellow background on hover */
                    color: black; /* Ensure black text for contrast */
                }
                .main-menu-button {
                    position: fixed; /* Fixed to stay on top even when scrolled */
                    top: 10px; /* Space from the top */
                    left: 10px; /* Space from the left */
                    background-color: #f0f0f0; /* Match menu button background */
                    color: black;
                    border-radius: 5px;
                    padding: 8px 20px; /* Larger padding for better click area */
                    font-size: 16px;
                    border: 1px solid #ddd;
                    z-index: 9999; /* Ensure it stays on top */
                    cursor: pointer; /* Show pointer cursor */
                }
                .main-menu-button:hover {
                    background-color: yellow; /* Yellow background on hover */
                    color: black; /* Ensure black text for contrast */
                }
                /* Styling for the menu containing the buttons */
                .menu-buttons-container {
                    position: fixed;
                    top: 50px; /* Below the main menu button */
                    left: 10px; /* Align with the main menu button */
                    background-color: #f0f0f0;
                    border-radius: 5px;
                    padding: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    display: none; /* Hide initially */
                    z-index: 9998;
                }
                /* Styling for each menu button */
                .menu-buttons-container button {
                    display: block;
                    background-color: #444;
                    color: white;
                    border: none;
                    padding: 10px;
                    margin: 5px 0;
                    width: 100%;
                    font-size: 16px;
                    cursor: pointer;
                }
                /* Hover effect for menu buttons */
                .menu-buttons-container button:hover {
                    background-color: yellow;
                    color: black;
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
        """,
                unsafe_allow_html=True,
            )

            # if st.button("☰", key="menu-button"):
            #     st.session_state.show_menu = not st.session_state.show_menu

            # Display the expandable menu if `show_menu` is True
            # if not st.session_state.show_menu:
            #     with st.container():
            #         st.markdown('<div class="expandable-menu">', unsafe_allow_html=True)

            #         # Add buttons inside the expandable menu
            #         if st.button("User Guide"):
            #             st.session_state.clicked_option = "User Guide"
            #         if st.button("Chatbot"):
            #             st.session_state.clicked_option = "Chatbot"
            #         if st.button("Monitoring"):
            #             st.session_state.clicked_option = "Monitoring"

            #         st.markdown('</div>', unsafe_allow_html=True)

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
                svg_base64 = (
                    "data:image/svg+xml;base64,"
                    + b64encode(svg_image.encode()).decode()
                )

                # Check if the hidden button is clicked
                if st.button("Send", on_click=None):
                    data = {"query": user_input}
                    try:
                        output = conversation_chat(data, st.session_state["history"])
                        if isinstance(output[0], str):
                            print("Inside isinstance str")
                            st.session_state["history"].append([user_input, output[0]])
                            st.session_state["past"].append(user_input)
                            st.session_state["generated"].append(output[0])
                        else:
                            st.session_state["history"].append(
                                [user_input, output[0][0]]
                            )
                            # st.session_state["df"].append({"Question":user_input, "Answer":output[0][0], "Latency":output[1], "Total_Cost($)":output[0][1]})  #we can store this data to mongo or s3 for qa fine-tuning.
                            st.session_state["past"].append(user_input)
                            st.session_state["generated"].append(output[0][0])
                    except Exception:
                        st.session_state["generated"].append(
                            "There is some issue with API Key, Usage Limit exceeds for the Day!!"
                        )

            if st.session_state["generated"]:
                print(st.session_state["generated"])
                with container:
                    for i in range(len(st.session_state["generated"])):
                        with st.container():
                            message_func(st.session_state["past"][i], is_user=True)
                            if "output" in locals():
                                if isinstance(output[0], str):
                                    message_func(
                                        f"<strong>Latency:</strong> {output[1]}s<br>"
                                        f'{st.session_state["generated"][i]}',
                                        is_user=False,
                                    )
                                else:
                                    message_func(
                                        f"<strong>Latency:</strong> {output[1]}s<br>"
                                        f'<strong>Faithfullness:</strong> {metrices["faithfulness"]}<br>'
                                        f'<strong>Context_Utilization:</strong> {metrices["context_utilization"]}<br>'
                                        f'<strong>Harmfulness:</strong> {metrices["harmfulness"]}<br>'
                                        f'<strong>Correctness:</strong> {metrices["correctness"]}<br>'
                                        f'<strong>Completion Tokens:</strong> {output[0][1]["token_usage"]["completion_tokens"]}<br>'
                                        f'<strong>Prompt Tokens:</strong> {output[0][1]["token_usage"]["prompt_tokens"]}<br>'
                                        f'{st.session_state["generated"][i]}',
                                        is_user=False,
                                    )
                                # message_func(f"**Latency**:{output[1]}s\t\t\t**Total_Cost**: ${output[0][1]}\n{st.session_state['generated'][i]}")


if __name__ == "__main__":
    main()
