import streamlit as st
import sqlite3
from datetime import datetime, date
from authentication import authenticate_user, register_user, username_exists, init_db, DB_FILE, save_user_data, get_user_data
from PIL import Image
import numpy as np
import os
from chatbotInterface import user_input, display_chat, skinDiseasePrediction

# Initialize the database
init_db()

# Function to format chat messages
def format_chat_message(role, message):
    timestamp = datetime.now().strftime("%H:%M:%S")  # Get the current time in HH:MM:SS format
    return f"[{role} {timestamp}] {message}"

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# Main app
def main():
    st.set_page_config("Health Chatbot", layout="wide")
    st.title("Health Chatbot")

    with st.sidebar:
        # Login or Sign-Up page
        if not st.session_state.logged_in:
            st.sidebar.header("Login")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login", key="User_Login"):
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.sidebar.success(f"Welcome, {username}!")
                    return
                else:
                    st.sidebar.error("Invalid username or password")

            # Add a Sign-Up button
            if st.sidebar.button("Sign Up", key="User_Signup"):
                st.session_state.show_signup = True

            # Show the Sign-Up form if the button is clicked
            if "show_signup" in st.session_state and st.session_state.show_signup:
                st.sidebar.header("Sign Up")
                new_username = st.sidebar.text_input("New Username")
                new_password = st.sidebar.text_input("New Password", type="password")
                confirm_password = st.sidebar.text_input("Confirm Password", type="password")
                if st.sidebar.button("Register"):
                    if new_password != confirm_password:
                        st.sidebar.error("Passwords do not match. Please try again.")
                    elif username_exists(new_username):
                        st.sidebar.error("Username already exists. Please choose a different username.")
                    else:
                        if register_user(new_username, new_password, "User"):
                            st.sidebar.success("Account created successfully! Please log in.")
                            st.session_state.show_signup = False
                            return
                        else:
                            st.sidebar.error("An error occurred during registration. Please try again.")
        else:
            # Logged-in user options
            if st.sidebar.button("Logout"):
                if "chat_history" in st.session_state and "uploaded_images" in st.session_state:
                    save_user_data(
                        username=st.session_state.username,
                        chat_history=st.session_state.chat_history,
                        uploaded_images=st.session_state.uploaded_images
                    )
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.chat_history = []
                st.session_state.uploaded_images = []
                st.experimental_rerun()

            # Chat History Button
            if st.sidebar.button("Chat History"):
                st.session_state.show_dashboard = not st.session_state.show_dashboard

            # Date Selector for Chat History
            selected_date = st.sidebar.date_input("Select Date", value=date.today())

            # Image Upload
            st.sidebar.header("Upload an Image")
            uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                pil_image = Image.open(uploaded_file)
                st.image(pil_image, caption="Uploaded Image", use_container_width=True)
                opencv_image = np.array(pil_image)
                if st.sidebar.button("Submit Image", key="submit_image"):
                    with st.spinner("Processing..."):
                        skinDiseasePrediction(opencv_image)

    # Chatbot and Image Upload Interface
    if st.session_state.logged_in:
        st.header("Chat with the Bot")
        user_question = st.text_input("Ask your doubts")
        if st.button("Submit", key="submit_question"):
            if user_question:
                formatted_user_message = format_chat_message("User", user_question)
                st.session_state.chat_history.append(formatted_user_message)

                # Simulate chatbot response
                chatbot_response = "This is a simulated response."
                formatted_chatbot_message = format_chat_message("Chatbot", chatbot_response)
                st.session_state.chat_history.append(formatted_chatbot_message)

                # Display chat history
                for message in st.session_state.chat_history:
                    st.write(message)

        # Show Dashboard if Chat History Button is Clicked
        if st.session_state.show_dashboard:
            st.header("Your Dashboard")
            user_data = get_user_data(st.session_state.username, specific_date=selected_date.isoformat())
            if user_data:
                for entry in user_data:
                    st.subheader(f"Date: {entry['date']}")
                    st.write("Chat History:")
                    for chat in entry["chat_history"]:
                        st.write(chat)
                    st.write("Uploaded Images:")
                    for image_path in entry["uploaded_images"]:
                        st.image(image_path, caption=image_path)
            else:
                st.write("No data found for the selected date.")
    else:
        st.warning("Please log in to access the chatbot and dashboard.")

if __name__ == "__main__":
    main()