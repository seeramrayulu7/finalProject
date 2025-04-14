import streamlit as st
import sqlite3
from datetime import datetime
from authentication import authenticate_user, register_user, username_exists, init_db, DB_FILE
from PIL import Image
import numpy as np
import os
from chatbotInterface import user_input, display_chat, skinDiseasePrediction

# Initialize the database
init_db()

# Function to retrieve user data
# Main app
def main():
    st.set_page_config("Health Chatbot", layout="wide")
    st.title("Health Chatbot")
    if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.show_signup = False
    with st.sidebar:
        # Login or Sign-Up page

        if not st.session_state.logged_in:
            st.sidebar.header("Login")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login",key="User_Login"):
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.sidebar.success(f"Welcome, {username}!")
                    return
                else:
                    st.sidebar.error("Invalid username or password")

            # Add a Sign-Up button
            if st.sidebar.button("Sign Up",key="User_Signup"):
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

        # Logged-in user dashboard
    if st.session_state.logged_in is True:
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.experimental_rerun()

        # Chatbot and Image Upload Interface
        st.header("Chat with the Bot")
        display_chat()

        # Input field for user questions
        user_question = st.text_input("Ask your doubts")
        if st.button("Submit", key="submit_question"):
            if user_question:
                user_input(user_question, True)
            else:
                st.warning("Please enter a question.")
            

        with st.sidebar:
            st.title("Menu")
            uploaded_file = st.file_uploader("Upload Your image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
        
            if st.button("Submit", key="submit_image"):
                with st.spinner("Processing..."):
                    pil_image = Image.open(uploaded_file)
                    st.image(pil_image, caption="Uploaded Image", use_container_width=True)
                    opencv_image = np.array(pil_image)
                    skinDiseasePrediction(opencv_image)

if __name__ == "__main__":
    main()