import streamlit as st
import ast
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import pickle
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import imutils
import keras
import time
from datetime import datetime

# Load the model
skin_predict = keras.models.load_model("new_skin_disease.keras")
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_question" not in st.session_state:
    st.session_state.processed_question = False

def crop_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    return new_img

# Function to read PDF and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_content = pdf.read()
        pdf_reader = PdfReader(BytesIO(pdf_content))
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):   
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index"
    try:
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        vector_store = FAISS.from_texts(['HI'], embedding=embeddings)

    if isinstance(text_chunks, str):
        text_chunks = [text_chunks]
    vector_store.add_texts(text_chunks)
    vector_store.save_local(index_path)

def get_conversational_chain():
    prompt_template = '''You are a medical AI trained to help with skin disease detection. You should answer the question based on the context and your knowledge of skin diseases. 

    If the question is about skin diseases, you have access to a model that can predict the possibility of the following diseases:
    'Acitinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion'.
    
    Provide an answer based on your knowledge and the context. For any disease prediction, indicate the likely disease based on the image provided or context from previous interactions. 
    
    Avoid disclaimers like "I'm an AI and can't take responsibility" unless specifically asked about limitations. If asked about a skin disease, give an answer based on the highest likelihood from the model predictions.
    
    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    '''
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def skinDiseasePrediction(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    arr = skin_predict.predict(image)
    prediction = max(arr[0])
    index = np.argmax(arr[0])
    arr_of_dis = [
        'Acitinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Melanoma',
        'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion'
    ]
    
    if prediction > 0.5:
        user_input(f"The user has uploaded an image just now and there is a High chance of presence of **{arr_of_dis[index]}**!! Consult a doctor immediately.", True)
    else:
        user_input("The user has uploaded an image just now and there is a Low chance of presence of any diseases. Consult a doctor if the issue persists.", True)

def user_input(user_question, flag):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # if docs:
    #     # Include the most relevant documents as context for answering
    #     context = " ".join([doc['text'] for doc in docs])
    #     chain = get_conversational_chain()
    #     response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)
    if docs:
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    else:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        response = model.generate({"role": "user", "content": user_question})
        response = {"output_text": response["content"]}

    get_vector_store(
        get_text_chunks(f"""Question at {time.time()}:{user_question} Answer at {time.time()}: {response["output_text"]}""")
    )

    # Add the user message and bot response to the chat history
    st.session_state.chat_history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "role": "user",
        "message": user_question
    })
    st.session_state.chat_history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "role": "bot",
        "message": response["output_text"]
    })

    if flag:
        st.write(response["output_text"])

    # Mark the question as processed
    st.session_state.processed_question = True

    return response

def display_chat():
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**User ({message['time']}):** {message['message']}")
        else:
            st.markdown(f"**Bot ({message['time']}):** {message['message']}")

# Main function
def main():
    st.set_page_config("Health chatbot")
    st.header("Health Chatbot")

    # Display chat history
    display_chat()

    # Input field for user questions
    user_question = st.text_input("Ask your doubts")

    # Process the question only if it hasn't been processed yet
    if user_question and not st.session_state.processed_question:
        user_input(user_question, True)

    # Reset the processed flag when the input field is cleared
    if not user_question:
        st.session_state.processed_question = False

    with st.sidebar:
        st.title("Menu")
        uploaded_files = st.file_uploader("Upload Your image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
        if st.button("Submit"):
            with st.spinner("Processing..."):
                for uploaded_file in uploaded_files:
                    pil_image = Image.open(uploaded_file)
                    st.image(pil_image, caption="Uploaded Image", use_container_width=True)
                    opencv_image = np.array(pil_image)
                    skinDiseasePrediction(opencv_image)

if __name__ == "__main__":
    main()
