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
# from keras.models import load_model

## Load the model
# skin_predict = load_model("skin_disease.h5")

# with open("",'rb') as f:
#     skin_predict = pickle.load(f)

skin_predict = keras.models.load_model("new_skin_disease.keras")
load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def crop_img(img):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
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

# def extract_parameters_from_text(text):
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt_template = f"""Extract the parameters: age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall for heart disease prediction.
# If a parameter is not available, provide the following average parameters: 
# 54.366337, 0.683168, 0.966997, 131.623762, 246.264026, 0.148515, 0.528053, 149.646865, 0.326733, 1.039604, 1.399340, 0.729373, 2.313531.
# Parameters extracted from the following text: {text}"""
#     prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
#     # Call the model and get the response
#     response = model.generate({"role": "user", "content": prompt})

#     # Debugging output to inspect the response structure
#     print("Model response:", response)

#     # Handle the response properly
#     if isinstance(response, str):
#         return response  # Return the raw string if it's a simple response
#     elif isinstance(response, list) and len(response) > 0:
#         return response[0]["content"]  # Assuming the model returns a list of messages
#     else:
#         return "Failed to extract parameters, please check the input text."


# Your existing functions...
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):   
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index"
    try:
        # Load existing index if it exists
        vector_store = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    except FileNotFoundError:
        # Create a new index if it doesn't exist
        vector_store = FAISS.from_texts([], embedding=embeddings)

    # Add new data to the index
    if isinstance(text_chunks, str):
        text_chunks = [text_chunks]  # Convert a single string to a list
    vector_store.add_texts(text_chunks)

    # Save the updated index
    vector_store.save_local(index_path)


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

def get_conversational_chain():

    
    prompt_template = '''If the question relates to skin diseases, you should answer based on the available context and your knowledge of skin diseases. If the question is general or outside of skin diseases, provide an answer based on your general knowledge.

     you have access to a model that can predict the possibility of the following diseases:
    'Acitinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion'.
    
    Answer based on the context for skin diseases, but for general questions, provide general knowledge. 

    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    '''
    # """
    # You are a skin disease detector assistant. Please answer the user's question in the most relevant way, using both available context and your own knowledge if the context does not fully answer the question.
    # You have access to a model that can detect the predict the possibility of presence of the following doseases:
    # 'Acitinic Keratosis',
    # 'Basal Cell Carcinoma',
    # 'Dermatofibroma',
    # 'Melanoma',
    # 'Nevus',
    # 'Pigmented Benign Keratosis',
    # 'Seborrheic Keratosis',
    # 'Squamous Cell Carcinoma',
    # 'Vascular Lesion'
    # Context:\n {context}?\n
    # Question: \n{question}\n

    # Answer:
    # """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.6)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def skinDiseasePrediction(image):

    # Ensure the image has three color channels
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 1:  # Single channel image
        image = np.repeat(image, 3, axis=-1)

    # Resize the image to 224x224
    image = cv2.resize(image, (224, 224))
    # Add batch dimension
    image = np.expand_dims(image, axis=0)  # Shape becomes (1, 224, 224, 3)
    
    print("Preprocessed input shape:", image.shape)
    arr = skin_predict.predict(image)
    print(arr)
    prediction = max(arr[0])
    index = 0
    index = np.argmax(arr[0])
    print(index)
    print(prediction)
    arr_of_dis = classes = [
    'Acitinic Keratosis',
    'Basal Cell Carcinoma',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Pigmented Benign Keratosis',
    'Seborrheic Keratosis',
    'Squamous Cell Carcinoma',
    'Vascular Lesion'
    ]
    
    if (prediction>0.5):
        user_input("The user has uploaded an image just now and there is High chance of presence of "+arr_of_dis[index]+" !! Please respond in the context to this disease until another image is uploaded. Also Please give the disease that is classified in bold letters first and then tell the user about the disease, its symptoms.",True)
    else:
        user_input("""The user has uploaded an image just now and there is Low chance of presence of the diseases 'Acitinic Keratosis',
        'Basal Cell Carcinoma',
        'Dermatofibroma',
        'Melanoma',
        'Nevus',
        'Pigmented Benign Keratosis',
        'Seborrheic Keratosis',
        'Squamous Cell Carcinoma',
        'Vascular Lesion' .
        Consult a doctor if issue persists """,True)

# def user_input(user_question,flag):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     get_vector_store(get_text_chunks(f'''Question at {time.time()}:{user_input} 
#     Answer at {time.time}: {response["output_text"]}'''))
#     print("Flag is ",flag)
#     if flag:
#         st.write(response["output_text"])
#     return response

def user_input(user_question, flag):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # If relevant context is found, use it
    if docs:
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    else:
        # Fallback to direct Google Generative AI response
        print("No relevant context found. Falling back to general knowledge.")
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        response = model.generate({"role": "user", "content": user_question})
        response = {"output_text": response["content"]}

    # Log the interaction in the vector store
    get_vector_store(
        get_text_chunks(
            f"""Question at {time.time()}:{user_question} 
            Answer at {time.time()}: {response["output_text"]}"""
        )
    )

    # Display the response
    print("Flag is", flag)
    if flag:
        st.write(response["output_text"])

    return response


# Main function
def main():
    st.set_page_config("Health chatbot")
    st.header("Upload your images")
    user_question = st.text_input("Ask your doubts")
    get_vector_store(get_text_chunks("No images Uploaded yet"))
    if user_question:
        user_input(user_question,True)
    with st.sidebar:
        st.title("Menu")
        uploaded_files = st.file_uploader("Upload Your image", type=['jpg','jpeg','png'], accept_multiple_files=True)  # Accept multiple images
    
        if st.button("Submit"):
            with st.spinner("Processing..."):
                for uploaded_file in uploaded_files:
                    pil_image = Image.open(uploaded_file)
                    st.image(pil_image, caption="Uploaded Image", use_column_width=True)
                    # Convert the PIL image to a NumPy array
                    opencv_image = np.array(pil_image)
                    skinDiseasePrediction(opencv_image)
                    
if __name__ == "__main__":
    main()