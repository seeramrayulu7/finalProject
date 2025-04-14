from image_uploader_chatbot_nonmodel4 import get_vector_store,get_text_chunks
from clearFaiss import clear_vector_store
input = f"The user wants to know some information"
get_vector_store(get_text_chunks(input))
clear_vector_store("Hi")