from image_uploader_chatbot_nonmodel4 import get_vector_store,get_text_chunks
input = f"The user wants to know some information"
get_vector_store(get_text_chunks(input))
