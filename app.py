import streamlit as st
from utils import text
from embeddings import open_ai_embedding
from utils import chatbot
from dotenv import load_dotenv
import logging
from streamlit_chat import message

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

def main():
    st.set_page_config(page_title='ChatPDF', page_icon=':books:')

    st.header('Chat with your PDF files')

    user_question = st.text_input('Ask a question about your PDF files:')

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None

    if user_question:
        logger.info("User question received: %s", user_question)
        
        response = st.session_state.conversation_chain(user_question)['chat_history'][-1]
        message(user_question, is_user=True)
        message(response.content, is_user=False)


    with st.sidebar:
        st.subheader('Files')
        pdf_docs = st.file_uploader('Upload your PDF files', accept_multiple_files=True)

        if st.button('Process'):
            logger.info("Starting to process PDFs...")
            all_files_text = text.process_files(pdf_docs)

            logger.info("Extracted texts. Creating chunks...")
            chunks = text.create_text_chunks(all_files_text)

            logger.info("Chunks created. Creating embedding vector...")
            vector_store = open_ai_embedding.create_vector_store(chunks)
            logger.info("Vector created successfully: %s", vector_store)

            logger.info("Creating conversation chain...") 
            st.session_state.conversation_chain = chatbot.create_conversation_chain(vector_store)
            logger.info("Conversation chain created successfully.") 

if __name__ == '__main__':
    main()