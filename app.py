import streamlit as st
from utils import text
from embeddings import open_ai_embedding
from utils import chatbot
from dotenv import load_dotenv
import logging
from streamlit_chat import message
import os
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

def main():
    st.set_page_config(page_title='ChatPDF', page_icon=':books:')

    st.header('Chat with your PDF files')

    user_question = st.text_input('Ask a question about your PDF files:')

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if user_question and st.session_state.conversation_chain:
        logger.info("User question received: %s", user_question)

        response = st.session_state.conversation_chain({"question": user_question})
        st.session_state.user_question = "" 

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

        for chat in st.session_state.chat_history[::-1]:
            message(chat["content"], is_user=(chat["role"] == "user"))

        with st.expander("Source used in the answer"):
            if response["source_documents"]:
                doc = response["source_documents"][0]
                source = doc.metadata.get("source", "Desconhecido")
                st.markdown(f"**Fonte:** {source}")
                st.markdown(f"> {doc.page_content[:300]}...")


    with st.sidebar:
        st.subheader('Files')
        pdf_docs = st.file_uploader('Upload your PDF files', accept_multiple_files=True)

        if st.button('Process'):
            if os.path.exists("./chroma_db"):
                logger.info("Loading existing vector store...")
                vector_store = open_ai_embedding.load_vector_store()
            else:
                logger.info("Extracting text from PDFs...")
                all_files_text = text.process_files(pdf_docs)

                logger.info("Creating chunks...")
                chunks = text.create_text_chunks(all_files_text)

                logger.info("Creating vector store and persisting embeddings...")
                vector_store = open_ai_embedding.create_vector_store(chunks)

            logger.info("Creating conversation chain...") 
            st.session_state.conversation_chain = chatbot.create_conversation_chain(vector_store)
            logger.info("Conversation chain created successfully.")

if __name__ == '__main__':
    main()

