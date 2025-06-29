import streamlit as st
from utils import text, relational_db, chatbot
from embeddings import open_ai_embedding
from dotenv import load_dotenv
import logging
from streamlit_chat import message


def setup():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    load_dotenv()
    return logging.getLogger(__name__)


logger = setup()


def init_session_state():
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def display_source(chat):
    with st.expander("Fonte da resposta"):
        if chat.get("source_type") == "Banco Relacional":
            st.markdown("**Fonte:** Banco Relacional")
        elif chat.get("source_docs"):
            doc = chat["source_docs"][0]
            source = doc.metadata.get("source", "Desconhecido")
            st.markdown(f"**Fonte:** {source}")
            st.markdown(f"> {doc.page_content[:300]}...")
        else:
            st.markdown("**Fonte:** Desconhecida")


def process_question(conversation_chain, question):
    response = conversation_chain({"question": question})
    answer = response["answer"]
    source_docs = response.get("source_documents", [])

    fallback_needed = "não está disponível" in answer.lower()

    if fallback_needed or not source_docs:
        fallback_answer = relational_db.aswer_question(question)
        return fallback_answer, "Banco Relacional", []
    else:
        return answer, "Vector Store", source_docs


def add_to_chat_history(user_question, answer, source_type, source_docs):
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "source_type": source_type,
            "source_docs": source_docs,
        }
    )


def display_chat_history():
    for chat in st.session_state.chat_history[::-1]:
        message(chat["content"], is_user=(chat["role"] == "user"))
        if chat["role"] == "assistant":
            display_source(chat)


def process_uploaded_files(docs):
    logger.info("Processando arquivos enviados...")
    all_files_text = text.process_files(docs)
    chunks = text.create_text_chunks(all_files_text)
    vector_store = open_ai_embedding.create_vector_store(chunks)
    logger.info("Arquivos processados e vetor criado com sucesso.")
    return vector_store


def main():
    st.set_page_config(page_title="ChatPDF", page_icon="📚")
    st.header("Converse com seus arquivos e banco relacional")

    init_session_state()

    user_question = st.text_input("Faça uma pergunta sobre seus arquivos PDF:")

    if user_question and st.session_state.conversation_chain:
        logger.info(f"Pergunta do usuário: {user_question}")
        answer, source_type, source_docs = process_question(
            st.session_state.conversation_chain, user_question
        )
        add_to_chat_history(user_question, answer, source_type, source_docs)

    display_chat_history()

    with st.sidebar:
        st.subheader("Arquivos")
        docs = st.file_uploader("Envie seus arquivos", accept_multiple_files=True)

        if st.button("Processar"):
            vector_store = process_uploaded_files(docs)
            st.session_state.conversation_chain = chatbot.create_conversation_chain(vector_store)
            logger.info("Cadeia de conversação criada com sucesso.")


if __name__ == "__main__":
    main()
