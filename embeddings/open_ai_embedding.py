from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

def load_vector_store():
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vector_store

def create_vector_store(chunks_with_metadata):
    if not os.path.exists("./chroma_db"):
        os.makedirs("./chroma_db")

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks_with_metadata,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vector_store.persist()
    return vector_store
