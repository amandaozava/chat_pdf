import os
import logging
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Configuração de log
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_vector_store(chunks_with_metadata, persist_directory="./chroma_db"):
    """
    Cria ou atualiza um banco vetorial com base em chunks contendo metadados com (source, hash).
    Apenas chunks novos (hashes ainda não indexados) serão adicionados.
    """

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    embeddings = OpenAIEmbeddings()

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        existing_docs = vector_store.get(include=["metadatas"])
        existing_keys = {
            (meta.get("source"), meta.get("hash"))
            for meta in existing_docs["metadatas"]
            if "source" in meta and "hash" in meta
        }

        logger.info(f"Chunks já indexados: {len(existing_keys)}")

        logger.info(f"Chunks recebidos para indexar: {len(chunks_with_metadata)}")

        new_chunks = [
            doc
            for doc in chunks_with_metadata
            if (doc.metadata.get("source"), doc.metadata.get("hash")) not in existing_keys
        ]

        if not new_chunks:
            logger.info("Nenhum novo chunk para indexar.")
            return vector_store

        logger.info(f"Indexando {len(new_chunks)} novos chunks.")
        vector_store.add_documents(new_chunks)
        vector_store.persist()
        return vector_store

    else:
        logger.info("Criando novo vector store com todos os chunks.")
        vector_store = Chroma.from_documents(
            documents=chunks_with_metadata,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        vector_store.persist()
        return vector_store
