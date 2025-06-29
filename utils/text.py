import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredFileLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import hashlib


def load_file(uploaded_file):
    """
    Salva temporariamente o UploadedFile e carrega o conteúdo com loader adequado.
    """
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_file_path = tmp_file.name

    if ext == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif ext == ".txt":
        loader = TextLoader(temp_file_path, encoding="utf-8")
    elif ext == ".csv":
        loader = CSVLoader(temp_file_path, encoding="latin-1")
    elif ext == ".docx":
        loader = Docx2txtLoader(temp_file_path)
    else:
        loader = UnstructuredFileLoader(temp_file_path)

    docs = loader.load()

    os.remove(temp_file_path)
    return docs


def process_files(files):
    texts = {}
    for file in files:
        print("Processando arquivo: ", file.name)
        docs = load_file(file)
        content = ""
        for doc in docs:
            content += doc.page_content + "\n"
        texts[file.name] = content
    return texts


def create_text_chunks(all_files_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1500, chunk_overlap=300, length_function=len
    )

    all_chunks = []
    for file_name, text in all_files_text.items():
        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            hash_chunk = hashlib.sha256(chunk.encode("utf-8")).hexdigest()

            all_chunks.append(
                Document(page_content=chunk, metadata={"source": file_name, "hash": hash_chunk})
            )
    return all_chunks
