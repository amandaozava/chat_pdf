from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def process_files(files):
    texts = {}
    for file in files:
        pdf = PdfReader(file)
        content = ""
        for page in pdf.pages:
            content += page.extract_text()
        texts[file.name] = content
    return texts



def create_text_chunks(all_files_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1500,
        chunk_overlap=300, #um pedaço do chunk anterior (contexto)
        length_function=len
    )

    all_chunks = []

    for file_name, text in all_files_text.items():
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(
                Document(page_content=chunk, metadata={"source": file_name})
            )

    return all_chunks