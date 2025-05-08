from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def process_files(files):
    text = ""
    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def create_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1500,
        chunk_overlap=300,   #um pedaço do chunk anterior (contexto)
        length_function=len
    )

    chuncks = text_splitter.split_text(text)

    return chuncks
