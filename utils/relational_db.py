from langchain.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_DB = f"sqlite:///{os.path.join(BASE_DIR, 'dbs', 'animais_exemplo.sqlite')}"


def create_db_chain():
    """
    Cria uma cadeia de consulta para o banco de dados relacional.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    db = SQLDatabase.from_uri(CAMINHO_DB)

    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    return db_chain


def aswer_question(question):
    """
    Responde a uma pergunta usando o banco de dados relacional.
    """
    db_chain = create_db_chain()
    resposta = db_chain.run(question)
    return resposta
