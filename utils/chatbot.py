from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


def create_conversation_chain(vector_store):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Prompt customizado para restringir respostas ao conteúdo dos documentos
    prompt_template = """
        Você é um assistente que responde **apenas** com base nos documentos fornecidos.
        Não invente informações. Se a resposta não estiver nos documentos, diga claramente que a informação não está disponível.

        Contexto:
        {context}

        Pergunta:
        {question}
    """

    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    return conversation_chain
