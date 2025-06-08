from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI


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

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,  
        output_key="answer",       
    )

    return conversation_chain