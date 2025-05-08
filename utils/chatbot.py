from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

def create_conversation_chain(vector_store):

    llm = OpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vector_store.as_retriever(),
        memory=memory,
        return_source_documents=False,
        output_key="answer"
    )

    return conversation_chain