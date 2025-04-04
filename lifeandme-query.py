from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()


def build_chat_history(chat_history_list):



    chat_history = []
    for message in chat_history_list:

        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))


    return chat_history



def query(question, chat_history):

    """
    This function does the following:
    1. Recieves two parameters - 'question' - a string and 'chat_history' - a python List of tuples containing accumulating question -answer pairs
    2. Lload the local FAISS database ehere the entire website is stored as Embedding vectors
    3. Create a ConversationalBufferMemory object with chat_history'
    4. Create a Conversational RetrievalChain object with the FAISS DB as the Retriever (LLM lets us create Retriever objects against dara stores)
    5. Invoke the Retriever object with the Query and Chat History
    6. Returns the response
    """

    chat_history = build_chat_history(chat_history)
    embeddings = OpenAIEmbeddings(base_url="https://api.avalai.ir/v1", api_key="aa-mom4ge07aOt9jqa70NYPvZLvHiUUMg2ip89VgoeDWHglsKM8")
    new_db = FAISS.load_local("faiss_index" , embeddings , allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name = 'gpt-4o', temperature=0,  base_url="https://api.avalai.ir/v1", api_key="aa-mom4ge07aOt9jqa70NYPvZLvHiUUMg2ip89VgoeDWHglsKM8") 
    

    condense_question_system_template = (

        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT  answer the question, "
        "just reformulate it if needed and otherwise return it as is."

    )

    condense_question_prompt = ChatPromptTemplate.from_messages(

        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),

        ]
    )


    history_aware_retriever = create_history_aware_retriever(

        llm, new_db.as_retriever(), condense_question_prompt


    )

    system_prompt = (

        "You are an assistant for question-answering tasks on lifeandme company customer support. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"

    )

    qa_prompt = ChatPromptTemplate.from_messages(


        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),

        ]
    )


    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return convo_qa_chain.invoke(

        {
            "input": question,
            "chat_history": chat_history,

        }

    )

        
def show_ui():

    st.title("LifeandMe Customer Support ChatBot")
    st.image("BOT.png")
    st.subheader("Please enter your query")


    if "messages" not in st.session_state:

        st.session_state.messages = []
        st.session_state.chat_history = []


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("Enter your LifeandMe related Query: "):

        with st.spinner("Working on your query..."):
            response = query(question=prompt, chat_history=st.session_state.chat_history)

        # ✅ Save user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").markdown(prompt)

        # ✅ Save AI response
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.chat_message("assistant").markdown(response["answer"])

        # ✅ Append to chat history
            st.session_state.chat_history.append((prompt, response["answer"]))  # Use append instead of extend


if __name__ == "__main__":

        show_ui()
            

                



            
    
    
