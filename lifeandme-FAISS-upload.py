from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ this is the new one

import os


def upload_documents():
    """
    This function:
    1. Loads both HTML and PDF documents from a given folder.
    2. Splits the loaded text into chunks.
    3. Converts chunks into embeddings.
    4. Stores them in a FAISS vector database.
    """

    base_folder = "Lifeandme_intro"
    documents = []

    # Load all files in directory
    for file_name in os.listdir(base_folder):
        file_path = os.path.join(base_folder, file_name)

        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith(".html") or file_name.endswith(".htm"):
            loader = UnstructuredFileLoader(file_path)
        else:
            continue  # skip unsupported file types

        docs = loader.load()
        documents.extend(docs)

    print(f"{len(documents)} Pages Loaded")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )

    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} Documents...")

    print(split_documents[0].metadata)

    # Create embeddings and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


        #base_url="https://api.avalai.ir/v1",
        #api_key="aa-83NkaoDGyrW6YntO09P9kr9veMZlNBs2xblWf1hdB8OpPs3N"
    

    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local("faiss_index")


def faiss_query():

    """
    This function does the following:
    1. Load the local FAISS Database
    2. Trigger a Semantic Similarity Search using a Query
    3. This retrieves semantically matching Vectors from the DB

    """

    

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    

    query = "Explain the Candidate Onboarding  process."
    docs = new_db.similarity_search(query)


    for doc in docs:

        print("##---- Page ---##")
        print(doc.metadata['source'])
        print("---- Content ---##")
        print(doc.page_content)


if __name__ == "__main__":

    upload_documents()

    faiss_query()
    
