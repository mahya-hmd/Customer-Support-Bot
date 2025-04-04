from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


def upload_html():

    """
    This function does the following:
    1. Reads recursively through the given folder hr-polices (without current folder)
    2. Loads the pages (Documents)
    3. Loaded documents are split into chunks using Splitter
    4. These chunks are converted into language Embeddings and loaded as vectors into a local FAISS Vectors Database
    """


    loader = DirectoryLoader(path=r"Lifeandme_intro")
    documents = loader.load()
    print(f"{len(documents)} Pages Loaded")


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )


    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} Documents...")


    print(split_documents[0].metadata)

    embeddings = OpenAIEmbeddings(base_url="https://api.avalai.ir/v1", api_key="aa-mom4ge07aOt9jqa70NYPvZLvHiUUMg2ip89VgoeDWHglsKM8")
    db = FAISS.from_documents(split_documents, embeddings)
    
    db.save_local("faiss_index")


def faiss_query():

    """
    This function does the following:
    1. Load the local FAISS Database
    2. Trigger a Semantic Similarity Search using a Query
    3. This retrieves semantically matching Vectors from the DB

    """

    

    embeddings = OpenAIEmbeddings(base_url="https://api.avalai.ir/v1", api_key="aa-mom4ge07aOt9jqa70NYPvZLvHiUUMg2ip89VgoeDWHglsKM8")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    

    query = "Explain the Candidate Onboarding  process."
    docs = new_db.similarity_search(query)


    for doc in docs:

        print("##---- Page ---##")
        print(doc.metadata['source'])
        print("---- Content ---##")
        print(doc.page_content)


if __name__ == "__main__":

    upload_html()

    faiss_query()
    
