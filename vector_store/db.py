
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    EMBEDDING_MODEL,
    CHROMA_PATH
)


def get_embeddings():

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    return embeddings


def create_vectorstore(chunks):

    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name="pdf_rag_collection",
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )

    existing_count = vectorstore._collection.count()

    if existing_count == len(chunks) and existing_count > 0:

        print("Existing ChromaDB loaded!")

        return vectorstore

    if existing_count > 0:

        raise ValueError(
            "Existing database has a different number of chunks. "
            "Please remove the chroma_db folder and rebuild it."
        )

    if not chunks:

        raise ValueError("No document chunks found.")

    print("Creating new ChromaDB...")

    vectorstore.add_documents(chunks)

    print("Documents successfully stored in ChromaDB!")

    return vectorstore
