
from config import RETRIEVER_K, RETRIEVER_FETCH_K


def create_retriever(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K
        }
    )

    return retriever
