from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_MODEL, TEMPERATURE
from rag.prompt import get_prompt


def create_rag_chain(retriever):

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE
    )

    # Get prompt template
    prompt = get_prompt()

    def rag_pipeline(question):

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(question)

        # Combine retrieved content
        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        if not context.strip():
            return {
                "answer": "I don't know. No relevant information was found in the document.",
                "sources": []
            }

        # Generate prompt
        formatted_prompt = prompt.invoke({
            "context": context,
            "question": question
        })

        # Generate response
        response = llm.invoke(formatted_prompt)

        # Extract source information
        sources = []

        for doc in retrieved_docs:
            sources.append({
                "page": doc.metadata.get("page", "Unknown"),
                "source": doc.metadata.get("source", "Unknown")
            })

        # Extract answer
        answer = response.content

        if isinstance(answer, list):
            answer = "\n".join(
                block["text"]
                for block in answer
                if isinstance(block, dict)
                and block.get("type") == "text"
            )

        # Return answer and sources
        return {
            "answer": answer,
            "sources": sources
        }

    return rag_pipeline