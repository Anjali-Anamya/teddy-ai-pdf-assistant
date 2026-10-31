
from langchain_core.prompts import ChatPromptTemplate


def get_prompt():

    template = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            You are a helpful AI assistant.

            Answer questions using only the provided document context.

            Follow these rules:
            1. Use only the information present in the context.
            2. Do not make up facts or information.
            3. If the answer is not present in the context,
               clearly say that you don't know.
            4. Give clear, accurate and well-structured answers.
            5. Use bullet points wherever appropriate.
            """
        ),

        (
            "human",
            """
            Document Context:

            {context}

            Question:

            {question}

            Provide a clear and accurate answer.
            """
        )

    ])

    return template
