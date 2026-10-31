
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def load_pdf():

    reader = PdfReader(str(PDF_PATH))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():

            document = Document(
                page_content=text,
                metadata={
                    "source": PDF_PATH.name,
                    "page": page_number
                }
            )

            documents.append(document)

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    return chunks


def process_pdf():

    documents = load_pdf()

    chunks = split_documents(documents)

    print(f"Total pages loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    return chunks
