from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from typing import List


def load_pdfs_from_directory(directory_path):
    print("Creating Directory Loader...")

    loader = DirectoryLoader(
        directory_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    print("Loading PDFs...")
    documents = loader.load()

    print("Finished Loading PDFs")
    print("Total Documents:", len(documents))

    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    print("Filtering Metadata...")

    minimal_docs = []

    for doc in docs:
        src = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    print("Metadata Filtering Complete")
    return minimal_docs


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    print("Splitting Documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    text_chunks = text_splitter.split_documents(documents)

    print("Chunks Created:", len(text_chunks))

    return text_chunks


def download_embeddings():
    print("Loading Embedding Model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding Model Loaded Successfully")

    return embeddings