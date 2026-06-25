from dotenv import load_dotenv
import os

from src.utils import (
    load_pdfs_from_directory,
    filter_to_minimal_docs,
    split_documents,
    download_embeddings
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing in .env")

# --------------------------------------------------

print("\n========== STARTING ==========\n")

# --------------------------------------------------
# Load PDFs
# --------------------------------------------------

print("Loading PDFs...")

documents = load_pdfs_from_directory("data")

print(f"Loaded {len(documents)} pages")

# --------------------------------------------------
# Metadata Cleanup
# --------------------------------------------------

documents = filter_to_minimal_docs(
    documents
)

print("Metadata Cleaned")

# --------------------------------------------------
# Chunking
# --------------------------------------------------

text_chunks = split_documents(
    documents,
    chunk_size=800,
    chunk_overlap=100
)

print(f"Total Chunks: {len(text_chunks)}")

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

print("Loading Embeddings...")

embeddings = download_embeddings()

print("Embeddings Ready")

# --------------------------------------------------
# Pinecone
# --------------------------------------------------

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index_name = "medical-research"

if not pc.has_index(index_name):

    print("Creating Pinecone Index...")

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Index Created")

else:

    print("Index Already Exists")

# --------------------------------------------------
# Upload
# --------------------------------------------------

print("Uploading Chunks...")

PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

print("\nDocuments Uploaded Successfully")
print("\n========== COMPLETE ==========")