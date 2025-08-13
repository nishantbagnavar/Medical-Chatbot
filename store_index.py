from dotenv import load_dotenv
import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helpers import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Prepare data
extracted_data = load_pdf_files("data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)
embedding = download_embeddings()

# Create index if not exists
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)

# Connect to the index
index = pc.Index(index_name)

# Store documents in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name
)
