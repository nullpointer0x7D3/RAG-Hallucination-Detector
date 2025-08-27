import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader # a PDF file parser, outputs chunks | this one does not process images or tables
from langchain.vectorstores import Qdrant

# uses sentencetransformer which is a library to download model weights, no API key needed, caches weights locally.
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings") # use relevant medical embedding model 

print("Loading documents...")
loader = DirectoryLoader('Data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader) # load docs from data, initalize PDF parser
documents = loader.load()
print(f"Loaded {len(documents)} documents")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,       # ~1,250 tokens per chunk
    chunk_overlap=250,      # 10% overlap for continuity
    separators=["\n\n", "\n", ". ", " ", ""]
)

texts = text_splitter.split_documents(documents)
print(f"Created {len(texts)} text chunks")
print(f"Average chunk size: {sum(len(t.page_content) for t in texts) / len(texts):.0f} characters")

# Clear existing database and create new one
url = "http://localhost:6333"
from qdrant_client import QdrantClient

client = QdrantClient(url=url, prefer_grpc=False)

# Delete existing collection if it exists
try:
    client.delete_collection("vector_db")
    print("Deleted existing vector_db collection")
except:
    print("No existing collection to delete")

# Create new collection with smaller chunks
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB Successfully Created with smaller chunks!")
print(f"Total chunks in database: {len(texts)}")
