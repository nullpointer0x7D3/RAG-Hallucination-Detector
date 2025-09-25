"""
Document ingestion script for Medical RAG System
Creates and populates the vector database with medical documents
"""

from utils import load_documents_into_db

if __name__ == "__main__":
    # Load documents into database
    db = load_documents_into_db()
    print("Document ingestion completed!")
