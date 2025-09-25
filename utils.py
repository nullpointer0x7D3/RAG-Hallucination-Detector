"""
Shared utilities for Medical RAG System
Contains common functions to avoid code duplication
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
import json
import re

# Global embeddings - initialized once
_embeddings = None

def get_embeddings():
    """Get or create the shared embeddings instance"""
    global _embeddings
    if _embeddings is None:
        _embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    return _embeddings

def load_documents_into_db(data_dir="Data/", persist_directory="./chroma_db", collection_name="vector_db"):
    """
    Load documents into the vector database
    Returns the created database instance
    """
    embeddings = get_embeddings()
    
    print("Loading documents...")
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
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

    # Create new database with documents
    db = Chroma.from_documents(
        texts,
        embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    print("Vector DB Successfully Created!")
    print(f"Total chunks in database: {len(texts)}")
    return db

def load_existing_db(persist_directory="./chroma_db", collection_name="vector_db"):
    """
    Load existing vector database
    Returns the database instance or None if not found
    """
    try:
        embeddings = get_embeddings()
        db = Chroma(
            embedding_function=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Check if database has documents
        existing_docs = db.similarity_search("test", k=1)
        if len(existing_docs) == 0:
            print("No documents found in existing database.")
            return None
        else:
            print(f"Loaded existing database with documents")
            return db
            
    except Exception as e:
        print(f"Could not load existing database: {e}")
        return None

def calculate_ragas_metrics(question, answer, contexts, verbose=True):
    """
    Calculate RAGAS metrics for faithfulness and answer relevancy
    """
    try:
        if verbose:
            # Print RAGAS evaluation details in a clean format
            print("\n" + "="*80)
            print("RAGAS EVALUATION DETAILS")
            print("="*80)
            
            print(f"QUESTION:")
            print(f"   {question}")
            print()
            
            print(f"LLM ANSWER:")
            print(f"   {answer}")
            print()
            
            print(f"CONTEXTS USED ({len(contexts)} documents):")
            for i, context in enumerate(contexts, 1):
                print(f"   Document {i}:")
                # Truncate long contexts for readability
                if len(context) > 300:
                    print(f"      {context[:300]}...")
                else:
                    print(f"      {context}")
                print()
        
        # Create dataset for RAGAS evaluation
        data = [{
            "question": question,
            "answer": answer,
            "contexts": contexts
        }]
        
        dataset = Dataset.from_list(data)
        
        if verbose:
            print(f"Running RAGAS evaluation...")
        
        # Run RAGAS evaluation
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy]
        )
        
        # Extract scores (handle both list and single value formats)
        faithfulness_score = results['faithfulness']
        if isinstance(faithfulness_score, list):
            faithfulness_score = faithfulness_score[0] if faithfulness_score else 0
            
        answer_relevancy_score = results['answer_relevancy']
        if isinstance(answer_relevancy_score, list):
            answer_relevancy_score = answer_relevancy_score[0] if answer_relevancy_score else 0
        
        if verbose:
            print(f"RAGAS RESULTS:")
            print(f"   Faithfulness: {faithfulness_score:.3f}")
            print(f"   Answer Relevancy: {answer_relevancy_score:.3f}")
            print("="*80)
        
        return {
            'faithfulness': round(faithfulness_score, 3),
            'answer_relevancy': round(answer_relevancy_score, 3)
        }
        
    except Exception as e:
        print(f"Error calculating RAGAS metrics: {e}")
        return {
            'faithfulness': 0.0,
            'answer_relevancy': 0.0
        }

def extract_ragas_scores(results, verbose=True):
    """
    Extract RAGAS scores from results, handling both list and single value formats
    """
    scores = {}
    
    for metric_name in ['faithfulness', 'answer_relevancy', 'context_recall', 'answer_similarity']:
        if metric_name in results:
            score = results[metric_name]
            if isinstance(score, list):
                score = score[0] if score else 0
            scores[metric_name] = score
        else:
            scores[metric_name] = 0.0
    
    if verbose:
        print("\nRAGAS Scores:")
        for metric, score in scores.items():
            print(f"{metric.replace('_', ' ').title()}: {score:.3f}")
    
    return scores

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
