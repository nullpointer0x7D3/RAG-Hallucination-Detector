import os

# Set environment variables BEFORE any other imports
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
# OpenAI API key will be loaded from environment variable
# Check if OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY environment variable is not set!")
    print("Please set it using: set OPENAI_API_KEY=your_key_here (Windows) or export OPENAI_API_KEY=your_key_here (Linux/Mac)")

from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from langchain.vectorstores import Chroma
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")# get html template 
app.mount("/static", StaticFiles(directory="static"), name="static") 

# Using Ollama with mistral:instruct model instead of local GGUF file

# Configuration for the LLM
max_new_tokens = 1024      # total allowed output tokens, how long the response can be
temperature = 0.2          # Slightly higher for better generation
top_k = 40                # Reduce for more focused responses
top_p = 0.85              # Slightly more conservative
repeat_penalty = 1.15     # Stronger repetition penalty


llm = Ollama(
    model="mistral:instruct",
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    repeat_penalty=repeat_penalty,
    num_predict=max_new_tokens,  # Max tokens to generate
    verbose=True
)


print("LLM Initialized....")
################################################################################################################################
prompt_template = """You are a medical expert assistant. Use the following medical document context to answer the question accurately and professionally.

Context: {context}

Question: {question}

Instructions:
- Provide a clear, accurate medical answer based on the context
- If the context doesn't contain enough information, state this clearly
- Use professional medical terminology appropriately
- Keep your response focused and concise

Medical Answer:"""

################################################################################################################################
# Vector Database Setup 
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

def load_documents_into_db():
    """Load documents into the vector database"""
    global db
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import DirectoryLoader
    from langchain.document_loaders import PyPDFLoader
    
    print("Loading documents...")
    loader = DirectoryLoader('Data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
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
        collection_name="vector_db",
        persist_directory="./chroma_db"  # Persist to disk
    )
    
    print("Vector DB Successfully Created!")
    print(f"Total chunks in database: {len(texts)}")

# Try to load existing database, or create new one if it doesn't exist
try:
    # Try to load existing database
    db = Chroma(
        embedding_function=embeddings,
        collection_name="vector_db",
        persist_directory="./chroma_db"  # Persist to disk
    )
    
    # Check if database has documents
    existing_docs = db.similarity_search("test", k=1)
    if len(existing_docs) == 0:
        print("No documents found in existing database. Loading documents...")
        load_documents_into_db()
    else:
        print(f"Loaded existing database with documents")
        
except Exception as e:
    print(f"Could not load existing database: {e}")
    print("Creating new database and loading documents...")
    load_documents_into_db()

################################################################################################################################

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question']) # feed prompt into vector dimensions

# Single retriever configuration - now with much larger context window
retriever = db.as_retriever(search_kwargs={"k":6})  # Get 6 documents with larger context

def calculate_ragas_metrics(question, answer, contexts):
    """
    Calculate RAGAS metrics for faithfulness and answer relevancy
    """
    try:
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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request}) # handles FASTAPI requests, seperate thread

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}
    
    # Execute RAG chain using the single retriever
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,  # Use the single retriever
        return_source_documents=True, 
        chain_type_kwargs=chain_type_kwargs, 
        verbose=True
    )
    response = qa(query)
    
    print(response)
    answer = response['result']
    
    # Get the EXACT documents that were used by the LLM
    # RetrievalQA returns source_documents in the response
    llm_used_docs = response.get('source_documents', [])
    
    # Get the SAME documents that were used in the RAG chain
    retrieval_docs = retriever.get_relevant_documents(query)  # Use the same retriever
    
    # All retrieved documents with scores (using the same retriever)
    all_retrievals = []
    for i, doc in enumerate(retrieval_docs):
        # Get similarity score for this document
        score_result = db.similarity_search_with_score(query, k=len(retrieval_docs))
        score = next((score for doc_with_score, score in score_result if doc_with_score.page_content == doc.page_content), 0.0)
        
        all_retrievals.append({
            "rank": i + 1,
            "content": doc.page_content,
            "source": doc.metadata.get('source', 'Unknown'),
            "score": float(score),
            "page": doc.metadata.get('page', 'N/A'),
            "is_primary": i < len(llm_used_docs)  # Mark as primary if actually used by LLM
        })
    
    # Calculate RAGAS metrics using ONLY the documents the LLM actually used
    contexts = [doc.page_content for doc in llm_used_docs]
    ragas_metrics = calculate_ragas_metrics(query, answer, contexts)
    
    # Add debug info about what the LLM actually used
    print(f"LLM used {len(llm_used_docs)} documents for RAGAS evaluation")
    print(f"RAGAS contexts: {len(contexts)}")
    
    response_data = jsonable_encoder(json.dumps({
        "answer": answer,
        "primary_source_document": retrieval_docs[0].page_content if retrieval_docs else "",
        "primary_doc": retrieval_docs[0].metadata.get('source', 'Unknown') if retrieval_docs else "",
        "all_retrievals": all_retrievals,
        "query": query,
        "total_retrievals": len(all_retrievals),
        "ragas_metrics": ragas_metrics,
        "llm_used_docs_count": len(llm_used_docs)  # Show how many docs LLM actually used
    }))
    
    res = Response(response_data)
    return res