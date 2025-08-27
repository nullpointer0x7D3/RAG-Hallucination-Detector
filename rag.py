from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
import os
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

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

################################################################################################################################

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question']) # feed prompt into vector dimensions

# Single retriever configuration - now with much larger context window
retriever = db.as_retriever(search_kwargs={"k":6})  # Get 6 documents with larger context

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
            "is_primary": i < 2  # Mark first 2 as primary (used in RAG chain)
        })
    
    response_data = jsonable_encoder(json.dumps({
        "answer": answer,
        "primary_source_document": retrieval_docs[0].page_content if retrieval_docs else "",
        "primary_doc": retrieval_docs[0].metadata.get('source', 'Unknown') if retrieval_docs else "",
        "all_retrievals": all_retrievals,
        "query": query,
        "total_retrievals": len(all_retrievals)
    }))
    
    res = Response(response_data)
    return res