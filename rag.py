import os

# Set environment variables BEFORE any other imports
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["OPENAI_API_KEY"] = "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"

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
import openai

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

def classify_hallucination_severity(question, answer, faithfulness_score, answer_relevancy_score):
    """
    Classify the severity of a detected hallucination using OpenAI
    """
    try:
        # Set up OpenAI client
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Create the classification prompt
        classification_prompt = f"""
You are a medical AI safety expert. Analyze the following medical AI response for hallucination severity.

QUESTION: {question}

AI ANSWER: {answer}

RAGAS METRICS:
- Faithfulness Score: {faithfulness_score:.3f} (measures grounding in source material)
- Answer Relevancy Score: {answer_relevancy_score:.3f} (measures relevance to question)

HALLUCINATION SEVERITY CLASSIFICATION FRAMEWORK:

CRITICAL (4): 
- Contains potentially harmful medical misinformation
- Could lead to dangerous medical decisions
- Contradicts established medical facts
- Makes specific medical claims without proper grounding

HIGH (3):
- Contains significant factual errors
- Could mislead medical professionals
- Makes broad medical generalizations
- Contains partially incorrect information

MODERATE (2):
- Contains minor factual inaccuracies
- Somewhat misleading but not dangerous
- Lacks proper medical context
- Contains vague or unclear information

LOW (1):
- Minor inconsistencies with source material
- Slightly off-topic but not harmful
- Lacks detail but generally safe
- Minor formatting or clarity issues

SAFE (0):
- No significant hallucination detected
- Response is accurate and well-grounded
- Appropriate for medical context

Based on the question, answer, and RAGAS scores, classify the hallucination severity.

Respond with ONLY a JSON object in this exact format:
{{
    "severity_level": [0-4],
    "severity_label": "[SAFE/LOW/MODERATE/HIGH/CRITICAL]",
    "confidence": [0.0-1.0],
    "reasoning": "Brief explanation of why this severity was assigned",
    "risk_factors": ["list", "of", "specific", "risk", "factors"],
    "recommendation": "Specific recommendation for handling this response"
}}
"""

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical AI safety expert specializing in hallucination detection and severity classification."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=500
        )
        
        # Parse the response
        result = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                severity_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing OpenAI response: {e}")
            print(f"Raw response: {result}")
            # Fallback classification based on scores
            if faithfulness_score < 0.3 or answer_relevancy_score < 0.3:
                severity_data = {
                    "severity_level": 4,
                    "severity_label": "CRITICAL",
                    "confidence": 0.8,
                    "reasoning": "Very low RAGAS scores indicate critical hallucination risk",
                    "risk_factors": ["Extremely low faithfulness", "Very low relevancy"],
                    "recommendation": "Do not use this response for medical decisions"
                }
            elif faithfulness_score < 0.5 or answer_relevancy_score < 0.5:
                severity_data = {
                    "severity_level": 3,
                    "severity_label": "HIGH",
                    "confidence": 0.7,
                    "reasoning": "Low RAGAS scores indicate high hallucination risk",
                    "risk_factors": ["Low faithfulness", "Low relevancy"],
                    "recommendation": "Verify all information before use"
                }
            else:
                severity_data = {
                    "severity_level": 2,
                    "severity_label": "MODERATE",
                    "confidence": 0.6,
                    "reasoning": "Moderate RAGAS scores indicate some hallucination risk",
                    "risk_factors": ["Below threshold scores"],
                    "recommendation": "Review response carefully"
                }
        
        return severity_data
        
    except Exception as e:
        print(f"Error in hallucination severity classification: {e}")
        # Fallback classification
        if faithfulness_score < 0.3 or answer_relevancy_score < 0.3:
            return {
                "severity_level": 4,
                "severity_label": "CRITICAL",
                "confidence": 0.5,
                "reasoning": "Classification failed - using fallback based on very low scores",
                "risk_factors": ["System error", "Very low RAGAS scores"],
                "recommendation": "Do not use this response"
            }
        else:
            return {
                "severity_level": 2,
                "severity_label": "MODERATE",
                "confidence": 0.5,
                "reasoning": "Classification failed - using fallback",
                "risk_factors": ["System error", "Below threshold scores"],
                "recommendation": "Review response carefully"
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
    
    # Check if hallucination is detected and classify severity
    hallucination_detected = ragas_metrics['faithfulness'] < 0.7 or ragas_metrics['answer_relevancy'] < 0.7
    severity_data = None
    
    if hallucination_detected:
        print("Hallucination detected - classifying severity...")
        severity_data = classify_hallucination_severity(
            query, 
            answer, 
            ragas_metrics['faithfulness'], 
            ragas_metrics['answer_relevancy']
        )
        print(f"Severity classification: {severity_data['severity_label']} (Level {severity_data['severity_level']})")
    
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
        "llm_used_docs_count": len(llm_used_docs),  # Show how many docs LLM actually used
        "hallucination_detected": hallucination_detected,
        "severity_data": severity_data
    }))
    
    res = Response(response_data)
    return res