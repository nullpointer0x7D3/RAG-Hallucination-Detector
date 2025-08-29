from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings") # state same embedding model as ingest.py

# Use Chroma as an in-memory vector store (no external service needed)
db = Chroma(
    embedding_function=embeddings,
    collection_name="vector_db"
)

print(db)
print("######")
query = "What is Metastatic disease?"

docs = db.similarity_search_with_score(query=query, k=2) # you ask the database for the most similar documents to the query
for i in docs:                                           # you take the prompt and use the embedding model to properly transform it
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})

# the above code print each retrieved document with its score, relative to the number of kwarg