from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings") # state same embedding model as ingest.py

url = "http://localhost:6333"

client = QdrantClient( # initialize connection object to Qdrant server
    url=url, prefer_grpc=False
)

print(client)
print("##############")

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

print(db)
print("######")
query = "What is Metastatic disease?"

docs = db.similarity_search_with_score(query=query, k=2) # you ask the database for the most similar documents to the query
for i in docs:                                           # you take the prompt and use the embedding model to properly transform it
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})

# the above code print each retrieved document with its score, relative to the number of kwarg