import chromadb

client = chromadb.PersistentClient(path="./vector_db")

collection = client.get_collection("langchain")

print("Chunks:", collection.count())