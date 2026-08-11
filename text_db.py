from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="./vector_db",
    settings=Settings(anonymized_telemetry=False)
)

db = Chroma(
    client=client,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k":3})

docs = retriever.invoke("What is Python?")

# for i, doc in enumerate(docs):
#     print("="*60)
#     print(i+1)
#     print(doc.page_content[:500])
for i, doc in enumerate(docs):
    print("=" * 60)
    print(f"Document {i+1}")
    print(doc.page_content)

for i, doc in enumerate(docs):
    print("=" * 60)
    print(f"Document {i+1}")
    print(doc.page_content)