from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings



persist_dir = 'db/chroma_db'

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model,
    collection_metadata={'hnsw:space': 'cosine'}
)

query = "what year tesla has started"

retriver = db.as_retriever(search_kwargs={'k': 5})

relevent_docs = retriver.invoke(query)

print('your question ----> ', query)

print('_____context_____')

for i, doc in enumerate(relevent_docs, 1):
    print(f"Chunk {i}: \n {doc.page_content} \n")