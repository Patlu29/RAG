import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
 
def load_docs(path='docs'): 
    if not os.path.exists(path):
        raise FileNotFoundError('folder not found')

    loader = DirectoryLoader(
        path = path,
        glob = '*.txt',
        loader_cls = TextLoader
    )
    documents = loader.load()

    if(len(documents) == 0): 
        raise FileNotFoundError('No docs found')

    for i, doc in enumerate(documents[:2]):
        print(doc.metadata['source'])
        print(doc.page_content[:100], len(doc.page_content))

    return documents




def split_docs(docs, chunk_size, chunk_overlap): # chunking
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size ,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(docs)
    
    if chunks: 
        for i, chunk in enumerate(chunks[:5]):
            print(chunk.metadata['source'])
            print(chunk.page_content, len(chunk.page_content))
            print('_' * 50)

    return chunks

def vector_store(chunks, persist_dir='db/chroma_db'):

    embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_metadata={
            'hnsw:space': 'cosine'
        }
    )

    print(vector_store)
    return vector_store



def main(): 
    docs = load_docs(path='docs')
    chunks = split_docs(docs,500,0)

    vector_store(chunks)




if __name__ == "__main__":
    main()