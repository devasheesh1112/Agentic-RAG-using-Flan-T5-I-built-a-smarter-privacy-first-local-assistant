import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def initialize_vector_db(folder_path):
    # Load
    docs = [] 
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    
    # Chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = text_splitter.split_documents(docs)
    
    # Vectorize
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = [c.page_content for c in chunks]
    db = Chroma(collection_name="rag_store", embedding_function=embedding_model)
    db.add_texts(texts)
    
    return db.as_retriever(search_kwargs={"k": 3})