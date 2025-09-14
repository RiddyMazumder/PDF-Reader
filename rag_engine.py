from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

def _device():
    try:
        # check if torch and cuda are available
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1 # if no torch, use cpu

def _extractive_pipeline():
    # span extractor
    return pipeline("question-answering", model="deepset/roberta-base-squad2", device=_device())

def _generative_llm():
    # small, local-friendly text2text model; works with RetrievalQA
    gen = pipeline("text2text-generation", model="google/flan-t5-base", device=_device())
    return HuggingFacePipeline(pipeline=gen)
    
def load_documents(pdf_paths):
    all_documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

def create_vector_store(documents):
    
    # split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # create vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store

def search_documents(query, documents):
    # Create vector store
    vector_store = create_vector_store(documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Load QA pipeline with extractive model
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    # Create Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )

    # Ask question
    answer = qa_chain.run(query)
    return answer




