from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def load_documents(pdf_paths):
    all_documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

def create_vector_store(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store

def search_documents(query, documents):
    # Create vector store
    vector_store = create_vector_store(documents)

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




