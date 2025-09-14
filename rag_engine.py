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

def _is_openended(question: str) -> bool:
    # simple heuristic: if question starts with "what", "how", "why", etc, it's open-ended
    open_ended_starts = ("what", "how", "why", "explain", "describe", "summarize", "overview", "describe", "bullet", "high level", "strengths", "summary", "compare")
    return any(start in question.lower() for start in open_ended_starts)

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
    retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # retrieve top 10 chunks (relevant)

    # extractive q and a for the non-open-ended questions
    e_pipe = _extractive_pipeline()  # load the HF extractive QA pipeline here 
    best = {"answer": None, "score": -1.0}  # keep track of the best answer seen so far and its confidence score
    top_chunks = retriever.get_relevant_documents(query)  # fetch top k chunks 

    for chunk in top_chunks:
        try:
            output = e_pipe({"question": query, "context": chunk.page_content})  # run extractive q and a on this chunk
            if output and output.get("score", 0) > best["score"]:  # result exists + has a higher score than current best, so update
                best = {"answer": output.get("answer", ""), "score": float(output.get("score", 0))}  
        except Exception:  
            pass  

    if best["answer"] and best["score"] > 0.1 and not _is_openended(query):  # if we found a decent answer, return it


    # generative llm for open-ended questions or if no good extractive answer found
    gen_llm = _generative_llm() 
    # Create Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=gen_llm,
        retriever=retrieve,
        return_source_documents=False
    )

    # Ask question
    answer = qa_chain.run(query)
    return answer




