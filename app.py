import streamlit as st
from transformers import pipeline
import PyPDF2
import re
import torch  # Ensure torch is imported

# Title of the app
st.title("Smart PDF Question Answering App")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file is not None:
    # Extract text from the PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    st.subheader("Extracted Resume Content:")
    st.text_area("Text from PDF", value=text, height=300)

    # Set device to CPU explicitly (this avoids issues with CUDA/meta tensor)
    device = -1  # Use CPU for inference
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)

    # Function to handle special queries like 'Skills', 'Objective', etc.
    def handle_special_queries(question, context):
        # Check for "skills" query
        if "skills" in question.lower():
            skills_section = re.search(r'SKILLS\s*([^A-Z]+)', context, re.DOTALL)
            if skills_section:
                return skills_section.group(1).strip()
            else:
                return "Sorry, I couldn't find the Skills section in the resume."
        
        # Check for "objective" query
        elif "objective" in question.lower():
            objective_section = re.search(r'OBJECTIVE\s*([^A-Z]+)', context, re.DOTALL)
            if objective_section:
                return objective_section.group(1).strip()
            else:
                return "Sorry, I couldn't find the Objective section in the resume."
        
        # Check for "projects" query
        elif "projects" in question.lower():
            projects_section = re.search(r'PROJECTS\s*([^A-Z]+)', context, re.DOTALL)
            if projects_section:
                return projects_section.group(1).strip()
            else:
                return "Sorry, I couldn't find the Projects section in the resume."
        
        # Check for "education" query
        elif "education" in question.lower():
            education_section = re.search(r'EDUCATION\s*([^A-Z]+)', context, re.DOTALL)
            if education_section:
                return education_section.group(1).strip()
            else:
                return "Sorry, I couldn't find the Education section in the resume."
        
        return None  # If no special query matched, return None

    # User input for question
    st.subheader("Ask a Question About the Resume:")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question:
            # First, handle special queries (e.g., skills, objective, projects)
            special_answer = handle_special_queries(question, text)
            if special_answer:
                st.success(f"Answer: {special_answer}")
            else:
                try:
                    result = qa_pipeline({"question": question, "context": text})
                    if 'answer' in result and result['answer']:
                        st.success(f"Answer: {result['answer']}")
                    else:
                        st.warning("Sorry, I couldn't find any relevant information.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question.")












# # app.py
# import streamlit as st
# from PyPDF2 import PdfReader
# from transformers import pipeline

# # Load QA pipeline
# qa_pipeline = pipeline("question-answering")

# # Streamlit app title
# st.title("Smart Resume Question Answering")

# # PDF file uploader
# uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# context = ""
# if uploaded_file is not None:
#     # Read PDF content
#     pdf_reader = PdfReader(uploaded_file)
#     context = ""
#     for page in pdf_reader.pages:
#         context += page.extract_text()

#     st.subheader("Extracted Resume Content:")
#     st.text_area("Text from PDF", value=context, height=300)

# # Question input
# question = st.text_input("Ask a question about the resume:")

# # QA button
# if st.button("Get Answer"):
#     if not context.strip():
#         st.warning("Please upload a resume first.")
#     elif not question.strip():
#         st.warning("Please enter a question.")
#     else:
#         try:
#             result = qa_pipeline({
#                 "question": question,
#                 "context": context
#             })
#             st.success(f"Answer: {result['answer']}")
#         except Exception as e:
#             st.error(f"Error: {e}")




