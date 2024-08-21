import streamlit as st
import PyPDF2
import re

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Function to preprocess the text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Initialize the question-answering pipeline
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


# Streamlit app
st.title("PDF Question Answering")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Extract and preprocess text from the PDF
    text = extract_text_from_pdf(uploaded_file)
    preprocessed_text = preprocess_text(text)
    
    st.write("Text extracted from the PDF:")
    st.write(preprocessed_text[:2000])  # Display the first 2000 characters for preview
    
    # Input for user queries
    query = st.text_input("Enter your question:")
    
    if query:
        # Answer the user's question
        answer = qa_pipeline({
            'question': query,
            'context': preprocessed_text
        })['answer']
        
        st.write(f"Answer: {answer}")
