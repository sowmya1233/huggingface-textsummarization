import streamlit as st
import pdfplumber
import docx
from transformers import pipeline

# Load models
summarizer = pipeline("summarization", model="t5-small")
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
grammar_fixer = pipeline("text2text-generation", model="t5-small")

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Streamlit App
st.title("📄 AI Document Assistant (Hugging Face NLP)")

uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_docx(uploaded_file)

    st.subheader("📃 Extracted Text:")
    st.text_area("Document Content", text, height=200)

    st.subheader("🛠 Choose Action:")

    action = st.selectbox("Select a task", ["Summarize", "Ask a Question", "Fix Grammar"])

    if action == "Summarize":
        if st.button("Summarize"):
            summary = summarizer("summarize: " + text[:1000], max_length=100, min_length=30, do_sample=False)
            st.success(summary[0]['summary_text'])

    elif action == "Ask a Question":
        question = st.text_input("Ask your question:")
        if st.button("Get Answer") and question:
            answer = qa(question=question, context=text[:1000])
            st.success(answer['answer'])

    elif action == "Fix Grammar":
        if st.button("Correct Grammar"):
            corrected = grammar_fixer("fix grammar: " + text[:512])[0]['generated_text']
            st.success(corrected)
