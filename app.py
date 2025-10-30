import streamlit as st
import pdfplumber
import docx
from transformers import pipeline

# --- Load models with caching ---
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="t5-small", framework="pt", device=-1)
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2", framework="pt", device=-1)
    grammar_fixer = pipeline("text2text-generation", model="t5-small", framework="pt", device=-1)
    return summarizer, qa, grammar_fixer

summarizer, qa, grammar_fixer = load_models()

# --- Helper functions ---
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def truncate_text(text, max_chars=2000):
    return text[:max_chars] if len(text) > max_chars else text

# --- Streamlit UI ---
st.title("📄 AI Document Assistant")

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
            summary = summarizer(
                "summarize: " + truncate_text(text),
                max_length=150,
                min_length=50,
                do_sample=False
            )
            st.success(summary[0]['summary_text'])

    elif action == "Ask a Question":
        question = st.text_input("Ask your question:")
        if st.button("Get Answer") and question:
            answer = qa(
                question=question,
                context=truncate_text(text)
            )
            st.success(answer['answer'])

    elif action == "Fix Grammar":
        if st.button("Correct Grammar"):
            corrected = grammar_fixer(
                "fix grammar: " + truncate_text(text, max_chars=512)
            )[0]['generated_text']
            st.success(corrected)
