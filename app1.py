import streamlit as st
import pandas as pd
import docx
import pdfplumber
import os
import json
from datetime import datetime


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "data_store.json")


def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data(new_record):
    data = load_data()
    data.append(new_record)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_txt(file):
    return file.read().decode("utf-8")

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def create_record(filename, source_type, file_type, content):
    return {
        "id": int(datetime.now().timestamp()),  
        "filename": filename,
        "source_type": source_type,
        "file_type": file_type,
        "content": content,
        "upload_time": datetime.now().isoformat()
    }


st.title("📂 Dynamic Text Analysis - Input Module")

st.write("Upload your text files or paste text below to begin analysis.")

uploaded_file = st.file_uploader(
    "Upload a file (.txt, .csv, .docx, .pdf)", 
    type=["txt", "csv", "docx", "pdf"]
)

text_input = st.text_area("Or paste text directly here:")


if uploaded_file is not None:
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".txt":
            content = read_txt(uploaded_file)
        elif file_extension == ".csv":
            content = read_csv(uploaded_file)
        elif file_extension == ".docx":
            content = read_docx(uploaded_file)
        elif file_extension == ".pdf":
            content = read_pdf(uploaded_file)
        else:
            st.error(" Unsupported file type!")
            content = None

        if content:
            record = create_record(uploaded_file.name, "file", file_extension, content)
            save_data(record)
            st.success(f" File uploaded & stored in data_store.json (ID: {record['id']})")
            st.subheader("Extracted Text (Preview):")
            st.write(content[:1000] + "..." if len(content) > 1000 else content)

    except Exception as e:
        st.error(f" Error processing file: {e}")

elif text_input.strip() != "":
    record = create_record("pasted_text", "pasted", "raw", text_input)
    save_data(record)
    st.success(f" Text input saved in data_store.json (ID: {record['id']})")
    st.subheader("Input Text (Preview):")
    st.write(text_input)

else:
    st.info("Please upload a file or paste text to continue.")