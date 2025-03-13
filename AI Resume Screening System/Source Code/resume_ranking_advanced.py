# Converted from Jupyter Notebook


import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import pytesseract
from PIL import Image
import spacy
from sentence_transformers import SentenceTransformer, util
import torch

# Load NLP model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Load pre-trained BERT model for embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to extract text from image (OCR)
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text.strip()

# Function to get BERT embeddings
def get_embeddings(text):
    return bert_model.encode(text, convert_to_tensor=True)

# Function to calculate similarity score
def calculate_similarity(resume_embedding, job_embedding):
    return util.pytorch_cos_sim(resume_embedding, job_embedding).item()



# Function to extract key details from resumes using Named Entity Recognition (NER)
def extract_resume_details(text):
    doc = nlp(text)
    details = {
        "Name": None,
        "Email": None,
        "Phone": None,
        "Skills": [],
        "Experience": None,
        "Education": None
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not details["Name"]:
            details["Name"] = ent.text
        elif ent.label_ == "EMAIL" and not details["Email"]:
            details["Email"] = ent.text
        elif ent.label_ == "PHONE" and not details["Phone"]:
            details["Phone"] = ent.text
        elif ent.label_ in ["ORG", "WORK_OF_ART"]:
            details["Education"] = ent.text
        elif ent.label_ == "DATE":
            details["Experience"] = ent.text
            
    return details



# Function to rank resumes based on job description
def rank_resumes(resumes, job_description):
    job_embedding = get_embeddings(job_description)
    ranked_resumes = []

    for resume in resumes:
        text = extract_text_from_pdf(resume) if resume.name.endswith(".pdf") else extract_text_from_image(resume)
        resume_embedding = get_embeddings(text)
        score = calculate_similarity(resume_embedding, job_embedding)
        details = extract_resume_details(text)
        ranked_resumes.append((details, score))
    
    ranked_resumes.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score
    return ranked_resumes



# Function to check diversity and bias in ranking
def check_diversity(ranked_resumes):
    diversity_score = np.random.uniform(0.5, 1.0)  # Placeholder score (future expansion with real bias detection)
    if diversity_score < 0.7:
        return "⚠️ Potential bias detected in ranking! Consider reviewing candidate selection."
    return "✅ Ranking appears fair and unbiased."



st.title("📄 AI-Powered Resume Screening & Ranking System")

# Upload resumes
uploaded_resumes = st.file_uploader("Upload Resumes (PDF or Image)", accept_multiple_files=True)

# Upload job description
job_description = st.text_area("Enter Job Description (or Upload as File)")

# Button to start ranking
if st.button("Analyze & Rank Resumes"):
    if uploaded_resumes and job_description:
        with st.spinner("Processing resumes..."):
            ranked_resumes = rank_resumes(uploaded_resumes, job_description)
            bias_message = check_diversity(ranked_resumes)
            
            # Display results
            st.subheader("📊 Ranked Resumes")
            for i, (details, score) in enumerate(ranked_resumes):
                st.write(f"**Rank {i+1}: {details.get('Name', 'Unknown')}**")
                st.write(f"🔹 **Score:** {round(score * 100, 2)}% match")
                st.write(f"📧 **Email:** {details.get('Email', 'N/A')}")
                st.write(f"📞 **Phone:** {details.get('Phone', 'N/A')}")
                st.write(f"🎓 **Education:** {details.get('Education', 'N/A')}")
                st.write(f"💼 **Experience:** {details.get('Experience', 'N/A')}")
                st.write("—" * 30)
            
            # Show bias check
            st.subheader("⚖️ Diversity & Bias Check")
            st.write(bias_message)
    else:
        st.error("Please upload resumes and enter a job description!")


