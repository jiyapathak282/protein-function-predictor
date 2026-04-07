import streamlit as st
import joblib
import numpy as np

# Load model and amino acids
model = joblib.load("protein_function_model.pkl")
amino_acids = joblib.load("amino_acids.pkl")

st.set_page_config(page_title="Protein Function Predictor", layout="centered")

st.title("🧬 Protein Function Prediction App")
st.write("Paste a protein sequence below to predict its functional class.")

# Input box
sequence = st.text_area("Protein Sequence", height=200)

def extract_features(sequence):
    sequence = sequence.upper()
    return np.array([sequence.count(aa) / len(sequence) for aa in amino_acids]).reshape(1, -1)

if st.button("Predict Function"):
    if sequence.strip() == "":
        st.warning("Please enter a protein sequence.")
    else:
        features = extract_features(sequence)
        prediction = model.predict(features)
        st.success(f"Predicted Function: **{prediction[0]}**")