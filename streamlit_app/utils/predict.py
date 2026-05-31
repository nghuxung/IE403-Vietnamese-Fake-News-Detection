import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "nghuxung/phobert-fakenews-v2"

id2label = {
    0: "REAL",
    1: "FAKE"
}

@st.cache_resource
def load_phobert():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def predict_news(text, model_choice="PhoBERT"):
    tokenizer, model = load_phobert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_id = int(torch.argmax(probs).item())

    return {
        "label": id2label[pred_id],
        "confidence": probs[pred_id].item() * 100,
        "real_prob": probs[0].item() * 100,
        "fake_prob": probs[1].item() * 100
    }