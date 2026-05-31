import torch
import streamlit as st
from transformers import PhobertTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "nghuxung/phobert-fakenews-v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    tokenizer = PhobertTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


def predict_phobert(text):
    if text is None or text.strip() == "":
        return None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    real_prob = probs[0].item()
    fake_prob = probs[1].item()

    prediction = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(real_prob, fake_prob)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_prob": real_prob,
        "fake_prob": fake_prob
    }