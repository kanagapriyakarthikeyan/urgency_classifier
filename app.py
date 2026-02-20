import numpy as np
import tensorflow as tf
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware

# -----------------------
# Load model + tokenizer
# -----------------------

model = tf.keras.models.load_model("urgency_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 25

label_map = {
    0: "Self-Care",
    1: "Doctor Visit",
    2: "Emergency"
}

# -----------------------
# FastAPI setup
# -----------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Medical Urgency API is running"}

@app.post("/predict")
def predict(data: TextInput):

    if data.text.strip() == "":
        return {"error": "Please enter symptoms"}

    seq = tokenizer.texts_to_sequences([data.text.lower()])
    pad = pad_sequences(seq, maxlen=max_length, padding="post")

    prediction = model.predict(pad)[0]
    idx = int(np.argmax(prediction))
    confidence = float(prediction[idx])

    return {
        "input": data.text,
        "urgency_level": label_map[idx],
        "confidence": round(confidence, 4)
    }