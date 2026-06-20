"""
app.py
Streamlit demo: upload a song, get a Sabrina Carpenter vs Taylor Swift prediction.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tempfile
import os

CLASSES = ["Sabrina", "Taylor"]
SR = 22050
FIXED_FRAMES = 1290
MODEL_PATH = "models/raga_cnn.pth"


# ── Same CNN architecture as training — must match exactly ──
class RagaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 161, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


@st.cache_resource
def load_model():
    """Loaded once and cached — not reloaded on every interaction."""
    model = RagaCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def audio_to_spectrogram(filepath):
    """Same preprocessing pipeline as 3_mel_spectrograms.py + training normalization."""
    y, sr = librosa.load(filepath, sr=SR, duration=30)  # use first 30s

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if mel_db.shape[1] < FIXED_FRAMES:
        pad_width = FIXED_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]

    return mel_db


def predict(model, mel_db):
    spec = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(spec_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = probs.argmax().item()
    return CLASSES[pred_idx], probs.numpy()


# ── UI ──
st.set_page_config(page_title="Sabrina vs Taylor Classifier", page_icon="🎵")

st.title("🎵 Sabrina Carpenter vs Taylor Swift")
st.write(
    "Upload a song clip and a CNN trained on mel spectrograms will guess "
    "which artist it's by — based purely on how it *sounds*, not lyrics or metadata."
)

uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

if uploaded_file is not None:
    # save to a temp file so librosa can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file)

    with st.spinner("Analyzing audio..."):
        model = load_model()
        mel_db = audio_to_spectrogram(tmp_path)
        prediction, probs = predict(model, mel_db)

    os.remove(tmp_path)

    # ── Result ──
    st.subheader(f"Prediction: **{prediction}**")

    col1, col2 = st.columns(2)
    col1.metric(CLASSES[0], f"{probs[0]*100:.1f}%")
    col2.metric(CLASSES[1], f"{probs[1]*100:.1f}%")

    # ── Spectrogram visualization ──
    st.subheader("Mel Spectrogram")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel frequency bins")
    st.pyplot(fig)

    st.caption(
        "This model only sees the audio waveform — no lyrics, no metadata. "
        "Trained from scratch on a small custom dataset, so treat predictions as a fun demo, not gospel."
    )
else:
    st.info("Upload a song above to get a prediction.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    