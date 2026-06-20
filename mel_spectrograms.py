#generates CNN's inputs
import os
import numpy as np
import librosa
import pandas as pd

CLASSES      = ["Sabrina", "Taylor"]
DATA_DIR     = "data"
OUTPUT_DIR   = "spectrograms"
SR           = 22050
FIXED_FRAMES = 1290 

os.makedirs(OUTPUT_DIR, exist_ok=True)
labels = []

for language in CLASSES:
    clip_dir = os.path.join(DATA_DIR, language, "clips")
    save_dir = os.path.join(OUTPUT_DIR, language)
    os.makedirs(save_dir, exist_ok=True)
    
    for fname in os.listdir(clip_dir):
        if not fname.endswith(".wav"):
            continue
        
        filepath = os.path.join(clip_dir, fname)
        
        y, sr = librosa.load(filepath, sr=SR)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        if mel_db.shape[1] < FIXED_FRAMES:
            pad_width = FIXED_FRAMES - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        else:
            mel_db = mel_db[:, :FIXED_FRAMES]
        
        # fix 2a: actually save the spectrogram
        save_path = os.path.join(save_dir, fname.replace(".wav", ".npy"))
        np.save(save_path, mel_db)
        
        # fix 2b: actually record the label
        labels.append({"filename": fname.replace(".wav", ".npy"), "language": language})

# fix 1: moved outside BOTH loops — runs once, after everything is processed
df = pd.DataFrame(labels)
df.to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)
print(f"Saved {len(df)} labels to labels.csv")
        
        
        
        
        
        
        
        
        
        
        
        
        






