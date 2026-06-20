import os
import numpy as np
import pandas as pd
import librosa

CLASSES    = ["Sabrina", "Taylor"]
DATA_DIR   = "data"
OUTPUT_CSV = "features/features.csv"   #Place where final features would be saved
SR         = 22050 #sample rate?
#SR means taking snapshots of the audio-waveform per second to get an idea of how amplitude is changing at that exact moment 
#WAV files are already samples so sr usually re-samples it at fixed rate of 22050!!!!!!!!!

os.makedirs("features", exist_ok=True)  # creating features folder if i doesnt exist already

rows = [] #empty list to collect data for every audio clip 

for language in CLASSES:
    clip_dir = os.path.join(DATA_DIR, language, "clips")
    
    for fname in os.listdir(clip_dir):
        if not fname.endswith(".wav"):
            continue
        
        filepath = os.path.join(clip_dir, fname)
        
        # fix 1: load the actual audio file, keep SR as the constant
        y, sr = librosa.load(filepath, sr=SR)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std  = np.std(mfccs, axis=1)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr, axis=1)
        zcr_std  = np.std(zcr, axis=1)
        
        # fix 2: spectral_centroid needs sr=
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid, axis=1)
        centroid_std  = np.std(centroid, axis=1)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std  = np.std(chroma, axis=1)
        
        final_features = np.concatenate((
            mfccs_mean, mfccs_std,
            chroma_mean, chroma_std,
            centroid_mean, centroid_std,
            zcr_mean, zcr_std
        ))
        
        # fix 3: spread the array into individual numbered columns
        row = {"filename": fname, "language": language}
        for i, val in enumerate(final_features):
            row[f"feature_{i}"] = val
        rows.append(row)

# fix 3 continued: actually save to CSV
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Saved {len(df)} rows to {OUTPUT_CSV}")
print(f"  Columns: {len(df.columns)} ({len(final_features)} features + filename + language)")




















