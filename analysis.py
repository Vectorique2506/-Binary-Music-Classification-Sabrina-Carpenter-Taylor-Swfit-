"""
8_analysis.py
Loads the trained CNN, evaluates on the test set, and produces:
1. A confusion matrix
2. Side-by-side spectrogram comparison (correct class 0, correct class 1, misclassified)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

CLASSES  = ["Sabrina", "Taylor"]
SPEC_DIR = "spectrograms"
MODEL_PATH = "models/raga_cnn.pth"

os.makedirs("analysis", exist_ok=True)


class SpectrogramDataset(Dataset):
    def __init__(self, dataframe, spec_dir):
        self.df = dataframe
        self.spec_dir = spec_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        language = row["language"]
        filepath = os.path.join(self.spec_dir, language, row["filename"])

        spec = np.load(filepath)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = torch.tensor(spec, dtype=torch.float32)
        spec = spec.unsqueeze(0)

        label = 0 if language == CLASSES[0] else 1
        label = torch.tensor(label, dtype=torch.long)

        return spec, label, row["filename"], language


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


labels_df = pd.read_csv(os.path.join(SPEC_DIR, "labels.csv"))
labels_df["song_id"] = labels_df["filename"].str.rsplit("_", n=1).str[0]
unique_songs = labels_df["song_id"].unique()
train_songs, test_songs = train_test_split(unique_songs, test_size=0.2, random_state=42)
test_df = labels_df[labels_df["song_id"].isin(test_songs)].reset_index(drop=True)

test_dataset = SpectrogramDataset(test_df, SPEC_DIR)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"Evaluating on {len(test_df)} test clips...")

device = torch.device("cpu")
model = RagaCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []
all_filenames = []
all_languages = []
all_specs = []

with torch.no_grad():
    for specs, labels, filenames, languages in test_loader:
        specs_device = specs.to(device)
        outputs = model(specs_device)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_filenames.extend(filenames)
        all_languages.extend(languages)
        all_specs.extend(specs.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = (all_preds == all_labels).mean()
print(f"Test Accuracy: {test_acc:.4f}")


# 1. Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{CLASSES[0]} vs {CLASSES[1]} — Confusion Matrix (Acc: {test_acc:.1%})")
plt.tight_layout()
plt.savefig("analysis/confusion_matrix.png", dpi=150)
plt.close()
print("Saved analysis/confusion_matrix.png")


# 2. Results dataframe
results_df = pd.DataFrame({
    "filename": all_filenames,
    "true_label": all_languages,
    "predicted_label": [CLASSES[0] if p == 0 else CLASSES[1] for p in all_preds],
    "correct": all_preds == all_labels
})
results_df.to_csv("analysis/test_predictions.csv", index=False)
print("Saved analysis/test_predictions.csv")

misclassified = results_df[~results_df["correct"]]
print(f"\nMisclassified clips ({len(misclassified)} of {len(results_df)}):")
print(misclassified[["filename", "true_label", "predicted_label"]].to_string(index=False))


# 3. Side-by-side spectrogram comparison
def find_example(condition_mask):
    idxs = np.where(condition_mask)[0]
    return idxs[0] if len(idxs) > 0 else None

correct_class0_idx = find_example((all_labels == 0) & (all_preds == 0))
correct_class1_idx = find_example((all_labels == 1) & (all_preds == 1))
misclassified_idx  = find_example(all_preds != all_labels)

examples = [
    (f"Correct: {CLASSES[0]}", correct_class0_idx),
    (f"Correct: {CLASSES[1]}", correct_class1_idx),
    ("Misclassified", misclassified_idx),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (title, idx) in zip(axes, examples):
    if idx is None:
        ax.set_title(f"{title}\n(none found)")
        ax.axis("off")
        continue

    spec = all_specs[idx][0]
    fname = all_filenames[idx]
    true_lang = all_languages[idx]
    pred_lang = CLASSES[0] if all_preds[idx] == 0 else CLASSES[1]

    ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"{title}\n{fname}\nTrue: {true_lang} | Pred: {pred_lang}", fontsize=9)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel frequency bins")

plt.tight_layout()
plt.savefig("analysis/spectrogram_comparison.png", dpi=150)
plt.close()
print("Saved analysis/spectrogram_comparison.png")

print("\n✓ Analysis complete. Check the analysis/ folder for:")
print("  - confusion_matrix.png")
print("  - spectrogram_comparison.png")
print("  - test_predictions.csv")




