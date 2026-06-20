import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

CLASSES = ["Sabrina", "Taylor"]
SPEC_DIR = "spectrograms"

labels_df = pd.read_csv(os.path.join(SPEC_DIR, "labels.csv"))

labels_df["song_id"] = labels_df["filename"].str.rsplit("_", n=1).str[0]
unique_songs = labels_df["song_id"].unique()
train_songs, test_songs = train_test_split(unique_songs, test_size=0.2, random_state=42)

train_df = labels_df[labels_df["song_id"].isin(train_songs)].reset_index(drop=True)
test_df  = labels_df[labels_df["song_id"].isin(test_songs)].reset_index(drop=True)

print(f"Train: {len(train_df)} clips, Test: {len(test_df)} clips")


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

        label = 0 if language == "Sabrina" else 1
        label = torch.tensor(label, dtype=torch.long)

        return spec, label


train_dataset = SpectrogramDataset(train_df, SPEC_DIR)
test_dataset  = SpectrogramDataset(test_df, SPEC_DIR)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")


class RagaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

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


# ── Sanity check shapes ──
model_check = RagaCNN()
sample_batch, sample_labels = next(iter(train_loader))
print("Input shape:", sample_batch.shape)
output_check = model_check(sample_batch)
print("Output shape:", output_check.shape)


# ── Training setup ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = RagaCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 15

# ── Training loop ──
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for specs, labels in train_loader:
        specs, labels = specs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # fix: moved outside the batch loop — now runs once per epoch, after
    # total_loss / correct / total have all been fully accumulated
    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f}")


# ── Evaluation on test set ──
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for specs, labels in test_loader:
        specs, labels = specs.to(device), labels.to(device)
        outputs = model(specs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), "models/raga_cnn.pth")
print("Model saved to models/raga_cnn.pth")

































