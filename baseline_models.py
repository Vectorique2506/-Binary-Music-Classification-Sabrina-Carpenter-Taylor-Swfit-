import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler









# 1. Load the features CSV
df = pd.read_csv("features/features.csv")

# 2. Extract song_id by stripping off the "_000.wav" / "_001.wav" clip suffix
#    "Thangamey_002.wav" -> "Thangamey"
df["song_id"] = df["filename"].str.rsplit("_", n=1).str[0]

# 3. Get unique songs, split THOSE into train/test (not individual clips)
unique_songs = df["song_id"].unique()
train_songs, test_songs = train_test_split(
    unique_songs, test_size=0.2, random_state=42
)

# 4. Assign every row to train/test based on which group its song fell into
train_df = df[df["song_id"].isin(train_songs)]
test_df  = df[df["song_id"].isin(test_songs)]

print(f"Train: {len(train_df)} clips from {len(train_songs)} songs")
print(f"Test:  {len(test_df)} clips from {len(test_songs)} songs")

print(df.groupby("language")["song_id"].nunique())
print(test_df["language"].value_counts())




# 5. Separate X (features) and y (label) for each split
feature_cols = [c for c in df.columns if c not in ["filename", "language", "song_id"]]

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)



le = LabelEncoder()
le.fit(df["language"])   # fit on full set so train/test share the same encoding
y_train = le.transform(train_df["language"])
y_test  = le.transform(test_df["language"])

print(dict(zip(le.classes_, le.transform(le.classes_))))

# 6. Train Logistic Regression
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train_scaled, y_train)
log_reg_preds = log_reg.predict(X_test_scaled)
print("Logistic Regression accuracy:", accuracy_score(y_test, log_reg_preds))

# 7. Train SVM
svm = SVC(kernel="rbf")
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
print("Support Vector Machine accuracy:", accuracy_score(y_test, svm_preds))

# 8. Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
nb_preds = nb.predict(X_test_scaled)
print("Naive Bayes accuracy:", accuracy_score(y_test, nb_preds))
















