import pandas as pd
import numpy as np
import string
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 

# ==========================
# 1. LOAD DATA
# ==========================
df = pd.read_csv("data/sms.tsv", sep="\t", names=["label", "message"])

# ==========================
# 2. CLEANING FUNCTION
# ==========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["cleaned"] = df["message"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# ==========================
# 3. SPLIT DATA
# ==========================
# Note: Using lowercase 'x' to stay consistent with your style
x_train, x_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label_num"], test_size=0.2, random_state=42
)

# ==========================
# 4. VECTORIZE TEXT
# ==========================
# We use ngram_range=(1,2) to catch phrases like "lucky draw"
tfidf = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))

# This creates the numerical data for the model
x_train_vec = tfidf.fit_transform(x_train)
x_test_vec = tfidf.transform(x_test)

# ==========================
# 5. TRAIN MODEL
# ==========================
# Using MultinomialNB for better text classification
model = MultinomialNB(alpha=0.1) 

# FIX: We use 'x_train_vec' which was created in the step above
model.fit(x_train_vec, y_train)

# ==========================
# 6. SAVE ARTIFACTS
# ==========================
# This is what app.py needs to run
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
print("✅ Model and Vectorizer saved successfully!")

# ==========================
# 7. PREDICTION FUNCTION (For testing)
# ==========================
def predict_message(message):
    cleaned = clean_text(message)
    msg_vec = tfidf.transform([cleaned])
    prediction = model.predict(msg_vec)[0]
    return "SPAM 🚨" if prediction == 1 else "HAM ✅"

# ==========================
# 8. TEST PREDICTIONS
# ==========================
print("\nTesting predictions:")
print("Test 1 (Spam):", predict_message("Congratulations! You won a lucky draw and cash prize"))
print("Test 2 (Ham):", predict_message("Hey, are we meeting tomorrow for coffee?"))