import pandas as pd
import string
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("data/sms.tsv", sep="\t", names=["label", "message"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["cleaned"] = df["message"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

x_train, x_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label_num"], test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=5000)
x_train_vec = tfidf.fit_transform(x_train)

model = LogisticRegression()
model.fit(x_train_vec, y_train)

# save model + vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("Model and vectorizer saved!")
