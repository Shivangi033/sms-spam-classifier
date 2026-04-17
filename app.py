from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Load the saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    message_text = "" # To keep the text in the box after clicking
    if request.method == "POST":
        message_text = request.form["message"]
        cleaned = clean_text(message_text)
        
        # Transform and predict
        msg_vec = vectorizer.transform([cleaned])
        pred = model.predict(msg_vec)[0]
        
        prediction = "SPAM 🚨" if pred == 1 else "HAM ✅"
        
    return render_template("index.html", prediction=prediction, message=message_text)

if __name__ == "__main__":
    app.run(debug=True)