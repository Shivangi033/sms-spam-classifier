# SMS Spam Classifier

A simple Flask application that classifies SMS messages as spam or ham (not spam) using a machine learning model.

## Project Structure

- `app.py` - Flask web app that loads pre-trained `model.pkl` and `vectorizer.pkl` to make predictions.
- `main.py` - Training script using `TfidfVectorizer` and `MultinomialNB`, then saves the model artifacts.
- `save_model.py` - Alternative training script using `LogisticRegression` and a TF-IDF vectorizer.
- `data/sms.tsv` - Dataset containing labeled SMS messages.
- `templates/index.html` - Web UI template for submitting SMS text.
- `static/style.css` - Style sheet for the web UI.

## Requirements

- Python 3.8+
- Flask
- pandas
- numpy
- scikit-learn

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
```

2. Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install flask pandas numpy scikit-learn
```

## Training the Model

Before running the web app, train the model and save the artifacts.

```bash
python main.py
```

This will generate:

- `model.pkl`
- `vectorizer.pkl`

> If you want to use the alternative logistic regression version, run `python save_model.py` instead.

## Running the App

Start the Flask server:

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000/
```

## Usage

- Enter an SMS message into the text area.
- Click `Predict`.
- The app will display either `SPAM 🚨` or `HAM ✅`.

## Screenshots

The app UI should look like the images below after you add them to the `screenshots/` folder:

![Ham Prediction](screenshots/ham_prediction.png)

![Spam Prediction](screenshots/spam_prediction.png)

## Notes

- The text is cleaned before prediction by lowercasing, removing URLs, non-letter characters, punctuation, and extra whitespace.
- The `main.py` script splits the dataset into training and test sets and uses `TfidfVectorizer` with n-grams.

## License

This project is provided as-is for learning and experimentation.
