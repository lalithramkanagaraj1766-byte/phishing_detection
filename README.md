# Phishing Email Detection Model

A machine learning screening tool that classifies an email as **Phishing** or **Safe** using
TF-IDF text features, URL/keyword features and Logistic Regression. Built with Python, Streamlit,
Scikit-learn, Pandas, NumPy, Plotly and regular expressions.

## Run in VS Code
1. Install Python 3.9+ and open the `phishing_detection` folder in VS Code (File > Open Folder).
2. Open the terminal (Ctrl + `) and create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py` (opens at http://localhost:8501)

## Dataset
`data/dataset.csv` is a **small SYNTHETIC demo dataset** (about 880 computer-generated emails including some deliberately tricky cases, not real-world data).
Its model scores are unrealistically high. For realistic results, upload a real dataset
(for example a public phishing/spam email dataset from Kaggle) using the sidebar uploader.

Expected CSV format (only `body` and `label` are required):

| subject | sender | body | urls | label |
|---|---|---|---|---|
| Verify your account | a@x.xyz | Click http://... | http://... | Phishing |

Accepted labels: Phishing/Safe, spam/ham, 1/0, Legitimate. Text column may also be named `text`, `email_text` or `message`.

## How it works (for your review)
1. **Preprocessing** (`model.py > load_dataset, clean_text`): drop missing rows, remove duplicates, lowercase,
   replace URLs with a token, remove punctuation, map labels to 1 (phishing) / 0 (safe).
2. **Features** (`extract_features`): TF-IDF on the text, plus counts of suspicious keywords, urgent words,
   URLs, and URL patterns (IP address, `@`, shortener, http, long URL, many subdomains, odd domain endings).
   URLs are analysed as text only and never opened.
3. **Split**: 80% training / 20% testing (stratified). Scaling and TF-IDF are fitted on training data only (no leakage).
4. **Model**: Logistic Regression (main). Multinomial Naive Bayes is shown only as a comparison.
5. **Evaluation**: also a 5-fold cross-validation F1 score; accuracy, precision, recall, F1, confusion matrix and classification report, all computed from the test set.

## Demo tip
On the Email Detection page, use the **Try phishing example** / **Try safe example** buttons to fill the form quickly.

## Files
- `app.py`: Streamlit interface (4 pages)
- `model.py`: preprocessing, features, training, prediction
- `test_model.py`: quick checks (run `python test_model.py`)
- `data/dataset.csv`: demo dataset

**Disclaimer:** This is a screening tool. It cannot guarantee that an email is safe.
