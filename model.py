"""Phishing Email Detection - machine learning backend.
TF-IDF (text) + hand-made URL/keyword features -> Logistic Regression."""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# ---------- word lists used for feature extraction ----------
SUSPICIOUS = ["password", "verify", "account", "login", "confirm", "bank", "credit card",
              "ssn", "suspended", "invoice", "payment", "prize", "winner", "free", "gift",
              "claim", "update", "billing", "security", "click here", "limited", "unusual"]
URGENT = ["urgent", "immediately", "now", "today", "24 hours", "final warning", "act now",
          "expires", "expire", "asap", "failure to", "blocked", "closure", "penalty", "warning"]
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly"]
BAD_TLDS = [".xyz", ".top", ".ru", ".info", ".click", ".tk", ".work", ".live", ".co"]
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
IP_RE = re.compile(r"^(?:https?://)?\d{1,3}(?:\.\d{1,3}){3}")

NUM_COLS = ["n_urls", "n_suspicious_urls", "n_susp_words", "n_urgent_words",
            "has_ip_url", "has_at_symbol", "has_shortener", "has_http_only",
            "sender_suspicious", "exclamations", "caps_ratio"]


# ---------- preprocessing ----------
def clean_text(text):
    """Lowercase, replace URLs by a token, remove punctuation/numbers."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_urls(body, urls=""):
    """Collect URLs from the body and the optional URL field (text only, never visited)."""
    found = URL_RE.findall(str(body)) + URL_RE.findall(str(urls))
    return list(dict.fromkeys(u.rstrip(".,;)") for u in found))


def analyse_url(url):
    """Return a list of suspicious patterns found in one URL string."""
    u, flags = url.lower(), []
    host = re.sub(r"^https?://", "", u).split("/")[0]
    if IP_RE.match(u): flags.append("Uses IP address instead of domain")
    if "@" in u: flags.append("Contains '@' symbol")
    if any(s in host for s in SHORTENERS): flags.append("URL shortener")
    if u.startswith("http://"): flags.append("Not secure (http, no HTTPS)")
    if len(u) > 75: flags.append("Very long URL")
    if host.count(".") >= 3: flags.append("Many subdomains")
    if host.count("-") >= 2: flags.append("Many hyphens in domain")
    if any(host.endswith(t) for t in BAD_TLDS): flags.append("Suspicious domain ending")
    if any(w in host for w in ["login", "verify", "secure", "account", "update"]):
        flags.append("Phishing word in domain")
    return flags


def analyse_sender(sender):
    s = str(sender).lower().strip()
    if not s or s == "nan" or "@" not in s: return []
    domain, flags = s.split("@")[-1], []
    if any(domain.endswith(t) for t in BAD_TLDS): flags.append("Suspicious sender domain ending")
    if domain.count("-") >= 1 and any(w in domain for w in ["secure", "verify", "alert", "update", "login"]):
        flags.append("Sender domain imitates a security service")
    if re.search(r"\d{3,}", s.split("@")[0]): flags.append("Many digits in sender name")
    return flags


def keyword_hits(text, words):
    t = str(text).lower()
    return [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", t)]


def extract_features(subject="", sender="", body="", urls=""):
    """Turn one email into a dictionary of numeric features + readable indicators."""
    full = f"{subject} {body}"
    url_list = find_urls(full, urls)
    url_flags = {u: analyse_url(u) for u in url_list}
    all_flags = [f for fl in url_flags.values() for f in fl]
    sus, urg = keyword_hits(full, SUSPICIOUS), keyword_hits(full, URGENT)
    letters = [c for c in full if c.isalpha()]
    feats = {
        "n_urls": len(url_list),
        "n_suspicious_urls": sum(1 for fl in url_flags.values() if len(fl) >= 2),
        "n_susp_words": len(sus), "n_urgent_words": len(urg),
        "has_ip_url": int("Uses IP address instead of domain" in all_flags),
        "has_at_symbol": int("Contains '@' symbol" in all_flags),
        "has_shortener": int("URL shortener" in all_flags),
        "has_http_only": int("Not secure (http, no HTTPS)" in all_flags),
        "sender_suspicious": int(len(analyse_sender(sender)) > 0),
        "exclamations": full.count("!"),
        "caps_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
    }
    info = {"suspicious_words": sus, "urgent_words": urg, "urls": url_flags,
            "sender_flags": analyse_sender(sender)}
    return feats, info


def build_features(df):
    """Create the model input table (cleaned text + numeric features) from a dataframe."""
    rows = [extract_features(r.subject, r.sender, r.body, r.urls)[0] for r in df.itertuples()]
    X = pd.DataFrame(rows, columns=NUM_COLS)
    X.insert(0, "text", (df["subject"] + " " + df["body"]).map(clean_text).values)
    return X


# ---------- dataset loading ----------
POS = {"phishing", "phish", "spam", "1", "phishing email", "malicious", "fraud"}
NEG = {"safe", "legitimate", "ham", "0", "safe email", "legit", "benign", "not phishing"}


def load_dataset(source):
    """Read CSV (path or uploaded file), clean it and map labels to 1=Phishing, 0=Safe."""
    df = pd.read_csv(source)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    renames = {"email_text": "body", "text": "body", "message": "body", "content": "body",
               "email_type": "label", "class": "label", "type": "label", "target": "label"}
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})
    if "body" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain an email text column ('body' or 'text') and a 'label' column.")
    for col in ["subject", "sender", "urls"]:
        if col not in df.columns: df[col] = ""
    df = df[["subject", "sender", "body", "urls", "label"]]
    df = df.dropna(subset=["body", "label"])                      # missing text/label: drop row
    df[["subject", "sender", "urls"]] = df[["subject", "sender", "urls"]].fillna("")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    unknown = set(df["label"]) - POS - NEG
    if unknown:
        raise ValueError(f"Unrecognised labels: {sorted(unknown)[:5]}. Use Phishing/Safe (or 1/0, spam/ham).")
    df["is_phishing"] = df["label"].isin(POS).astype(int)
    df = df.drop(columns="label").drop_duplicates().reset_index(drop=True)  # remove duplicates
    if df["is_phishing"].nunique() < 2 or df["is_phishing"].value_counts().min() < 5:
        raise ValueError("Dataset needs at least 5 phishing and 5 legitimate emails.")
    return df


# ---------- training & evaluation ----------
def make_pipeline(clf):
    pre = ColumnTransformer([("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                                       min_df=2, max_features=5000), "text"),
                             ("num", MinMaxScaler(), NUM_COLS)])
    return Pipeline([("features", pre), ("clf", clf)])


def metrics_of(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def train_model(df, test_size=0.2, seed=42):
    """Split first, then fit ONLY on the training part (no data leakage)."""
    X, y = build_features(df), df["is_phishing"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    model = make_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    nb = make_pipeline(MultinomialNB()).fit(X_tr, y_tr)        # optional comparison
    # 5-fold cross-validation: the pipeline is re-fitted inside every fold, so there is no leakage
    cv = cross_val_score(make_pipeline(LogisticRegression(max_iter=1000)), X, y, scoring="f1",
                         cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    names = model.named_steps["features"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    clean = lambda n: n.split("__", 1)[1]
    results = {
        "metrics": metrics_of(y_te, pred),
        "nb_metrics": metrics_of(y_te, nb.predict(X_te)),
        "confusion": confusion_matrix(y_te, pred, labels=[0, 1]),
        "report": classification_report(y_te, pred, target_names=["Safe", "Phishing"], output_dict=True, zero_division=0),
        "cv_f1": cv.tolist(),
        "n_train": len(X_tr), "n_test": len(X_te),
        "top_phishing": [(clean(names[i]), float(coefs[i])) for i in order[::-1][:10]],
        "top_safe": [(clean(names[i]), float(coefs[i])) for i in order[:10]],
    }
    return model, results


# ---------- prediction ----------
def predict_email(model, subject="", sender="", body="", urls=""):
    """Classify one email. Returns label, phishing probability and indicators."""
    feats, info = extract_features(subject, sender, body, urls)
    row = pd.DataFrame([feats], columns=NUM_COLS)
    row.insert(0, "text", clean_text(f"{subject} {body}"))
    p = float(model.predict_proba(row)[0][1])
    return {"label": "Phishing" if p >= 0.5 else "Safe", "phishing_prob": p, "features": feats, "info": info}


def explain(out):
    """Short plain-English reasons for the result (used by the app)."""
    i, f, reasons = out["info"], out["features"], []
    if i["urgent_words"]: reasons.append(f"Urgent/threatening words: {', '.join(i['urgent_words'][:5])}")
    if i["suspicious_words"]: reasons.append(f"Suspicious keywords: {', '.join(i['suspicious_words'][:5])}")
    for url, flags in i["urls"].items():
        if flags: reasons.append(f"URL {url[:45]} -> {', '.join(flags[:3])}")
    if i["sender_flags"]: reasons.append("Sender: " + ", ".join(i["sender_flags"]))
    if f["exclamations"] >= 3: reasons.append("Many exclamation marks")
    return reasons or ["No strong warning signs found in the text, URLs or sender"]
