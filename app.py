"""Phishing Email Detection - Streamlit dashboard. Run: streamlit run app.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from model import load_dataset, train_model, predict_email, explain

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="wide")

NAVY, CARD, CYAN, BLUE, RED, GREEN = "#0a1128", "#121c3d", "#22d3ee", "#3b82f6", "#ef4444", "#22c55e"
st.markdown(f"""
<style>
.stApp {{background:{NAVY}; color:#e2e8f0;}}
[data-testid="stSidebar"] {{background:#070d1f; border-right:1px solid #1e2a52;}}
h1,h2,h3 {{color:#f1f5f9; font-family:'Segoe UI',sans-serif;}}
.kpi {{background:{CARD}; border:1px solid #1e2a52; border-left:4px solid {CYAN}; border-radius:10px; padding:16px 20px;}}
.kpi .v {{font-size:30px; font-weight:700; color:{CYAN};}} .kpi .l {{font-size:13px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;}}
.result {{border-radius:12px; padding:22px; text-align:center; font-size:28px; font-weight:800; margin:10px 0;}}
.bad {{background:rgba(239,68,68,.15); border:2px solid {RED}; color:{RED};}}
.good {{background:rgba(34,197,94,.15); border:2px solid {GREEN}; color:{GREEN};}}
.tag {{display:inline-block; background:#1e2a52; border:1px solid {BLUE}; color:#cfe3ff; border-radius:14px; padding:2px 10px; margin:2px; font-size:13px;}}
.tag.r {{border-color:{RED}; color:#fecaca;}}
</style>""", unsafe_allow_html=True)


def kpi(label, value):
    st.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div></div>', unsafe_allow_html=True)


def style(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                      margin=dict(t=40, b=20, l=20, r=20))
    return fig


@st.cache_resource(show_spinner="Training model...")
def get_model(df):
    return train_model(df)


# ---------- sidebar: navigation + dataset ----------
st.sidebar.title("🛡️ PhishGuard ML")
page = st.sidebar.radio("Navigation", ["Dashboard Overview", "Email Detection", "Feature Analysis", "Model Performance"])
st.sidebar.markdown("---")
upload = st.sidebar.file_uploader("Upload your own dataset (CSV)", type="csv")
st.sidebar.caption("Columns: `subject`, `sender`, `body`, `urls`, `label` (Phishing/Safe). "
                   "Only `body` and `label` are required.")
try:
    if upload is not None:
        df, source = load_dataset(upload), f"Uploaded: {upload.name}"
    else:
        df, source = load_dataset("data/dataset.csv"), "Built-in SYNTHETIC demo dataset"
    model, res = get_model(df)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()
st.sidebar.info(f"Dataset: {source}")

# ---------- A. dashboard ----------
if page == "Dashboard Overview":
    st.title("Dashboard Overview")
    n_ph, n_ok = int(df.is_phishing.sum()), int((df.is_phishing == 0).sum())
    c = st.columns(4)
    with c[0]: kpi("Total emails", len(df))
    with c[1]: kpi("Phishing", n_ph)
    with c[2]: kpi("Legitimate", n_ok)
    with c[3]: kpi("Model accuracy", f"{res['metrics']['accuracy']:.1%}")
    st.write("")
    left, right = st.columns(2)
    with left:
        fig = px.pie(names=["Phishing", "Safe"], values=[n_ph, n_ok], hole=.5, title="Dataset distribution",
                     color=["Phishing", "Safe"], color_discrete_map={"Phishing": RED, "Safe": GREEN})
        st.plotly_chart(style(fig), use_container_width=True)
    with right:
        st.subheader("How it works")
        st.markdown("1. Clean text and remove duplicates\n2. Split into **train / test** sets\n"
                    "3. TF-IDF + URL/keyword features\n4. Logistic Regression learns from the training set\n"
                    "5. Evaluate on unseen test emails")
        st.caption(f"Train: {res['n_train']} emails | Test: {res['n_test']} emails")
    if upload is None:
        st.warning("You are using a small synthetic demo dataset (computer-generated, not real emails). "
                   "Scores on it are very high because the data is simple. Upload a real dataset for realistic results.")
    with st.expander("Preview dataset"):
        st.dataframe(df.head(20), use_container_width=True)

# ---------- B. detection ----------
elif page == "Email Detection":
    st.title("Email Detection")
    SAMPLES = {"Try phishing example": ("Urgent: Your PayPal account has been suspended", "security@paypal-alerts.xyz",
               "Dear customer, we detected unusual activity. Verify your password immediately or your account will be blocked within 24 hours. Login now: http://paypal-secure-login.xyz/verify", ""),
               "Try safe example": ("Meeting notes for Monday", "priya@college.edu",
               "Hi team, please find the notes from today's discussion. Let me know if I missed anything. Thanks, Priya", "")}
    def fill(vals):
        for k, v in zip(["subj", "send", "body", "urls"], vals): st.session_state[k] = v
    b1, b2, _ = st.columns([1, 1, 3])
    for col, (name, vals) in zip([b1, b2], SAMPLES.items()):
        col.button(name, on_click=fill, args=(vals,))
    subject = st.text_input("Email subject", key="subj")
    sender = st.text_input("Sender email address (optional)", key="send")
    body = st.text_area("Email body", height=180, key="body")
    urls = st.text_input("URLs (optional, separate with spaces)", key="urls")
    if st.button("🔍 Analyse Email", type="primary"):
        if not body.strip() and not subject.strip():
            st.warning("Please enter an email subject or body first.")
        else:
            st.session_state["last"] = predict_email(model, subject, sender, body, urls)
    out = st.session_state.get("last")
    if out:
        p = out["phishing_prob"]
        if out["label"] == "Phishing":
            st.markdown(f'<div class="result bad">⚠️ PHISHING &nbsp; ({p:.1%} probability)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result good">✅ SAFE &nbsp; ({1 - p:.1%} confidence)</div>', unsafe_allow_html=True)
        st.progress(min(max(p, 0.0), 1.0), text=f"Phishing probability: {p:.1%}")
        info = out["info"]
        n_flags = len(info["suspicious_words"]) + len(info["urgent_words"]) + sum(len(v) for v in info["urls"].values())
        st.markdown("**Why this result?**\n" + "\n".join(f"- {r}" for r in explain(out)))
        st.write(f"Found **{n_flags}** indicators. Open **Feature Analysis** in the sidebar for details.")
    st.caption("⚠️ This is a machine learning screening tool. It cannot guarantee that an email is safe. "
               "Never click unknown links or share passwords, even if the result says Safe.")

# ---------- C. feature analysis ----------
elif page == "Feature Analysis":
    st.title("Feature Analysis")
    out = st.session_state.get("last")
    if not out:
        st.info("Analyse an email on the **Email Detection** page first.")
    else:
        info, f = out["info"], out["features"]
        c = st.columns(4)
        with c[0]: kpi("URLs found", f["n_urls"])
        with c[1]: kpi("Suspicious keywords", f["n_susp_words"])
        with c[2]: kpi("Urgent words", f["n_urgent_words"])
        with c[3]: kpi("Suspicious URLs", f["n_suspicious_urls"])
        st.write("")
        tags = lambda items, cls="": "".join(f'<span class="tag {cls}">{i}</span>' for i in items) or "None found"
        st.subheader("Suspicious keywords"); st.markdown(tags(info["suspicious_words"]), unsafe_allow_html=True)
        st.subheader("Urgent / threatening words"); st.markdown(tags(info["urgent_words"], "r"), unsafe_allow_html=True)
        st.subheader("URL indicators (text analysis only, links are never opened)")
        if info["urls"]:
            for url, flags in info["urls"].items():
                st.code(url, language=None)
                st.markdown(tags(flags, "r") if flags else "No suspicious pattern", unsafe_allow_html=True)
        else:
            st.write("No URLs found")
        st.subheader("Sender indicators"); st.markdown(tags(info["sender_flags"], "r"), unsafe_allow_html=True)
        st.subheader("Numeric features given to the model")
        st.dataframe(pd.DataFrame([f]).T.rename(columns={0: "value"}), use_container_width=True)
    st.subheader("Words the model links most strongly to each class")
    a, b = st.columns(2)
    for col, key, title, color in [(a, "top_phishing", "Phishing", RED), (b, "top_safe", "Safe", GREEN)]:
        d = pd.DataFrame(res[key], columns=["feature", "weight"])
        fig = px.bar(d, x="weight", y="feature", orientation="h", title=title, color_discrete_sequence=[color])
        fig.update_yaxes(autorange="reversed")
        col.plotly_chart(style(fig), use_container_width=True)

# ---------- D. model performance ----------
else:
    st.title("Model Performance")
    st.caption(f"Evaluated on {res['n_test']} unseen test emails (model trained on {res['n_train']}).")
    m = res["metrics"]
    c = st.columns(4)
    for col, k in zip(c, ["accuracy", "precision", "recall", "f1"]):
        with col: kpi(k.replace("f1", "F1-score").capitalize(), f"{m[k]:.2%}")
    st.write("")
    left, right = st.columns(2)
    with left:
        cm = res["confusion"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred Safe", "Pred Phishing"], y=["Actual Safe", "Actual Phishing"],
                                   text=cm, texttemplate="%{text}", colorscale=[[0, NAVY], [1, BLUE]], showscale=False))
        fig.update_layout(title="Confusion matrix"); fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig), use_container_width=True)
    with right:
        nb = res["nb_metrics"]
        d = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1"] * 2,
                          "Model": ["Logistic Regression"] * 4 + ["Naive Bayes (comparison)"] * 4,
                          "Score": [m[k] for k in ["accuracy", "precision", "recall", "f1"]] +
                                   [nb[k] for k in ["accuracy", "precision", "recall", "f1"]]})
        fig = px.bar(d, x="Metric", y="Score", color="Model", barmode="group", title="Metrics comparison",
                     color_discrete_sequence=[CYAN, BLUE])
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(style(fig), use_container_width=True)
    cv = pd.Series(res["cv_f1"])
    st.info(f"5-fold cross-validation F1-score: **{cv.mean():.2%}** (+/- {cv.std():.2%}). "
            "This checks that the result does not depend on one lucky train/test split.")
    st.subheader("Classification report")
    rep = pd.DataFrame(res["report"]).T.round(3)
    st.dataframe(rep, use_container_width=True)
    st.markdown("**Precision**: of emails flagged phishing, how many really were. "
                "**Recall**: of all real phishing emails, how many were caught. "
                "**F1**: balance of the two.")
    if upload is None:
        st.warning("Synthetic demo data: near-perfect scores do not reflect real-world performance.")
