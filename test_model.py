"""Quick checks. Run: python test_model.py"""
from model import (load_dataset, clean_text, find_urls, analyse_url, extract_features,
                   train_model, predict_email)

df = load_dataset("data/dataset.csv")
assert df.duplicated().sum() == 0 and df.isna().sum().sum() == 0, "dataset not clean"
assert set(df.is_phishing) == {0, 1}
assert clean_text("URGENT!! http://x.com now 123") == "urgent urltoken now"
assert find_urls("see www.a.com/x, ok") == ["www.a.com/x"]
assert "Uses IP address instead of domain" in analyse_url("http://192.168.1.5/login")
assert "URL shortener" in analyse_url("http://bit.ly/abc")
feats, _ = extract_features("Urgent", "", "Verify your password now http://bit.ly/x")
assert feats["n_urls"] == 1 and feats["n_urgent_words"] >= 1 and feats["has_shortener"] == 1
model, res = train_model(df)
assert res["n_train"] + res["n_test"] == len(df)
assert res["confusion"].sum() == res["n_test"]
bad = predict_email(model, "Urgent: account suspended", "a@secure-login.xyz",
                    "Verify your password immediately: http://paypal-secure-login.xyz/verify")
ok = predict_email(model, "Meeting notes", "priya@college.edu", "Hi team, notes attached. Thanks, Priya")
assert bad["label"] == "Phishing" and ok["label"] == "Safe"
print("All tests passed. Accuracy:", round(res["metrics"]["accuracy"], 3))
