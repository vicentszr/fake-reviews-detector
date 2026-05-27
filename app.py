import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import pickle

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Reviews Analyzer",
    page_icon="🔍",
    layout="centered"
)

# ── LOAD AND TRAIN MODEL ──────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv('fake reviews dataset.csv')
    df['label_num'] = df['label'].map({'CG': 1, 'OR': 0})
    
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['text_'])
    y = df['label_num']
    
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X, y)
    
    return model, vectorizer

# ── UI ────────────────────────────────────────────────────────
st.title("🔍 Fake Reviews Analyzer")
st.markdown("Paste a customer review below and find out if it's **real or fake**.")

with st.spinner("Loading model..."):
    model, vectorizer = load_model()

review = st.text_area("Enter review here:", height=150, placeholder="Type or paste a review...")

if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        probability = model.predict_proba(review_tfidf)[0]
        
        if prediction == 1:
            st.error(f"⚠️ FAKE REVIEW — Confidence: {probability[1]*100:.1f}%")
        else:
            st.success(f"✅ REAL REVIEW — Confidence: {probability[0]*100:.1f}%")
        
        st.markdown(f"**Review length:** {len(review)} characters")