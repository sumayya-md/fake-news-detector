import streamlit as st
import joblib

st.title("📰 Fake News Detector")

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

news_text = st.text_area("Paste news article text below:")

if st.button("Check if Fake"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([news_text])
        pred = model.predict(vec)
        st.success("🚫 Fake News" if pred[0] == 1 else "✅ Real News")