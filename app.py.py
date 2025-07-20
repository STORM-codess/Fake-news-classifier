import streamlit as st
import pickle
import shap
import numpy as np

# Load model and vectorizer
model = pickle.load(open('model_1.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_1.pkl', 'rb'))

# Wrapper for SHAP to work with TF-IDF input
def predict_proba_fn(texts):
    X = vectorizer.transform(texts)
    return model.predict_proba(X)

# Background sample for SHAP explainer
X_background = vectorizer.transform([" "])
explainer = shap.Explainer(model, X_background, feature_names=vectorizer.get_feature_names())

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's **Real** or **Fake**:")

# Input
user_input = st.text_area("News Text:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("❗ Please enter some text")
    else:
        # Transform and predict
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0]

        # Output
        if prediction == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")
        st.info(f"Confidence: {prob[prediction]*100:.2f}%")

        # SHAP Explanation
        with st.expander("🔍 Explanation (SHAP)"):
            st.subheader("📊 Top Words Influencing Prediction")
            shap_values = explainer(X_input)
            shap.plots.bar(shap_values[0], max_display=10)
            st.pyplot(bbox_inches='tight')

