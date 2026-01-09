🔍 Fake News Classifier with Explainable AI

(Machine Learning Project | Real-World Misinformation Detection)

The Fake News Classifier is a machine learning–based system designed to automatically detect misleading or false news articles, with a specific focus on the Indian news ecosystem.

The project combines robust NLP pipelines, multiple classification models, and Explainable AI (XAI) techniques to ensure predictions are not only accurate, but also transparent and interpretable.

🧩 Real-World Problem Addressed

The rapid spread of misinformation poses serious risks to:

Public trust

Democratic processes

Social harmony

Most fake news detection systems act as black boxes, offering predictions without explanations.
This project addresses that gap by integrating SHAP-based explainability, allowing users to understand why a piece of news is classified as real or fake.

🚀 Key Features
🧹 Text Preprocessing Pipeline

Tokenization

Stopword removal

Text normalization

TF-IDF vectorization for high-quality feature extraction

🤖 Model Training & Evaluation

Trained and evaluated multiple classifiers:

Logistic Regression

Random Forest

Naive Bayes

Key highlights:

Comparative model performance analysis

High accuracy on a cleaned and balanced dataset

Focus on generalization rather than overfitting

🔎 Explainable AI with SHAP

Integrated SHAP (SHapley Additive exPlanations)

Word-level contribution analysis

Clear interpretation of:

Why a news article is labeled fake

Which terms influenced a real prediction

📊 Dashboard & Analytical Insights

(Optional Streamlit integration)

Fake news trends over time

Most influential words for fake vs real news

Model confidence visualization

Interactive exploration of predictions

🧠 Tech Stack
Core Technologies

Python

scikit-learn

Pandas

NumPy

NLP & Explainability

TF-IDF Vectorization

SHAP (Explainable AI)

Visualization & UI

Matplotlib

Streamlit (optional dashboard)

Dataset

Cleaned and balanced dataset of Indian news articles

Binary labels: Fake / Real

🧠 System Workflow

News article or headline is provided as input

Text preprocessing and TF-IDF vectorization

Classification using trained ML model

SHAP computes word-level contributions

Prediction + explanation displayed to user

🔮 Future Enhancements

Multilingual fake news detection (Hindi + regional languages)

Social media post classification

Transformer-based models (BERT, IndicBERT)

Real-time browser or API integration

Credibility scoring instead of binary labels

📜 License

This project is intended for educational and research purposes.
Commercial use requires prior permission from the author.
