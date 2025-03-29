import streamlit as st
import pickle

# โหลดโมเดลและ TfidfVectorizer
with open('review_model_preprocessed.pkl', 'rb') as f:
    tfidf, model = pickle.load(f)

# สร้าง Streamlit UI
st.title('Review Rating Prediction (Naive Bayes, Preprocessed)')
review_text = st.text_area('Enter your review:')

if st.button('Predict'):
    if review_text:
        review_tfidf = tfidf.transform([review_text])
        prediction = model.predict(review_tfidf)[0]
        st.write(f'Predicted Rating: {prediction}')
    else:
        st.warning('Please enter a review.')