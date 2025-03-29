import streamlit as st
import pickle
import re

# กำหนดฟังก์ชัน preprocess
def preprocess(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

# โหลดโมเดลและ TfidfVectorizer
with open('review_nb_model_preprocessed.pkl', 'rb') as f:
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
