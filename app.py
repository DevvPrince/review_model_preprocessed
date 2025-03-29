import streamlit as st
import joblib
import re

# ฟังก์ชัน preprocess
def preprocess(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

# โหลดโมเดลและ TfidfVectorizer โดยใช้ joblib
#model_filename = 'review_nb_model_preprocessed.pkl' #เปลี่ยนชื่อไฟล์

model_filename = 'review_model_preprocessed.pkl'

#print(f"Loading model from: {os.path.abspath(model_filename)}")  # แสดงพาธที่กำลังโหลดโมเดล
tfidf, model = joblib.load(model_filename)

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
