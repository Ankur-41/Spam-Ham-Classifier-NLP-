import dill
import string
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Loading the trained model and vectorizer
with open('mnb_model.pkl','rb') as file:
    model = pickle.load(file)
with open('vectorizer.pkl','rb') as file:
    vectorizer = pickle.load(file)

# Loading preprocessor function
with open('preprocessor_func.pkl','rb') as file:
    preprocessing_func = dill.load(file)

# Streamlit App UI
st.set_page_config(page_title='Spam/Ham Prediction',page_icon="📩")
st.title('📩 Spam / Ham Email Classifier')
st.write('Enter your text or email message below : ')
user_inp = st.text_area('Type your message here : ')

if st.button('Predict'):
    if user_inp.strip() == '':
        st.warning('Please enter a message for prediction')
    else:
        # preprocess user input
        cleaned_text = preprocessing_func(user_inp)

        # vectorize
        vec_txt = vectorizer.transform([cleaned_text])

        # predict
        prediction = model.predict(vec_txt)[0]

        if prediction == 1:
            st.success("✅ This message is **Not Spam (Ham)**.")
        else:
            st.error("🚨 This message is likely **Spam**!")

st.markdown("---")
st.caption("Developed by Ankur Kumar | Streamlit NLP App")










