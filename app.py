import dill
import string
import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Loading the trained model and vectorizer
with open('mnb_model.pkl','rb') as file:
    model = pickle.load(file)
with open('vectorizer.pkl','rb') as file:
    vectorizer = pickle.load(file)

# Loading preprocessor function
# with open('preprocessor_func.pkl','rb') as file:
#     preprocessing_func = dill.load(file)

# Streamlit App UI
st.set_page_config(page_title='Spam/Ham Prediction',page_icon="📩")
st.title('📩 Spam / Ham Email Classifier')
st.write('Enter your text or email message below : ')
user_inp = st.text_area('Type your message here : ')

# if it gives connection error then we will use this instead of preprocessor_func.pkl
def preprocessed_text(txt_inp):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = txt_inp.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    words = [word for word in text.split() if word.isascii()]
    text = ' '.join(words)
    # tokenize
    pr_words = word_tokenize(text)
    # lemmatize and remove stopwords
    pr_words = [lemmatizer.lemmatize(word,pos='v')for word in pr_words if word not in stop_words]
    return ' '.join(pr_words)



if st.button('Predict'):
    if user_inp == '':
        st.warning('Please enter a message for prediction')
    else:
        # preprocess user input
        cleaned_text = preprocessed_text(user_inp)

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










