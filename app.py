import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from helper import *
st.set_page_config(page_title="SMS Classifier", layout="wide")
import time


nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

df = pd.read_csv('/config/workspace/clean.csv', encoding='ISO-8859-1')
spam_df = df[df["Predictions"] == "spam"]
ham_df = df[df["Predictions"] == "ham"]

selected = streamlit_menu()

if selected == "Live Demo":
    # Load models and transformers
    tfidf = pickle.load(open('word_to_vector.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    st.subheader("💬 SMS Classifier")
    input_sms = st.text_area("Enter the SMS", placeholder="Enter Message",
                             help='Enter message to verify whether it is spam or not spam.')
    st.sidebar.title("Sample Spam Messages")
    st.sidebar.markdown('''1. Congratulations! You've won a free cruise to the Bahamas! Claim your prize now!''')
    st.sidebar.markdown('''2. Urgent: Your account has been locked. Click the link to verify your identity.''')
    st.sidebar.write("━━━━━━━━━━━━━━")
    st.sidebar.title("Sample Ham Messages")
    st.sidebar.markdown('''1. Hi there! Just wanted to check in and see how you're doing. Let's catch up soon!''')
    st.sidebar.markdown('''2. The meeting is scheduled for tomorrow at 2 PM in the conference room. See you there!''')
    # Center the Predict button using CSS
    st.markdown(
        """
        <style>
            .stButton > button {
                display: block;
                margin: 0 auto;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button('Predict'):
        if not input_sms:
            st.warning("Please enter a message for prediction.")
        else:
            transform_sms = data_transform(input_sms)
            vector_sms = tfidf.transform([transform_sms])
            output = model.predict(vector_sms)[0]

            if output == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")

elif selected == "Live EDA":
    st.sidebar.write("### Live EDA")
    st.write('In Progress will update soon')

elif selected == "Source Code":
    st.sidebar.write("### Source Code 📁")
    st.write("### GitHub Repository 🚀")
    st.write("If you're interested in the source code, you can find it on GitHub. Feel free to explore and provide feedback.")
    st.write("GitHub Repository: [SMS Classifier](https://github.com/iamaakashpal/SMS-CLASSIFIER)")

    st.write("### Dataset 📊")
    st.write("For access to the dataset used in this project, you can find it at the following link.")
    st.write("Dataset: [Dataset Link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)")

    st.write("### Exploratory Data Analysis (EDA) Notebook 📈")
    st.write("For a detailed exploration of the dataset and insights gained, check the EDA notebook.")
    st.write("EDA Notebook: [EDA Notebook](https://www.kaggle.com/code/aakashpal/spam-ham-classifier-exploratory-data-analysis/notebook)")

    st.write("### Live Demo ▶️")
    st.write("For a live working demonstration of the SMS spam classifier, visit the provided demo link.")
    st.write("Live Demo: [Live Demo](a)")

    st.write("### YouTube Video 🎥")
    st.write("Check out the YouTube video for a visual overview of the SMS Spam or Ham Classifier project.")
    st.write("YouTube Video: In Progress")

elif selected == "Connect with Me":
    st.sidebar.write("### Connect with Me")
    st.write("#### Feel free to reach out for feedback, questions, or collaborations!")

    # Set a fixed size for all images
    image_size = "50px"

    # Use HTML and CSS to display images with specified height and width
    icons_html = [
        f"<a href='mailto:aakashpal1198@gamil.com'><img src='https://raw.githubusercontent.com/iamaakashpal/SMS-CLASSIFIER/main/social/gmail.png' style='margin-right: 25px;' height='{image_size}' width='{image_size}'></a>",
        f"<a href='https://github.com/iamaakashpal' target='_blank'><img src='https://raw.githubusercontent.com/iamaakashpal/SMS-CLASSIFIER/main/social/github.png' style='margin-right: 25px;' height='{image_size}' width='{image_size}'></a>",
        f"<a href='https://www.linkedin.com/in/iamaakashpal/' target='_blank'><img src='https://raw.githubusercontent.com/iamaakashpal/SMS-CLASSIFIER/main/social/linkedin.png' style='margin-right: 25px;' height='{image_size}' width='{image_size}'></a>",
        f"<a href='https://www.kaggle.com/aakashpal' target='_blank'><img src='https://raw.githubusercontent.com/iamaakashpal/SMS-CLASSIFIER/main/social/kaggle.png' style='margin-right: 25px;' height='{image_size}' width='{image_size}'></a>",
        f"<a href='https://www.youtube.com/channel/UCYBlRxDjQSFtst6WnjcXsqQ' target='_blank'><img src='https://raw.githubusercontent.com/iamaakashpal/SMS-CLASSIFIER/main/social/yt.png' height='{image_size}' width='{image_size}'></a>",
    ]

    st.markdown(" ".join(icons_html), unsafe_allow_html=True)







