import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

def streamlit_menu(example=1):
        selected = option_menu(
            menu_title=None,  # required
            options=["Live Demo", "Live EDA", "Source Code", "Connect with Me"],  # required
            icons=["house", "book", "envelope","envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

def data_transform(message):
    # Step 1: Convert to lowercase
    lower_message =  message.lower()
    
    # Step 2: Tokenize the message
    list_of_words = nltk.word_tokenize(lower_message)

    # Step 3: Remove non-alphanumeric characters
    updated_list_of_word = []

    for i in list_of_words:
        if i.isalnum():
            updated_list_of_word.append(i)

    # Step 4: Remove stopwords and punctuation
    list_without_stopwords = []
    for i in updated_list_of_word:
        if i not in stopwords.words('english') and i not in string.punctuation:
            list_without_stopwords.append(i)

    # Step 5: Stemming
    final_list = []
    for i in list_without_stopwords:
        stem_word = ps.stem(i)
        final_list.append(stem_word)
    
    # Join the final list into a string and return
    return " ".join(final_list)
