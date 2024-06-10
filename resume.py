import streamlit as st
import fitz
import re
import string
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,roc_curve
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sentence_transformers import SentenceTransformer, util

st.title("Automated Resume Filtering System")
st.write('<hr style="height: px; background-color: gray; border: none; margin: px 0;" />', unsafe_allow_html=True)
interface = st.container()


uploaded_cv_files , uploaded_jd_files = st.columns(spec = [1, 1])


with interface:
    with uploaded_cv_files:
        st.markdown('<h1 style="font-size: 20px;">Upload CVs</h1>', unsafe_allow_html=True)
        uploaded_cv_files = st.file_uploader("", type=['pdf'])
    
    with uploaded_jd_files:
        st.markdown('<h1 style="font-size: 20px;">Upload Job Descriptions</h1>', unsafe_allow_html=True)
        uploaded_jd_files = st.file_uploader(" ",  type=['pdf'])
    
    df_cv = pd.DataFrame(columns=['cv'])
    df_job = pd.DataFrame(columns=['job'])    
  
    if uploaded_cv_files and uploaded_jd_files:


        # PDF faylını oxumaq
        cv_files = uploaded_cv_files
        job_files = uploaded_jd_files   

        
        document = fitz.open(cv_files)
        text = ""
        # Hər səhifənin mətnini oxumaq
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()

        df_cv = df_cv.append({'cv': text}, ignore_index=True)

        document = fitz.open(job_files)
        text = ""
        # Hər səhifənin mətnini oxumaq
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()

        df_job = df_job.append({'job': text}, ignore_index=True)


        nltk.download('stopwords')
        nltk.download('wordnet')
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        # Mətn təmizləmə funksiyası (stop words çıxarılması daxil)
        def clean_text_cv(text):
            text = text.lower()
            text = re.sub(r'\b\d{3,}\b', '', text)
            text = re.sub(r'[^\w\s.]', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            match = re.search(r'(objective|professional summary)', text, re.IGNORECASE)
            if text.strip()[-1].isdigit():
                text = re.sub(r'\d+\s*$', '', text).strip()
            if match:
                return text[match.end():]  # Başlıqdan sonrakı mətn hissəsini qaytar
            return text  

            # Stop words sil
            text = ' '.join(word for word in text.split() if word not in stop_words)

            # Lemmatization
            text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())    

        def clean_text_job(text):
            text = text.lower()
            text = re.sub(r'\b\d{3,}\b', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            match = re.search(r'(how to apply)', text, re.IGNORECASE)
            if match:
                return text[:match.start()]  # Başlıqdan əvvəlki mətn hissəsini qaytar
            return text  # Əgər başlıq tapılmazsa, bütün mətni qaytar

            # Stop words sil
            text = ' '.join(word for word in text.split() if word not in stop_words)

            # Lemmatization
            text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())


        df_cv['clean_cv'] = df_cv['cv'].apply(clean_text_cv)
        df_job['clean_job'] = df_job['job'].apply(clean_text_job)

        cv_texts = df_cv['clean_cv'].tolist()
        job_texts = df_job['clean_job'].tolist()
        
        with open('best_model.pickle', 'rb') as file:
            svm_model = pickle.load(file)

        # Modeli yükləyin
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


        # CV və iş təsvirlərini vektorlaşdirin
        cv_embeddings = model.encode(cv_texts, convert_to_tensor=True)
        job_embeddings = model.encode(job_texts, convert_to_tensor=True)

        # Cosine Similarity hesablayın
        similarity_scores = util.pytorch_cos_sim(cv_embeddings, job_embeddings)
        netice = svm_model.predict(similarity_scores)
        tensor_data = torch.tensor(similarity_scores)

        # Tensoru faizə çevirmək
        percentage = tensor_data.item() * 100

        # Faizi formatlamaq
        formatted_percentage = "{:.1f}%".format(percentage)

        if 'button_clicked' not in st.session_state:
            st.session_state.button_clicked = False
        def callback():
            st.session_state.button_clicked = True
        if (st.button('Predict', on_click=callback) or st.session_state.button_clicked):
            if netice == 0:  
                st.markdown('Your CV does not match the vacancy')
            else:
                st.markdown('Your CV is suitable for the vacancy')
            st.markdown(f'Your cv and vacancy similarity percentage {formatted_percentage} ')