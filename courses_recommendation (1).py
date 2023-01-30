import streamlit as st
import pandas as pd
import numpy as np

st.header("Course Recommender System")
st.subheader("This system will recommend the five most similar courses to your course of choice.")

data = pd.read_csv(r'C:\Users\MASTER\Dataset\EdX.csv')  
data['Description'] = data['Name'] + ' ' + data['About'] + ' ' + data['Course Description']
data.drop('Link', axis = 1, inplace = True)

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

en_stopwords = stopwords.words("English")
lemma = WordNetLemmatizer()

def clean(text):
    text = re.sub("[^A-Za-z1-9 ]", "", text) #removes punctuation marks
    text = text.lower() #changes to lower case
    tokens = word_tokenize(text) #tokenize the text
    clean_list = [] 
    for token in tokens:
        if token not in en_stopwords: #removes stopwords
            clean_list.append(lemma.lemmatize(token)) #lemmatizing and appends to clean_list
    return " ".join(clean_list)# joins the tokens

data.Description = data.Description.apply(clean)

def Vectorization(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    data_matrix = vectorizer.fit_transform(data.Description)
    return data_matrix

vectors = Vectorization(data.Description)

from sklearn.metrics.pairwise import cosine_similarity
                        

def Recommendation(test_matrix, row_num):
    similarity = cosine_similarity(test_matrix)
    similar_courses = list(enumerate(similarity[row_num]))
    sorted_similar_courses = sorted(similar_courses, key=lambda x:x[1], reverse= True)[1:6]
#This part will return the description of the recommended courses
    i = 0
    for item in sorted_similar_courses:
        course_description = data[data.index == item[0]]["Name"].values[0]
        recommendations = st.write(f"{i+1}\t{course_description}\n")
        i = i + 1
    return recommendations

#input article for recommendation
st.write('Welcome! Choose a course')
Course_name = (st.selectbox("Name of Course:", options = data.Name, label_visibility = 'collapsed'))
Course_index = data[data.Name == Course_name].index[0]

if st.button('Show similar courses'):
    Recommendation(vectors, Course_index)







