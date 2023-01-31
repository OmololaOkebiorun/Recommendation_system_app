import streamlit as st
import pandas as pd
import numpy as np

st.header("Course Recommender System")
st.subheader("This system will recommend the five most similar courses to your course of choice.")


data = pd.read_csv('EdX.csv')  
data['Description'] = data['Name'] + ' ' + data['About'] + ' ' + data['Course Description']
data.drop('Link', axis = 1, inplace = True)

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

en_stopwords = ['i','me','my','myself','we','our','ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her','hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the','and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in','out', 'on', 'off', 'over', 'under','again', 'further', 'then', 'once', 'here', 'there', 'when','where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most','other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too','very','s', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',"needn't",'shan',"shan't",'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',"wouldn't"]
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







