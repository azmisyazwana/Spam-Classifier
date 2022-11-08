import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#NLP MODEL
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1)
df.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)
df.drop_duplicates(inplace = True)  #Drop duplicate (cleaner dataset)
df['labels'] = df['labels'].map({'ham': 0, 'spam': 1})

def clean_data(message): #remove all the stopwords and all the punctuation
    message_without_punc = [character for character in message if character not in string.punctuation]
    message_without_punc = ''.join(message_without_punc)
    
    separator = ' '
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english')])

#Apply this function to every message
df['message'] = df['message'].apply(clean_data)

x = df['message']
y = df['labels']

#convert message into vectors
cv = CountVectorizer()

x = cv.fit_transform(x)


#Train and Test with Naive Bayes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

model = MultinomialNB().fit(x_train, y_train)

#Prediction testing data
predictions = model.predict(x_test)

#Compare the prediction and the actual
print("Accuracy Score: ", accuracy_score(y_test, predictions))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, predictions))
print("Classification Report: ")
print(classification_report(y_test, predictions))


def predict(text):
    labels = ['Not Spam', 'Spam']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message is looking to be: ' + labels[v])

# print(predict(['Congratulations, you won a lottery of $4000']))

st.title("Spam Classifier")
st.image("spam.webp")
user_input = st.text_input('Write your message')
submit = st.button('Predict')

if submit: #If the button pressed
    answer = predict([user_input])
    st.text(answer )
    


# st.title('Excel Plotter 📊')
