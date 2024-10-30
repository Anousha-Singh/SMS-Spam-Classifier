import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def transform_text(text):
    text = text.lower()   #Convert to lowercase
    text = text.split()   #Split text 

    ps = PorterStemmer()  #To find the root form of the words
    
    final_text = []
    for i in text:
        if (i.isalnum()) and (i not in stopwords.words('english')) and (i not in string.punctuation):
            final_text.append(ps.stem(i))   #Remove special characters, puntuations and stopwords 
    
    return " ".join(final_text)  

tfidf = pickle.load(open("vectorizer.pkl",'rb'))
modelNB = pickle.load(open("NBmodel.pkl",'rb'))
modelRF = pickle.load(open("RFmodel.pkl",'rb'))
modelKN = pickle.load(open("KNmodel.pkl",'rb'))

st.title("Spam Classifier")

input_text = st.text_input("Enter text:")

option = st.radio("Select a model: ", ["Naive Bayes","Random Forest","K Neighbours"], horizontal = True)

if option == "Naive Bayes":
    model = modelNB
elif option == "Random Forest":
    model = modelRF
else:
    model = modelKN


if st.button("Predict"):
    transformed_text = transform_text(input_text)

    vector_input = tfidf.transform([transformed_text])
    result = model.predict(vector_input)[0]
    # st.write("using: ", model)

    if(result==1):
        st.header("Spam")
    else:
        st.header("Not Spam")
