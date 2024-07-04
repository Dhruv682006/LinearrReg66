import streamlit as st
import pandas as pd
import numpy as np
import pickle

#Load the model
clf = pickle.load(open("mymodel.pkl","rb"))

def predict(data):
    clf = pickle.load(open("mymodel.pkl","rd"))
    return clf.predict(data)


st.title("advertising spends prediction using machine learning")
st.markdown("This Model Identify total spends on advertising")

st.header("Advertising spend on various Media")
coll,col2 = st.columns(2)


with coll:
    st.text("TV")
    tv = st.slider("Adver. spends on TV", 1.0,10000.0, 0.5)
    st.text("Radio")
    rd = st.slider("Adver. spends on Radio", 1.0,10000.0, 0.5)
    st.text("Newspaper")
    newspaper = st.slider("Adver. spends on Newspaper", 1.0,10000.0,0.5)
    
    st.text('')
    if st.button("seles prediction"):
        result = clf.predict(np.array([[tv,rd,newspaper]]))
        st.text(result[0])
        
        
st.markdown("developed By Dhruv Bhatt at NIELIT")