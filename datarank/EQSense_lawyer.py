import streamlit as st
import pandas as pd
from load_css import local_css

df = pd.read_csv("pyrdf2vec.csv")

st.set_page_config(
    page_title="EQSense",
    page_icon="✌️",
    layout="centered",
    initial_sidebar_state="expanded",
)
local_css("style.css")

METHOD = st.sidebar.radio("",("RDF2Vec","Word2Vec","FastText/BERT/GPT"))
if METHOD == "RDF2Vec":
    for i in range(df.shape[0]):
        # st.write(df['news'].values[i])
        # st.write(" ".join(eval(df['articles'].values[i])))
        # st.markdown("---")

        NEWS = f"<div align='justify: inter-word;><span class='highlight blue'><span class='bold'>NEWS-{i+1}: </span>{df['news'].values[i]}</span></div>"
        st.markdown(NEWS, unsafe_allow_html=True)

        ARTICLES = f"<div align='right'><span class='highlight red'><span class='bold'>ARTICLES: </span>{' '.join(eval(df['articles'].values[i]))}</span></div>"
        st.write("")
        st.markdown(ARTICLES, unsafe_allow_html=True)
        st.markdown("---")
else:
    st.title("Coming Soon")