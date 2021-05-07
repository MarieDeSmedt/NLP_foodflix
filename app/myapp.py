import sys
sys.path.insert(0,"/home/apprenant/PycharmProjects/NLP_foodflix")

import streamlit as st
import pandas as pd

from src.recommandation  import reco

st.title("Moteur de recommandation:")

# Importing the dataset
df = pd.read_csv('/home/apprenant/PycharmProjects/NLP_foodflix/data/02_intermediate/only_string.csv')
df = df.rename(columns={ df.columns[0]: "index" })
df['content'] = df[['product_name','brands','generic_name','categories']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
# Fillna
df['content'].fillna('Null', inplace = True)

user_input = st.text_input('recherche')

reco(user_input,df)