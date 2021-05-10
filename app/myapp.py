import sys
sys.path.insert(0, "/home/apprenant/PycharmProjects/NLP_foodflix")

import streamlit as st
import pandas as pd

from src.recommandation import get_recommandation, get_idea

# Importing the dataset
df = pd.read_csv('/home/apprenant/PycharmProjects/NLP_foodflix/data/02_intermediate/only_string.csv')
df = df.rename(columns={df.columns[0]: "index"})
df['content'] = df[['product_name', 'brands', 'generic_name', 'categories']].astype(str).apply(lambda x: ' // '.join(x),
                                                                                               axis=1)
df['content'].fillna('Null', inplace=True)

# ##########################################################################################"
#header
st.header("Démonstration NLP:")

# ###################################################################################
# sidebar
tool = st.sidebar.radio('Choose Tool:',["TfidfVectorizer","CountVectorizer", "BERT"])
new_input = st.sidebar.text_input('recherche')

# ################################################################################
# display
title = ""
if new_input :
    if tool == "TfidfVectorizer":
        results = get_recommandation(new_input, df)
        title = "Vous aimez le " + new_input + "?"
    elif tool == "CountVectorizer":
        results = get_idea(new_input, df)
        title = "Mais vous avez envie de changement?"
    else:
        title = ("et Bert?")


    st.title(title)
    for i in range(4):
        st.write(df['product_name'].iloc[results[i][1]])
        st.write(df['brands'].iloc[results[i][1]])
        st.write(df['generic_name'].iloc[results[i][1]])
        st.write(df['categories'].iloc[results[i][1]])
        st.balloons()
