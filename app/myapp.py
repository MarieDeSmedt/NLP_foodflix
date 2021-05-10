import sys
sys.path.insert(0, "/home/apprenant/PycharmProjects/NLP_foodflix")

import streamlit as st
import pandas as pd
import numpy as np

from src.recommandation import get_recommandation, get_idea, get_bert

def display_result(title,col_list,results):
    st.title(title)
    for i in range(4):
        nb = str(i + 1)
        my_expander = st.beta_expander("proposition n°" + nb, expanded=False)
        with my_expander:
            for col in col_list:
                if title == "Et avec bert?":
                    dis = results[i][1]
                    st.write(dis)
                else:
                    dis = df[col].iloc[results[i][1]]
                    if dis:
                        st.subheader(col, " : ")
                        st.write(dis)

# #####################################################################""

# Importing the dataset
df = pd.read_csv('/home/apprenant/PycharmProjects/NLP_foodflix/data/02_intermediate/only_string.csv')
df = df.rename(columns={df.columns[0]: "index"})
df['content'] = df[['product_name', 'brands', 'generic_name', 'categories']].astype(str).apply(lambda x: ' // '.join(x),
                                                                                               axis=1)
df['content'].fillna('Null', inplace=True)
col_list = ['product_name', 'brands', 'generic_name', 'categories']
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
        display_result(title, col_list, results)

    elif tool == "CountVectorizer":
        results = get_idea(new_input, df)
        title = "Mais vous avez envie de changement?"
        display_result(title, col_list, results)

    else:
        results = get_bert(new_input, df)
        title = "Et avec bert?"
        for i in range(4):
            nb = str(i + 1)
            my_expander = st.beta_expander("proposition n°" + nb, expanded=False)
            with my_expander:
                dis = results[i][1]
                st.write(dis)












