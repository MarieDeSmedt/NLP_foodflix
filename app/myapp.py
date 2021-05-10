import sys
sys.path.insert(0, "/home/apprenant/PycharmProjects/NLP_foodflix")

import streamlit as st
import pandas as pd

from src.recommandation import get_recommandation, get_idea, get_bert, get_col_name, get_col_unit,display_result




# #####################################################################""

# Importing the dataset
data = pd.read_csv('/home/apprenant/PycharmProjects/NLP_foodflix/data/01_raw/initial.csv')
df = pd.read_csv('/home/apprenant/PycharmProjects/NLP_foodflix/data/02_intermediate/only_string.csv')
df = df.rename(columns={df.columns[0]: "index"})
df['content'] = data[['product_name', 'brands', 'generic_name', 'categories']].astype(str).apply(
    lambda x: ' // '.join(x),
    axis=1)
df['content'].fillna('Null', inplace=True)

col_list = ['generic_name', 'brands', 'categories',
            'nutrition_grade_fr', 'energy_100g', 'fat_100g', 'saturated-fat_100g',
            'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g',
            'salt_100g', 'sodium_100g', 'fruits-vegetables-nuts_100g']
# ##########################################################################################"
# header
st.image('/home/apprenant/PycharmProjects/NLP_foodflix/images/foodflix.png', output_format='PNG')
# ###################################################################################
# sidebar
tool = st.sidebar.radio('Choose Tool:', ["TfidfVectorizer", "CountVectorizer", "Roberta"])
new_input = st.sidebar.text_input('recherche')

# ################################################################################
# display
title = ""
if new_input:
    if tool == "TfidfVectorizer":
        results = get_recommandation(new_input, df)
        title = "Vous aimez le " + new_input + "?"

    elif tool == "CountVectorizer":
        results = get_idea(new_input, df)
        title = "Mais vous avez envie de changement?"

    else:
        results = get_bert(new_input, df)
        title = "Et avec bert?"

    display_result(title, col_list, results,tool,data)
