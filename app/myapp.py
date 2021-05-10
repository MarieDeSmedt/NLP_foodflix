import sys
sys.path.insert(0, "/home/apprenant/PycharmProjects/NLP_foodflix")

import streamlit as st
import pandas as pd

from src.recommandation import get_recommandation, get_idea, get_bert, get_col_name, get_col_unit


def display_result(title, col_list, results):
    st.subheader(title)
    for i in range(4):
        if tool == "Roberta":
            dis = results[i][1]
            st.write(dis)
        else:
            my_expander = st.beta_expander(data['product_name'].iloc[results[i][1]], expanded=False)
            with my_expander:
                j = 0
                for col in col_list:
                    dis = data[col].iloc[results[i][1]]
                    if j <5:
                        col1, col2 = st.beta_columns(2)
                        with col1:
                            st.write(get_col_name(col), dis, get_col_unit(col))
                    else:
                        with col2:
                            st.write(get_col_name(col), dis, get_col_unit(col))
                    j += 1


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

    display_result(title, col_list, results)
