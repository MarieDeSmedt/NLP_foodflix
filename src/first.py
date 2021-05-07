
import pandas as pd
from more_utils import *
import wordcloud

data = pd.read_csv("/home/apprenant/PycharmProjects/NLP_foodflix/data/01_raw/initial.csv", sep=',', low_memory=False)



# just keep the strings for the npl

df = data.select_dtypes(include="object")
df = df.drop('nutrition_grade_fr', axis = 1)
info_df(df)

df.to_csv('/home/apprenant/PycharmProjects/NLP_foodflix/data/02_intermediate/only_string.csv')