
import pandas as pd
from more_utils import *

data = pd.read_csv("../data/01_raw/initial.csv", sep=',', low_memory=False)



# just keep the strings for the npl

df = data.select_dtypes(include="object")
df = df.drop('nutrition_grade_fr', axis = 1)
info_df(df)

df.to_csv('../data/02_intermediate/only_string.csv')