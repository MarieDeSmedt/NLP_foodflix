# Importing the libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity


def reco(user_input,df):
    vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
    vectors_content = vectorizer.fit_transform(df['content'])
    vectors_product = vectorizer.transform([user_input])
    cosine_similarities = linear_kernel(vectors_product,vectors_content)
    results = {}
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    similar_items = [(cosine_similarities[0][i], df['index'][i]) for i in similar_indices]
    results = reco(user_input)
    # on affiche les produits similaires
    for i in range(4):
        print(df.iloc[results[i][1]])
        print("----------------------")
