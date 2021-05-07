from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import linear_kernel


def get_recommandation(user_input, df):
    vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    vectors_content = vec.fit_transform(df['content'])
    vectors_product = vec.transform([user_input])
    cosine_similarities = linear_kernel(vectors_product, vectors_content)
    results={}
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    results = [(cosine_similarities[0][i], df['index'][i]) for i in similar_indices]
    return results

def get_idea(user_input, df):
    vec = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    vectors_content = vec.fit_transform(df['content'])
    vectors_product = vec.transform([user_input])
    cosine_similarities = linear_kernel(vectors_product, vectors_content)
    results = {}
    similar_indices = cosine_similarities[0].argsort()[:-10:-1]
    results_count = [(cosine_similarities[0][i], df['index'][i]) for i in similar_indices]
    return results_count