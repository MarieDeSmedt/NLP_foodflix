from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

from sentence_transformers import SentenceTransformer, models


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

def get_bert(user_input, df):
    mymodel = models.Transformer('camembert-base')
    pooling_models = models.Pooling(mymodel.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[mymodel, pooling_models])

    sentence = [user_input]
    sentences = df['content'].head(200).tolist()

    sentence_embeddings = model.encode(sentences)
    sentence_emb = model.encode(sentence)

    cosine_similarities = linear_kernel(sentence_emb, sentence_embeddings)
    results = {}
    similar_indices = cosine_similarities[0].argsort()[:-5:-1]
    results = [(cosine_similarities[0][i], sentences[i]) for i in similar_indices]
    return results