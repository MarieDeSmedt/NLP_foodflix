from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

from sentence_transformers import SentenceTransformer, models


def get_recommandation(user_input, df):
    vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    vectors_content = vec.fit_transform(df['content'])
    vectors_product = vec.transform([user_input])
    cosine_similarities = linear_kernel(vectors_product, vectors_content)
    results = {}
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
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    sentence = [user_input]
    sentences = df['content'].head(200).tolist()

    sentence_embeddings = model.encode(sentences)
    sentence_emb = model.encode(sentence)

    cosine_similarities = linear_kernel(sentence_emb, sentence_embeddings)
    results = {}
    similar_indices = cosine_similarities[0].argsort()[:-5:-1]
    results = [(cosine_similarities[0][i], sentences[i]) for i in similar_indices]
    return results

def get_col_name(i):
    switcher = {
        'generic_name': 'Nom:',
        'brands': 'Marque:',
        'categories': 'Catégorie:',
        'nutrition_grade_fr': 'Score:',
        'energy_100g': 'Energie aux 100g:',
        'fat_100g': 'Matières grasses:',
        'carbohydrates_100g': 'Glucides:',
        'fiber_100g': 'Fibres:',
        'sugars_100g': 'Sucre:',
        'proteins_100g': 'Protéines:',
        'fruits-vegetables-nuts_100g': 'Noix:',
        'salt_100g': 'Sel:',
        'sodium_100g': 'Sodium:',
        'saturated-fat_100g': 'Graisses saturées:'
    }
    return switcher.get(i, "Error")

def get_col_unit(i):
    switcher = {
        'generic_name': '',
        'brands': '',
        'categories': '',
        'nutrition_grade_fr': '',
        'energy_100g': '',
        'fat_100g': 'g/100',
        'carbohydrates_100g': 'g/100',
        'fiber_100g': 'g/100',
        'sugars_100g': 'g/100',
        'proteins_100g': 'g/100',
        'fruits-vegetables-nuts_100g': 'g/100',
        'salt_100g': 'g/100',
        'sodium_100g': 'g/100',
        'saturated-fat_100g': 'g/100 '
    }
    return switcher.get(i, "Error")