from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer, models

data = pd.read_csv('/home/apprenant/PycharmProjects/NLP_foodflix/data/01_raw/initial.csv')
df = pd.read_csv('/home/apprenant/PycharmProjects/NLP_foodflix/data/02_intermediate/only_string.csv')
df = df.rename(columns={df.columns[0]: "index"})
df['content'] = df[['product_name', 'brands', 'generic_name', 'categories']].astype(str).apply(lambda x: ' // '.join(x),
                                                                                               axis=1)
df['content'].fillna('Null', inplace=True)

print(data.columns)

# mymodel = models.Transformer('camembert-base')
# pooling_models = models.Pooling(mymodel.get_word_embedding_dimension(),
#                                 pooling_mode_mean_tokens=True,
#                                 pooling_mode_max_tokens=False)
# model = SentenceTransformer(modules =[mymodel, pooling_models])
#
#
# sentence = ["chocolat"]
# sentences = df['content'].head(15).tolist()
#
# sentence_embeddings = model.encode(sentences)
# sentence_emb = model.encode(sentence)
#
#
# cosine_similarities = linear_kernel(sentence_emb, sentence_embeddings)
# results = {}
# similar_indices = cosine_similarities[0].argsort()[:-5:-1]
# results = [(cosine_similarities[0][i], sentences[i]) for i in similar_indices]
# print(results)




# model = SentenceTransformer('paraphrase-distilroberta-base-v1')
#
# sentence = ['This framework generates embeddings for each input sentence']
#
# sentences = [ 'Sentences are passed as a list of string.',
#               'This framework generates chocolat for each input sentence',
#               'This framework embeddings for eachother input sentence',
#         'The quick brown fox jumps over the lazy dog.']
#
# sentence_embeddings = model.encode(sentences)
# sentence_emb = model.encode(sentence)
# #
# # for sentence, embedding in zip(sentences, sentence_embeddings):
# #     print("Sentence:", sentence)
# #     print("Embedding:", embedding)
# #     print("")
#
# cosine_similarities = linear_kernel(sentence_emb,sentence_embeddings)
# results = {}
# similar_indices = cosine_similarities[0].argsort()[:-10:-1]
# results= [(cosine_similarities[0][i],sentences[i]) for i in similar_indices]
# print(results[0])
#
# print("la phrase la plus ressemblante est : ",results[0][1] )