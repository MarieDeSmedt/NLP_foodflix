from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

sentence = ['This framework generates embeddings for each input sentence']

sentences = [ 'Sentences are passed as a list of string.',
              'This framework generates chocolat for each input sentence',
              'This framework embeddings for eachother input sentence',
        'The quick brown fox jumps over the lazy dog.']

sentence_embeddings = model.encode(sentences)
sentence_emb = model.encode(sentence)
#
# for sentence, embedding in zip(sentences, sentence_embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")

cosine_similarities = linear_kernel(sentence_emb,sentence_embeddings)
results = {}
similar_indices = cosine_similarities[0].argsort()[:-10:-1]
results= [(cosine_similarities[0][i],sentences[i]) for i in similar_indices]
print(results[0])

print("la phrase la plus ressemblante est : ",results[0][1] )