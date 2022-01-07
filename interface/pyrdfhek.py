from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
import pickle,os,sys
import numpy as np

news = sys.argv[-1]

f = open("pyrdf2vec/df_pyrdf2vec.pkl","rb")
df = pickle.load(f)
f.close()
f = open("pyrdf2vec/vert_data.pkl","rb")
vert_data = pickle.load(f)
f.close()
f = open("pyrdf2vec/articles.pkl","rb")
articles = pickle.load(f)
f.close()

import spacy
nlp = spacy.load("en_core_web_lg")

for i in os.listdir("pyrdf2vec/embeddings"):
    f = open(f"pyrdf2vec/embeddings/{i}","rb")
    locals()[i.replace(".embd","")] = pickle.load(f)
    f.close()

def stTriplets(para):
    ans = []
    for s in para:
        buff = os.popen(f"echo '{s}' | java -mx1g -cp 'pyrdf2vec/stanford-corenlp-4.2.2/*' edu.stanford.nlp.naturalli.OpenIE -max_entailments_per_clause 500").read()
        for t in buff.strip().split('\n'):
            ans.append([i.lower() for i in t.split('\t')[1:]])
    return ans

def grapher(trips):
    G = KG()
    for t in trips:
        if t!=[]:
            G.add_walk(Vertex(t[0]),Vertex(t[1]),Vertex(t[2]))
    return G

def g2e(G, entities):
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=10),
        walkers=[RandomWalker(40, 100, with_reverse=False, n_jobs=-1)],
        verbose=1
    )
    if entities!=[]:
        embeddings, literals = transformer.fit_transform(G,entities)
    return np.array(embeddings)

def maxFinder(d):
    MAXI = list(d.values()).index((max(list(d.values()))))
    return list(d.keys())[MAXI]
def sentTok(para):
    doc = nlp(para)
    return [i.text for i in doc.sents]
def T5(D):
    D = dict(sorted(D.items(), key=lambda item: item[1],reverse=True))
    return list(D.keys())[:5]
def solutionD(sent):
    trips = stTriplets(sentTok(sent))
    G = grapher(trips)
    q_entities = [i.name for i in list(G._entities)]
    
    if q_entities!=[]:
        q_embeddings = g2e(G,q_entities)
    else:
        return []
    
    from sklearn.metrics.pairwise import cosine_similarity
    q_entities = [i.name for i in list(G._entities)]
    rank = {}
    for ent in q_entities:
        for ar in vert_data:
            if ent in vert_data[ar]:
                A = q_embeddings[q_entities.index(ent)]
                B = globals()[f"A{ar}"][vert_data[ar].index(ent)]
                cs = round(float(cosine_similarity(A.reshape(-1,1).T,B.reshape(-1,1).T)),6)
                if ar not in rank.keys():
                    rank[ar] = [cs]
                else:
                    rank[ar].append(cs)
    
    if rank == {}:
        return []


    # Sum of every rank
    rank_sum = {}
    for i in rank:
        rank_sum[i] = sum(rank[i])
    return T5(rank_sum)

darts = solutionD(news)
print('$'+"$".join(darts))