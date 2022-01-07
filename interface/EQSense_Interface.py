import streamlit as st
import pandas as pd
import time
import os,pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Digital Legislation",
    page_icon="📱"
)

st.title("Digital Legislation Interface")
news = st.text_area("Please provide news article")
method = st.selectbox("Select a method:",['Word2Vec','BERT','pyRDF2Vec','RDF KG','Ensemble'])

### Function definitions starts ###
def w2v():
    status.markdown("**Running Word2Vec Algorithm...**")
    from nltk.stem import PorterStemmer

    f = open("word2vec/data_emb.pkl", "rb")
    data_emb = pickle.load(f)
    f.close()

    f = open("word2vec/df_word2vec.pkl", "rb")
    df = pickle.load(f)
    f.close()

    from gensim.models import Word2Vec
    model_sb = Word2Vec.load("word2vec/modelSB0_300")

    import string
    def remPunct(text):
        text = text.replace("(", " ").replace(")", " ")
        ans = "".join([c.lower() for c in text if c not in string.punctuation + "—"])
        return ans

    def remStop(text):
        ans = [i for i in text.split() if i not in stopwords.words('english')]
        return " ".join(ans)

    def Stemmer(text):
        ans = [PorterStemmer().stem(i) for i in text.split()]
        return " ".join(ans)

    def EQSense(sent, emb=data_emb, model=model_sb, df=df):
        sent = Stemmer(remStop(remPunct(sent)))
        buff = sent.split()
        query = np.zeros(emb.shape[1])

        c = 0
        for j in buff:
            try:
                query += model_sb.wv[j]
                c += 1
            except:
                pass
        if query.all() == np.zeros(emb.shape[1]).all():
            return []
        query = query / c

        cos_sim = []
        for i in emb:
            cos_sim.append(float(cosine_similarity(query.reshape(-1, 1).T, i.reshape(-1, 1).T)))
        ans = []
        for i in range(5):
            ind = np.argmax(cos_sim)
            cos_sim[ind] = -9999
            ans.append(df["PART-Article"][ind])
        return [i.split(" - ")[-1] for i in ans]

    darts = EQSense(news)
    status.write("")
    return darts


def p2v():
    status.markdown("**Running pyRDF2Vec Algorithm...**")
    darts = os.popen(f'''python3.8 pyrdfhek.py "{news.replace('`',' ')}"''').read()
    darts = darts.split('$')[-5:]
    status.write("")
    return darts


def rbert():
    status.markdown("**Running BERT Algorithm...**")
    from transformers import RobertaTokenizer, TFRobertaModel
    def roberta(sent):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = TFRobertaModel.from_pretrained('roberta-large')
        encoded_input = tokenizer(sent, return_tensors='tf', max_length=500)
        output = model(encoded_input)
        return np.array(output.pooler_output)[0]

    with open("bert/data_emb.pkl", "rb") as f:
        data_emb = pickle.load(f)
    with open("bert/parsed_articles.pkl", "rb") as f:
        parsed_articles = pickle.load(f)
    df = list(parsed_articles.keys())

    def EQSense(sent, emb=data_emb, df=df):
        query = roberta(sent)

        if query.all() == np.zeros(emb.shape[1]).all():
            return "Constitution was not violated"

        cos_sim = []
        for i in emb:
            cos_sim.append(float(cosine_similarity(query.reshape(-1, 1).T, i.reshape(-1, 1).T)))

        ans = []
        for i in range(5):
            ind = np.argmax(cos_sim)
            cos_sim[ind] = -9999
            ans.append(df[ind])
        return ans

    darts = EQSense(news)
    status.write("")
    return darts


def rkg():
    status.markdown("**Running RDF KG Algorithm...**")
    import pickle
    f = open("rdf/g.pkl", "rb")
    g = pickle.load(f)
    f.close()

    import nltk
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    from nltk.corpus import wordnet
    def get_some_word_synonyms(word):
        word = word.lower()
        synonyms = []
        synsets = wordnet.synsets(word)
        if (len(synsets) == 0):
            return []
        synset = synsets[0]
        lemma_names = synset.lemma_names()
        for lemma_name in lemma_names:
            lemma_name = lemma_name.lower().replace('_', ' ')
            if (lemma_name != word and lemma_name not in synonyms):
                synonyms.append(lemma_name)
        return synonyms

    def findSym(wrd):
        synonyms = []
        for syn in wordnet.synsets(wrd):
            for l in syn.lemmas():
                synonyms.append(l.name().replace("_", " "))
        return list(set(synonyms))

    from pyclausie import ClausIE
    cl = ClausIE.get_instance()

    def tripProv(text):
        try:
            trips = cl.extract_triples([text])
        except:
            return []
        return [[i.subject, i.predicate, i.object] for i in trips]

    import en_core_web_lg
    nlp = en_core_web_lg.load()

    def sentProv(txt):
        doc = nlp(txt)
        ans = []
        for sent in doc.sents:
            ans += tripProv(sent)
        return ans

    def queRier(trips):
        buff = []
        for trip in trips:
            obj1 = wnl.lemmatize(trip[0]).lower()
            obj2 = wnl.lemmatize(trip[2]).lower()
            q1 = '''SELECT ?o  WHERE { "''' + obj1 + '''" ?p ?o }'''
            q2 = '''SELECT ?o  WHERE { "''' + obj2 + '''" ?p ?o }'''
            try:
                buff += [i[0].value for i in g.query(q1)] + [i[0].value for i in g.query(q2)]
            except:
                pass
        ans = []
        for i in buff:
            if "###ART" in i:
                ans.append(i.replace("###ART", ""))
        return ans

    def EQSenseRDF(que):
        adata = queRier(sentProv(que))
        arts, counts = np.unique(adata, return_counts=1)
        ans = []
        for i in sorted(np.unique(counts), reverse=1)[:5]:
            ans += list(arts[np.where(counts == i)[0]])
        return ans[:5]
    darts = EQSenseRDF(news)
    status.write("")
    return darts

def printer(lst):
    if lst == []:
        st.success("Constitution was not violated")
    else:
        st.markdown("**Following articles from the Indian Constitution might be violated:**\n - " + "\n - ".join(lst))
### Function definitions ends   ###
if st.button("submit"):
    status = st.empty()
    if method == 'Word2Vec':
        printer(w2v())
    
    elif method == 'pyRDF2Vec':
        printer(p2v())
    
    elif method == 'BERT':
        printer(rbert())

    elif method == 'RDF KG':
        printer(rkg())

    elif method == "Ensemble":
        rkgo = rkg()
        rberto = rbert()
        p2vo = p2v()
        w2vo = w2v()
        darts = [rkgo[0],rberto[0],p2vo[0],w2vo[0]]
        if darts == []:
            st.success("Constitution was not violated")
        else:
            st.markdown("**Following articles from the Indian Constitution might be violated:**\n - "+"\n - ".join(darts))