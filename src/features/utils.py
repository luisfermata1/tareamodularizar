import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from src.data import nlp

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Eliminar stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Hacer bigrams
def make_bigrams(bigram_mod, texts):
    return [bigram_mod[doc] for doc in texts]

# Hacer trigrams
def make_trigrams(trigram_mod, bigram_mod, texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Lematización basada en el modelo de POS de Spacy
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def sent_to_words(sentences):
    for sentence in sentences:
        # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True elimina la puntuación