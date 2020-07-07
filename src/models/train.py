import re
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from src.features.utils import remove_stopwords, make_bigrams, make_trigrams, lemmatization, sent_to_words


df = pd.read_csv('../../data/raw/reviews.csv')

# Convertir a una lista
data = df.content.values.tolist()
pprint(data[:1])

# Eliminar emails
data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]

# Eliminar newlines
data = [re.sub(r'\s+', ' ', sent) for sent in data]

# Eliminar comillas
data = [re.sub(r"\'", "", sent) for sent in data]


data_words = list(sent_to_words(data))

# Construimos modelos de bigrams y trigrams
# https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
bigram = gensim.models.Phrases(data_words, min_count=5)
trigram = gensim.models.Phrases(bigram[data_words])

# Aplicamos el conjunto de bigrams/trigrams a nuestros documentos
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)



# Eliminamos stopwords
data_words_nostops = remove_stopwords(data_words)

# Formamos bigrams
data_words_bigrams = make_bigrams(bigram_mod, data_words_nostops)

# python3 -m spacy download en_core_web_lg
# Inicializamos el modelo 'en_core_web_lg' con las componentes de POS únicamente
#nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

# Lematizamos preservando únicamente noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Creamos diccionario
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

doc_lda = lda_model[corpus]

#Perplejidad
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Score de coherencia
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualizamos los temas
#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#vis

