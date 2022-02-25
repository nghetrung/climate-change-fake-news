import json
import numpy as np
import nltk
nltk.download('stopwords')

# create corpus of articles
with open('train.json', 'r') as f:
    data = json.load(f)

corpus = []
label = []
for i in data:
    corpus.append(data[i]['text'])
    label.append(data[i]['label'])
len(corpus)

# additional data
with open('train_real_1.json', 'r') as f:
    data_real = json.load(f)

for i in data_real:
    corpus.append(data_real[i]['text'])
    label.append(data_real[i]['label'])
len(corpus)


# loading dev data
with open('dev.json', 'r') as f1:
    dev_data = json.load(f1)
dev_corpus = []
dev_label = []
for i in dev_data:
    dev_corpus.append(dev_data[i]['text'])
    dev_label.append(dev_data[i]['label'])
len(dev_corpus)




# Preprocessing text
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from nltk.corpus import wordnet
from nltk import word_tokenize
nltk.download('words')
stop = stopwords.words('english') + list(string.punctuation)


from nltk.tag import pos_tag
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

import re
def hasNumbers(string):
    return bool(re.search(r'\d', string))

def isUrl(string):
    a = re.search('[https:]?[//]?[/]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ])+', string)
    if a is None:
        return False
    else:
        if len(string) < 12:
            return False
        return True

def my_tokenizer(doc):
    tokens = word_tokenize(doc)
    # lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    cleaned_tokens = []
    for i in tokens:
        i_lower = i.lower()
        if i_lower not in stop:
            if len(i_lower) == 1 and i_lower.isdigit():
                cleaned_tokens.append(i)
            elif len(i_lower) > 1:
                if not i_lower.isalnum():
                    if hasNumbers(i_lower):
                        if not isUrl(i_lower):
                            cleaned_tokens.append(i)
                else:
                    cleaned_tokens.append(i)
                # i = i.lower()
                #lemma = lemmatizer.lemmatize(i, get_wordnet_pos(i))
                # if (lemma not in stop) and len(lemma) > 1 and lemma.isalnum():
                    # cleaned_tokens.append(lemma) 
    cleaned_doc = " ".join(cleaned_tokens)   
    return cleaned_doc




from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers

preprocessed_corpus = []
for i, doc in enumerate(corpus):
    preprocessed_corpus.append(my_tokenizer(doc))

preprocessed_dev_corpus = []
for i, doc in enumerate(dev_corpus):
    preprocessed_dev_corpus.append(my_tokenizer(doc))


max_features = 20000 
tokenizer = Tokenizer(num_words=10000, lower=False)
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(preprocessed_corpus)
list_tokenized_train = tokenizer.texts_to_sequences(preprocessed_corpus)
list_tokenized_test = tokenizer.texts_to_sequences(preprocessed_dev_corpus)
len(list_tokenized_train[2000])
vocab_size = len(tokenizer.word_index) + 1

for word, index in tokenizer.word_index.items():
    if (word in stop) or (len(word) < 2):
        print(word, index)


maxlen = 3503 # length for each document input
X_train = pad_sequences(list_tokenized_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(list_tokenized_test, padding='post', maxlen=maxlen)

# GloVe
embeddings_index = dict()
embedding_dim = 300
f = open('glove.42B.300d.txt', encoding="utf8")
for line in f:
    #split up line into an indexed array
    values = line.split()
    #first index is word
    word = values[0]
    #store the rest of the values in the array as a new array
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs 
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
embeddings_index['climate']


all_embs = np.stack(list(embeddings_index.values()))
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, embedding_dim))
embedding_matrix = np.zeros((vocab_size, embedding_dim))
embeddedCount = 0
for word, i in tokenizer.word_index.items():
    #i-=1
    #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
    embedding_vector = embeddings_index.get(word)
    #and store inside the embedding matrix that we will train later on.
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
        embeddedCount+=1
print('total embedded:',embeddedCount,'common words')

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / vocab_size


# Word2Vec GoogleNews
from gensim.models import KeyedVectors
word2vecDict = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
embeddings_index = dict()
embedding_dim = 300

for word in word2vecDict.wv.vocab:
    embeddings_index[word] = word2vecDict.word_vec(word)
print('Loaded %s word vectors' % len(embeddings_index))

all_embs = np.stack(list(embeddings_index.values()))
emb_mean, emb_std = all_embs.mean(), all_embs.std()
embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, embedding_dim))
embedding_matrix = np.zeros((vocab_size, embedding_dim))

embeddedCount = 0
uncommon_words = []
for word, i in tokenizer.word_index.items():
    #i-=1
    #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
    embedding_vector = embeddings_index.get(word)
    #and store inside the embedding matrix that we will train later on.
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
        embeddedCount+=1
    else:
        if hasNumbers(word):
            embedding_matrix[i] = embeddings_index.get('NUMBER')
        else:
            embedding_matrix[i] = embeddings_index.get('UNKNOWN')
        uncommon_words.append(word)
print('total embedded:',embeddedCount,'common words')

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / vocab_size


# NN model
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))
# model.add(Bidirectional(LSTM(30, return_sequences=True, name='lstm_layer', dropout=0.1, recurrent_dropout=0.1)))
# model.add(layers.GlobalMaxPooling1D())
#model.add(layers.MaxPooling1D())
# model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(100, activation='relu'))
#model.add(layers.MaxPooling1D())
model.add(Dropout(0.1))
model.add(layers.Dense(50, activation='relu'))
# model.add(layers.MaxPooling1D())
model.add(Dropout(0.1))
model.add(layers.Flatten()) # add if using MaxPooling (not Global)
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(X_train, label, batch_size=4, epochs=6, verbose=1)
predictions = model.predict_classes(X_test)

from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
f1_score(dev_label, predictions)
precision_score(dev_label, predictions)
recall_score(dev_label, predictions)
accuracy_score(dev_label, predictions)

# try improve preprocessing step (uncommon words between our tokenizer.word_index vocab and the word2vec vocab)
# increase maxlen, add more max pooling layer
for index, tokens in enumerate(list_tokenized_train):
    if len(tokens) > 4000:
        print(index)
count = 100
for word, index in tokenizer.word_index.items():
    if count > 0:
        print(word, index)
    count -= 1


# predict test set
with open('test-unlabelled.json', 'r') as f:
    test_data = json.load(f)
test_corpus = []
for i in test_data:
    test_corpus.append(test_data[i]['text'])
len(test_corpus)

preprocessed_test_corpus = []
for i, doc in enumerate(test_corpus):
    preprocessed_test_corpus.append(my_tokenizer(doc))

list_tokenized = tokenizer.texts_to_sequences(preprocessed_test_corpus)
X_test_unlabelled = pad_sequences(list_tokenized, padding='post', maxlen=maxlen)
predictions_submit = model.predict_classes(X_test_unlabelled)

test_preds={}
for index, pred in enumerate(predictions_submit):
    test_preds['test-'+str(index)] = {'label': int(pred)}

with open('test-output.json', 'w') as r:  
    json.dump(test_preds, r)