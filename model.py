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


def space_tokenizer(doc):
    # ps = PorterStemmer()
    tokens = word_tokenize(doc)
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    cleaned_tokens = []
    for i in tokens:
        if i not in stop:
            i = i.lower()
            lemma = lemmatizer.lemmatize(i, get_wordnet_pos(i))
            cleaned_tokens.append(lemma)    
    return cleaned_tokens

for index, text in enumerate(corpus):
    corpus[index] = space_tokenizer(text)

for index, text in enumerate(dev_corpus):
    dev_corpus[index] = space_tokenizer(text)


# Bag of word using Count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words=stop, analyzer='word', tokenizer=space_tokenizer, lowercase=True, ngram_range=(1, 1))
bow = vectorizer.fit_transform(corpus)
len(vectorizer.get_feature_names())
vectorizer.get_feature_names()
dev_bow = vectorizer.transform(dev_corpus)


# TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
def identity_tokenizer(text):
    return text
tfidf_vectorizer = TfidfVectorizer(tokenizer=space_tokenizer)
tf_idf = tfidf_vectorizer.fit_transform(corpus)
len(tfidf_vectorizer.get_feature_names())
tfidf_vectorizer.get_feature_names()
dev_bow_tfidf = tfidf_vectorizer.transform(dev_corpus)


# Additional features (sentiments)
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
features_dev = np.empty(shape=(len(dev_corpus), 6))
for index, text in enumerate(dev_corpus):
    temp = []
    blob = TextBlob(text)
    temp.append(blob.sentiment.polarity)
    temp.append(blob.sentiment.subjectivity)
    score = analyser.polarity_scores(text)
    temp.append(score['neg'])
    temp.append(score['neu'])
    temp.append(score['pos'])
    temp.append(score['compound'])
    features_dev[index] = temp

final_features = np.concatenate((word2vec, features), axis=1)
final_dev_features = np.concatenate((word2vec_dev, features_dev), axis=1)

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(final_features)
X_dev = sc.transform(final_dev_features) 


# Word embeddings presentation
# Spacy
import spacy
# python -m spacy download en_core_web_md 
nlp = spacy.load('en_core_web_md')
doc = nlp("This is some text that I am processing with Spacy. My name is Trung")
doc.vector # mean vector of the sentence/doc

word_emb = np.empty(shape=(len(corpus),300))
for row, text in enumerate(corpus):
    cleaned_tokens = space_tokenizer(text)
    cleaned_doc = " ".join(cleaned_tokens)
    article = nlp(cleaned_doc)
    # article = nlp(text)
    word_emb[row] = article.vector

word_emb_dev = np.empty(shape=(len(dev_corpus),300))
for row, text in enumerate(dev_corpus):
    cleaned_tokens = space_tokenizer(text)
    cleaned_doc = " ".join(cleaned_tokens)
    article = nlp(cleaned_doc)
    # article = nlp(text)
    word_emb_dev[row] = article.vector

# Gensim
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vector = model["isn't"]
# model.most_similar('easy')
vectors = [model[x] for x in "This is some text I am processing with Spacy".split(' ')]
np_vectors = np.array(vectors)

word2vec = np.empty(shape=(len(corpus),300))
for row, text in enumerate(corpus):
    vectors = [model[x] for x in space_tokenizer(text) if x in model.vocab]
    vectors = np.array(vectors)
    mean_vector = np.sum(vectors, axis = 0)
    word2vec[row] = mean_vector

word2vec_dev = np.empty(shape=(len(dev_corpus),300))
for row, text in enumerate(dev_corpus):
    vectors = [model[x] for x in space_tokenizer(text) if x in model.vocab]
    vectors = np.array(vectors)
    mean_vector = np.sum(vectors, axis = 0)
    word2vec_dev[row] = mean_vector


# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(word2vec)
X_dev = sc.transform(word2vec_dev)


# One class SVM
from sklearn.svm import OneClassSVM
clf = OneClassSVM(nu=0.5) # 0.4 best for word embed, 0.55 for tf_idf, 0.3 for BOW, 0.5 for gensim word2vec 
clf.fit(bow)
clf.fit(tf_idf)
clf.fit(word_emb)
clf.fit(word2vec)

predictions = clf.predict(dev_bow_tfidf)
predictions = clf.predict(word_emb_dev)
predictions = clf.predict(word2vec_dev)
predictions
predictions[predictions == -1] = 0
predictions
dev_label = np.array(dev_label, dtype='int64')
dev_label


# Isolation forest
from sklearn.ensemble import IsolationForest
clf_iso_forest = IsolationForest(contamination=0.3)
clf_iso_forest.fit(tf_idf)
clf_iso_forest.fit(word2vec)

predictions = clf_iso_forest.predict(dev_bow_tfidf) # isolation forest perform very poor on tf_idf
predictions = clf_iso_forest.predict(word2vec_dev)
predictions
predictions[predictions == -1] = 0  ### So maybe oneclass classification is not appropriate for this problem since this is actually
                                    ### not an outlier problem. Proportion of real vs fake news might be equal. Or real news actually happens more than fake
                                    ### so fake news here should be the outlier but we only have data for fake


# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf_nb = MultinomialNB(alpha=0.1)
clf_nb.fit(tf_idf, label)
predictions = clf_nb.predict(dev_bow_tfidf)

# Logistic regression
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(C=10000, random_state=123, max_iter=5000, solver='liblinear')
clf_lr.fit(tf_idf, label)
predictions = clf_lr.predict(dev_bow_tfidf)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(n_estimators=200, criterion='entropy', verbose=1, n_jobs=-1, random_state=0)
clf_forest.fit(tf_idf, label)
predictions = clf_forest.predict(dev_bow_tfidf)



# Try autoencoder (neural net) for one class model
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

input_dim = 300
encoding_dim = 300
hidden_dim = int(encoding_dim / 2)

nb_epoch = 50
batch_size = 16
learning_rate = 0.1

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='tanh', activity_regularizer=regularizers.l2('10e-5'))(input_layer)
encoder = Dense(hidden_dim, activation='relu')(encoder)
decoder = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss="binary_crossentropy")
autoencoder.fit(word2vec, word2vec, epochs=nb_epoch, batch_size=batch_size, verbose=1)
# get error of the input
predicted = autoencoder.predict(word2vec)
np.mean(np.power(word2vec - predicted, 2), axis=1) # small error since we trained on them
# now predict dev
predicted_dev = autoencoder.predict(word2vec_dev)
predictions = np.mean(np.power(word2vec_dev - predicted_dev, 2), axis=1)
predictions[predictions >= 0.01] = 0
predictions[predictions != 0 ] = 1


# Feed forward NN
from keras.models import Sequential
# Initialising the ANN
nn_classifier = Sequential()

# Adding the input layer and the first hidden layer
# nn_classifier.add(Dense(activation="relu", kernel_initializer="uniform", units=50000, input_dim=X_train_neural.shape[1]))
nn_classifier.add(Dense(activation="relu", kernel_initializer="uniform", units=1000, input_dim=300))
nn_classifier.add(Dropout(0.1))

# Optional (adding a second hidden layer)
nn_classifier.add(Dense(activation="relu", kernel_initializer="uniform", units=500))
nn_classifier.add(Dropout(0.1))

# Adding the output layer
nn_classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
nn_classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fit the ANN to the training set
nn_classifier.fit(word_emb, label, batch_size=8, epochs=25, verbose=1)

predictions = nn_classifier.predict_classes(word_emb_dev)
predictions_nn = []
for i in predictions:
    predictions_nn.append(i[0])



# Train with embedding layer





# evaluate
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score
f1_score(dev_label, predictions)
precision_score(dev_label, predictions)
accuracy_score(dev_label, predictions)


# write dev predictions to json
preds = {}
for index, pred in enumerate(predictions):
    preds['dev-'+str(index)] = {'label': int(pred)}

with open('results.json', 'w') as r:  
    json.dump(preds, r)



# predict test set
with open('test-unlabelled.json', 'r') as f:
    test_data = json.load(f)
test_corpus = []
for i in test_data:
    test_corpus.append(test_data[i]['text'])
len(test_corpus)

for index, text in enumerate(test_corpus):
    test_corpus[index] = space_tokenizer(text)

test_preds = {}
test_tfidf = tfidf_vectorizer.transform(test_corpus)
test_predictions = clf_lr.predict(test_tfidf)
# test_predictions[test_predictions == -1] = 0 # for oneclass method

test_word2vec = np.empty(shape=(len(test_corpus),300))
for row, text in enumerate(test_corpus):
    vectors = [model[x] for x in space_tokenizer(text) if x in model.vocab]
    vectors = np.array(vectors)
    mean_vector = np.sum(vectors, axis = 0)
    test_word2vec[row] = mean_vector

X_test = sc.transform(test_word2vec)

test_predictions = nn_classifier.predict_classes(test_word2vec)
predictions_nn = []
for i in test_predictions:
    predictions_nn.append(i[0])


for index, pred in enumerate(test_predictions):
    test_preds['test-'+str(index)] = {'label': int(pred)}

with open('test-output.json', 'w') as r:  
    json.dump(test_preds, r)