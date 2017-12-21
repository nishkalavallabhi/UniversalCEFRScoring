from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, merge
from keras.layers import Embedding, Convolution1D, MaxPooling1D
from keras.layers import AveragePooling1D, LSTM, GRU
from keras.utils import np_utils
import sys, time, re, glob
from collections import defaultdict
from gensim.utils import simple_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

_white_spaces = re.compile(r"\s\s+")

maxchars = 200
embedding_dims = 100
batch_size = 32
nb_epoch = 10
nb_filter = 128
filter_length = 5
pool_length = 32
minfreq = 0
data_path = sys.argv[1]
minwordfreq = 15
maxwordlen = 400
seed = 1234

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def read_data():
    labels = []
    documents = []
    for data_file in glob.iglob(sys.argv[1]+"/*"):
        doc = open(data_file, "r").read().strip()
        wrds = doc.split(" ")
        label = data_file.split("/")[-1].split(".txt")[0].split("_")[-1]
        if label == "EMPTY": continue
        if len(wrds) >= maxwordlen:
            doc = " ".join(wrds[:maxwordlen])
        doc = _white_spaces.sub(" ", doc)
        labels.append(label)
        documents.append(doc)
        
    return (documents, labels)

def char_tokenizer(s):
    return list(s)

def word_tokenizer(s):
    return simple_preprocess(s)

def getWords(D):
    wordSet = defaultdict(int)
    max_features = 3
    for d in D:
        for c in word_tokenizer(d):
            wordSet[c] += 1
    for c in wordSet:
        if wordSet[c] > minwordfreq:
            max_features += 1
    return wordSet, max_features

def getChars(D):
    charSet = defaultdict(int)
    max_features = 3
    for d in D:
        for c in char_tokenizer(d):
            charSet[c] += 1
    for c in charSet:
        if charSet[c] > minfreq:
            max_features += 1
    return charSet, max_features

def transform(D, vocab, minfreq, tokenizer="char"):
    features = defaultdict(int)
    count = 0
    for i, k in enumerate(vocab.keys()):
        if vocab[k] > minfreq:
            features[k] = count
            count += 1
    
    start_char = 1
    oov_char = 2
    index_from = 3
    
    X = []
    for j, d in enumerate(D):
        x = [start_char]
        z = None
        if tokenizer == "word":
            z = word_tokenizer(d)
        else:
            z = char_tokenizer(d)
        for c in z:
            freq = vocab[c]
            if c in vocab:
                if c in features:
                    x.append(features[c]+index_from)
                else:
                    x.append(oov_char)
            else:
                continue
        X.append(x)
    return X
    
print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.time()
doc_train, y_labels = read_data()
print(time.time() - pt)

print("Transforming the datasets... ", end="")
sys.stdout.flush()
pt = time.time()

word_vocab, max_word_features = getWords(doc_train)
print("Number of features= ", max_word_features)
x_word_train = transform(doc_train, word_vocab, minwordfreq, tokenizer="word")

print(len(x_word_train), 'train sequences')

print(time.time() - pt)

print('Pad sequences (samples x time)')
x_word_train = sequence.pad_sequences(x_word_train, maxlen=maxwordlen)
print('x_train shape:', x_word_train.shape)


print("Transforming the labels... ", end="")
sys.stdout.flush()
pt = time.time()
unique_labels = list(set(y_labels))
print("Class labels = ",unique_labels)
n_classes = len(unique_labels)
indim = x_word_train.shape[1]

y_labels = [unique_labels.index(y) for y in y_labels]

y_train = np_utils.to_categorical(np.array(y_labels), len(unique_labels))

print('y_train shape:', y_train.shape)

print(time.time() - pt)

cv_accs, cv_f1 = [], []
k_fold = StratifiedKFold(10, random_state=seed)
all_gold = []
all_preds = []
for train, test in k_fold.split(x_word_train, y_labels):
    #print("TRAIN:", train, "TEST:", test)
    print('Build model...')

    model = Sequential()
    model.add(Embedding(max_word_features, embedding_dims, input_length=maxwordlen))
    #model.add(GRU(50))
    #model.add(AveragePooling1D(pool_length=8))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])


    model.fit(x_word_train[train], y_train[train],
              batch_size=batch_size,
              epochs=nb_epoch)

    y_pred = model.predict_classes(x_word_train[test])
    #print(y_pred, np.array(y_labels)[test], sep="\n")

    pred_labels = [unique_labels[x] for x in y_pred]
    gold_labels = [unique_labels[x] for x in np.array(y_labels)[test]]
    all_gold.extend(gold_labels)
    all_preds.extend(pred_labels)

    cv_f1.append(f1_score(np.array(y_labels)[test], y_pred, average="weighted"))
    print(confusion_matrix(gold_labels, pred_labels, labels=unique_labels))

print("\nF1-scores", cv_f1,sep="\n")
print("Average F1 scores", np.mean(cv_f1))
print(confusion_matrix(all_gold,all_preds))
