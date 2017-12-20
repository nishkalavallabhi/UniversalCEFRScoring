from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, merge
from keras.layers import Embedding, Convolution1D, MaxPooling1D
from keras.layers import AveragePooling1D, LSTM, GRU
from keras.utils import np_utils
import sys, time, re
from collections import defaultdict
from gensim.utils import simple_preprocess

_white_spaces = re.compile(r"\s\s+")
maxlen = 80
maxchars = 200
embedding_dims = 32
batch_size = 32
nb_epoch = 10
nb_filter = 128
filter_length = 5
pool_length = 32
minfreq = 0
data_path = "../data"
minwordfreq = 15
maxwordlen = 70

def read_data(data_file):
    labels = []
    documents = []
    with open(data_file, "r") as fp:
        linenum = 0
        for line in fp:
            if len(line) < 3: # skip empty lines and ^Z at the end
                continue
            doc, label = line.strip().split("\t")
            wrds = doc.split(" ")
            if len(wrds) >70:
                doc = " ".join(wrds[:70])
            doc = _white_spaces.sub(" ", doc)
            labels.append(label)
            documents.append(doc)
            linenum += 1
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

def getVocab(D):
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
doc_train, y_train = read_data(data_path + "/train/task1-train.txt")
print(time.time() - pt)

print("Reading the test set... ", end="")
sys.stdout.flush()
pt = time.time()
doc_test, y_test = read_data(data_path + "/gold/A.txt")
print(time.time() - pt)

print("Transforming the datasets... ", end="")
sys.stdout.flush()
pt = time.time()

word_vocab, max_word_features = getWords(doc_train)
print("Number of features= ", max_word_features)
x_word_train = transform(doc_train, word_vocab, minwordfreq, tokenizer="word")
x_word_test = transform(doc_test, word_vocab, minwordfreq, tokenizer="word")
print(len(x_word_train), 'train sequences')
print(len(x_word_test), 'test sequences')
print(time.time() - pt)

print('Pad sequences (samples x time)')
x_word_train = sequence.pad_sequences(x_word_train, maxlen=maxwordlen)
x_word_test = sequence.pad_sequences(x_word_test, maxlen=maxwordlen)
print('x_train shape:', x_word_train.shape)
print('x_test shape:', x_word_test.shape)

print("Transforming the labels... ", end="")
sys.stdout.flush()
pt = time.time()
unique_labels = list(set(y_test))
print("Class labels = ",unique_labels)
n_classes = len(unique_labels)
indim = x_word_train.shape[1]
y_train = [unique_labels.index(y) for y in y_train]
y_test = [unique_labels.index(y) for y in y_test]
y_train = np_utils.to_categorical(np.array(y_train), len(unique_labels))
y_test = np_utils.to_categorical(np.array(y_test), len(unique_labels))
print(time.time() - pt)

print('Build model...')

model = Sequential()
model.add(Embedding(max_word_features, embedding_dims, input_length=maxwordlen, dropout=0.2))
model.add(AveragePooling1D(pool_length=8))
model.add(Flatten())
#model.add(Dropout(0.5))
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(x_word_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(x_word_test, y_test))
