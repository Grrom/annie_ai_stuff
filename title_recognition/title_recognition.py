from nltk.stem.lancaster import LancasterStemmer
import random
import tensorflow
import tflearn
import numpy
import json
import nltk
import pickle


stemmer = LancasterStemmer()


with open("anime_titles.json") as file:
    titles = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []


for title in titles:
    tokenized = nltk.word_tokenize(title)
    words.extend(tokenized)
    docs_x.append(tokenized)
    docs_y.append(title)

    if title not in labels:
        labels.append(title)

words = [stemmer.stem(w.lower()) for w in words if w not in ["?", ":"]]
words = sorted(list(set(words)))

labels = sorted(labels)


training = []
output = []

out_empty = [0 for _ in range(len(titles))]

for x, doc in enumerate(docs_x):
    bag = []
    _words = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in _words:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

with open("anime_titles.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("title_recognition.tflearn")
