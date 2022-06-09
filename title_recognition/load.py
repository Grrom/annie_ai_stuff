import pickle
import json
import random
import tensorflow
import tflearn
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

with open("anime_titles.json") as file:
    data = json.load(file)

with open("anime_titles.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("title_recognition.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Enter message (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        print(results)
        title = labels[results_index]

        print(title)


chat()
