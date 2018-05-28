import os
import json
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
from textblob.classifiers import NLTKClassifier
import time
import nltk.classify
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
import re

import nltk
nltk.download('punkt')

class SVMClassifier(NLTKClassifier):
    #"""Class that wraps around nltk.classify module for SVM Classifier"""

    nltk_class = nltk.classify.SklearnClassifier(LinearSVC())

class RocchioClassifier(NLTKClassifier):
    """Class that wraps around nltk.classify module for SVM Classifier"""

    nltk_class = nltk.classify.SklearnClassifier(NearestCentroid())

class NNClassifier(NLTKClassifier):
    """Class that wraps around nltk.classify module for SVM Classifier"""

    nltk_class = nltk.classify.SklearnClassifier(KNeighborsClassifier())

def list_files_from_directory(directory):
    """Lists all file paths from given directory"""

    ret_val = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            ret_val.append(str(directory) + "/" + str(file))
    return ret_val


def read_file(path):
    """Reads all lines from file on given path"""

    f = open(path, "r")
    read = f.readlines()
    ret_val = []
    for line in read:
        if line.startswith("#"):
            pass
        else:
            ret_val.append(line)
    return ret_val


def process_line(line):
    """Returns sentence category and sentence in given line"""

    if "\t" in line:
        splits = line.split("\t")
        s_category = splits[0]
        sentence = splits[1].lower()
        for sw in stopwords:
            sentence = sentence.replace(sw, "")
        pattern = re.compile("[^\w']")
        sentence = pattern.sub(' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return s_category, sentence
    else:
        splits = line.split(" ")
        s_category = splits[0]
        sentence = line[len(s_category)+1:].lower()
        for sw in stopwords:
            sentence = sentence.replace(sw, "")
        pattern = re.compile("[^\w']")
        sentence = pattern.sub(' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return s_category, sentence


def create_json_file(input_folder, destination_file):
    """Writes training data from given folder into formatted JSON file"""

    tr_folder = list_files_from_directory(input_folder)
    all_json = []
    for file in tr_folder:
        lines = read_file(file)
        for line in lines:
            c, s = process_line(line)
            if s.endswith('\n'):
                s = s[:-1]
            json_data = {
                'text': s,
                'label': c
            }
            all_json.append(json_data)

    with open(destination_file, "w") as outfile:
        json.dump(all_json, outfile)


def prepare_test_data(input_folder):
    """Maps each sentence to it's category"""

    test_folder = list_files_from_directory(input_folder)
    t_sentences = []
    t_categories = []
    for file in test_folder:
        lines = read_file(file)
        for line in lines:
            c, s = process_line(line)
            if s.endswith('\n'):
                s = s[:-1]
            t_sentences.append(s)
            t_categories.append(c)
    return t_categories, t_sentences

# main

# loading stopwords
input_stopwords = read_file("C:/Users/jabid/Desktop/SentenceCorpus/SentenceCorpus/word_lists/stopwords.txt")
stopwords = []
for word in input_stopwords:
    if word.endswith('\n'):
        word = word[:-1]
        stopwords.append(word)


# prepare training and test data
create_json_file("training_set", "training.json")
categories, sentences = prepare_test_data("test_set")

# Bayes Classifier
print("Training Naive Bayes Classifier...")
start_nbc = time.time()
with open('training.json', 'r') as training:
    nbc = NaiveBayesClassifier(training, format="json")
stop_nbc = time.time()
print("Training Naive Bayes Classifier completed...")
elapsed = stop_nbc - start_nbc
print("Training time (in seconds): " + str(elapsed))
print("Testing Naive Bayes Classifier...")
correct = 0
start_nbc = time.time()
for i in range(0, len(sentences)):
    category = str(nbc.classify(sentences[i])).lower()
    expected = str(categories[i]).lower()
    if category == expected:
        correct += 1
stop_nbc = time.time()
elapsed = stop_nbc - start_nbc
print("Number of tests: " + str(len(sentences)))
print("Correct tests: " + str(correct))
accuracy = correct / len(sentences)
print("Naive Bayes Classifier accuracy: " + str(accuracy))
print("Testing time (in seconds): " + str(elapsed))

# Rocchio
print("Training Rocchio Classifier...")
start_nbc = time.time()
with open('training.json', 'r') as training:
    nbc = RocchioClassifier(training, format="json")
stop_nbc = time.time()
print("Training Rocchio Classifier completed...")
elapsed = stop_nbc - start_nbc
print("Training time (in seconds): " + str(elapsed))
print("Testing Rocchio Classifier...")
correct = 0
start_nbc = time.time()
for i in range(0, len(sentences)):
    category = str(nbc.classify(sentences[i])).lower()
    expected = str(categories[i]).lower()
    if category == expected:
        correct += 1
stop_nbc = time.time()
elapsed = stop_nbc - start_nbc
print("Number of tests: " + str(len(sentences)))
print("Correct tests: " + str(correct))
accuracy = correct / len(sentences)
print("Rocchio Classifier accuracy: " + str(accuracy))
print("Testing time (in seconds): " + str(elapsed))

# Knearest

print("Training Knearest Classifier...")
start_nbc = time.time()
with open('training.json', 'r') as training:
    nbc = NNClassifier(training, format="json")
stop_nbc = time.time()
print("Training Knearest Classifier completed...")
elapsed = stop_nbc - start_nbc
print("Training time (in seconds): " + str(elapsed))
print("Testing Knearest Classifier...")
correct = 0
start_nbc = time.time()
for i in range(0, len(sentences)):
    category = str(nbc.classify(sentences[i])).lower()
    expected = str(categories[i]).lower()
    if category == expected:
        correct += 1
stop_nbc = time.time()
elapsed = stop_nbc - start_nbc
print("Number of tests: " + str(len(sentences)))
print("Correct tests: " + str(correct))
accuracy = correct / len(sentences)
print("Knearest Classifier accuracy: " + str(accuracy))
print("Testing time (in seconds): " + str(elapsed))