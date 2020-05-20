import pathlib
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import src.file_manager as fm
from sklearn import metrics
import src.file_manager as fm

__all__ = ["display_benchmark", "confusion_matrix"]

def display_lines_length(data):
    dictionary = {}
    for line in data['lines']:
        length = len(line.split(' '))
        count = dictionary.get(length, 0) + 1
        dictionary[length] = count
    x = [pair[0] for pair in dictionary.items()]
    y = [pair[1] for pair in dictionary.items()]
    plt.bar(x, y)
    plt.show()


def lines_per_character(data):
    dictionary = {}
    for rank in data['character']:
        count = dictionary.get(rank, 0) + 1
        dictionary[rank] = count
    x = [pair[0] for pair in dictionary.items()]
    y = [pair[1] for pair in dictionary.items()]
    plt.xlabel("Character")
    plt.ylabel("Amount of lines")
    plt.bar(x, y)
    plt.show()

extraction_techniques = ["fasttext", "tfidf", "word2vec"]
benchmark_techniques = ["confidence_character", "confidence_predicted", "cross_entropy_loss"]

def display_benchmark(benchmark_type="confidence_character"):
    data_array = []
    for technique in extraction_techniques:
        data_array.append((technique, fm.get_df("3_benchmark_" + technique)))
    for (name, dataset) in data_array:
        x = dataset.index
        if benchmark_type == "confidence_character":
            y = dataset["average_confidence"]["character"]
            plt.ylabel("Confidence in character")
        elif benchmark_type == "confidence_predicted":
            y = dataset["average_confidence"]["predicted"]
            plt.ylabel("Confidence in prediction")
        else:
            y = dataset["cross_entropy"]["cross_entropy_loss"]
            plt.ylabel("Cross entropy loss")
        plt.scatter(x, y)
    plt.xlabel("Words per line")
    labels = [name for (name, _) in data_array]
    plt.legend(labels)
    plt.show()


def confusion_matrix(type="tfidf"):
    print("Confusion matrix for " + type)
    data = fm.get_df("2_classified_" + type)
    parsed_character = data["parsed"]["character"].tolist()
    classified_character = data["classified"]["character"].tolist()
    print(metrics.classification_report(parsed_character, classified_character))
    print(metrics.confusion_matrix(parsed_character, classified_character))

def display_rank(data_array):
    width = 1.0/len(data_array)-0.2 #create space around bars
    test = 0
    for data in data_array:
        dictionary = {}
        for rank in data[0]['rank']:
            count = dictionary.get(rank, 0) + 1
            dictionary[rank] = count
        x = [pair[0] for pair in dictionary.items()]
        y = [pair[1] for pair in dictionary.items()]
        r = [cur + width*(test-0.5) for cur in x]
        print(x)
        print(r)
        plt.xlabel("Rank of character")
        plt.ylabel("Amount")
        plt.bar(r, y, width=width)
        test += 1
    labels = [cur[1] for cur in data_array]
    plt.legend(labels)
    plt.show()