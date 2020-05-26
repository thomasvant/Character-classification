import pathlib
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import src.file_manager as fm
from sklearn import metrics
import src.file_manager as fm
import src

__all__ = ["confusion_matrix", "display_benchmark_per_wordcount"]


extraction_techniques = ["fasttext", "tfidf"]
benchmarks = ["accuracy","cross_entropy","predict_proba_predicted_character"]


def display_benchmark_per_wordcount(min_wordcount=False, benchmark="accuracy"):
    if min_wordcount:
        data = fm.get_df("3_benchmark_per_min_wordcount")
    else:
        data = fm.get_df("3_benchmark_per_wordcount")
    x = data.index
    for extraction in extraction_techniques:
        y = data[extraction][benchmark]
        plt.scatter(x, y)
    if min_wordcount:
        plt.xlabel("Min wordcount")
    else:
        plt.xlabel("Wordcount")
    plt.ylabel(benchmark)
    plt.legend(extraction_techniques)
    plt.show()


def confusion_matrix(technique="tfidf"):
    print("Confusion matrix for " + technique)
    data = fm.get_df("2_classified_" + technique)
    parsed_character = data["parsed"]["character"].tolist()
    classified_character = data["classified"]["character"].tolist()
    # print(metrics.classification_report(parsed_character, classified_character))
    cm = metrics.confusion_matrix(parsed_character, classified_character)
    print(cm)
    print(np.round(cm / cm.astype(np.float).sum(axis=1),2))