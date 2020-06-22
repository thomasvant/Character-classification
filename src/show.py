import pathlib
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import src.file_manager as fm
from sklearn import metrics
import src.file_manager as fm
import src
import seaborn as sn

plt.figure(dpi=200)

__all__ = ["confusion_matrix", "display_benchmark_per_wordcount", "display_changing_dataset_benchmark"]


extraction_techniques = {"fasttext":"fastText", "tfidf":"TF-IDF"}
benchmarks = {
    "accuracy": "Accuracy",
    "cross_entropy": "Cross entropy loss",
    "predict_proba_predicted_character": "Probability of classified character"
}


def display_benchmark_per_wordcount(min_wordcount=False, grid=False, benchmark="accuracy"):
    if min_wordcount:
        data = fm.get_df("3_benchmark_per_min_wordcount" + ("_grid" if grid else ""))
    else:
        data = fm.get_df("3_benchmark_per_wordcount")
    x = data.index
    for extraction in extraction_techniques.keys():
        y = data[extraction][benchmark]
        plt.scatter(x, y)
    if min_wordcount:
        plt.xlabel("Min wordcount")
    else:
        plt.xlabel("Wordcount")
    plt.ylabel(benchmark)
    plt.figure(dpi=200)

    plt.legend(extraction_techniques.values())
    plt.show()


def display_changing_dataset_benchmark(benchmark="accuracy", random=False, test_or_train="test"):
    data = src.file_manager.get_df("4_benchmark_change_testing_data_" + test_or_train + ("_random" if random else ""))
    x = data.index
    for extraction in extraction_techniques.keys():
        y = data[extraction][benchmark]
        plt.scatter(x, y)
    plt.ylabel(benchmarks.get(benchmark))
    if random:
        plt.xlabel("Randomly selected " + test_or_train + " data of equal size to setting a minimum wordcount")
    else:
        plt.xlabel("Minimum wordcount set on " + test_or_train + " data")
    plt.legend(extraction_techniques.values())
    plt.savefig('temp.png', dpi=200)
    plt.show()


def confusion_matrix(technique="tfidf"):
    print("Confusion matrix for " + technique)
    data = fm.get_df("2_classified_" + technique)
    labels = data["predict_proba_"].columns
    parsed_character = data["parsed"]["character"].tolist()
    classified_character = data["classified"]["character"].tolist()
    # print(metrics.classification_report(parsed_character, classified_character))
    cm = metrics.confusion_matrix(parsed_character, classified_character, labels=labels)
    print(cm)
    # print(np.round(cm / cm.astype(np.float).sum(axis=1),2))
    labels = data["predict_proba_"].columns.str.capitalize()
    sn.heatmap(cm, annot=True,cmap='Greens', fmt='g', xticklabels=labels, yticklabels=labels, vmin=100, vmax=700)
    plt.xlabel("Actual character")
    plt.ylabel("Predicted character")
    # tikzplotlib.save("confusion_matrix_fasttext.pgf")
    plt.savefig('temp.png')
    plt.show()