# best = lg.best_params_
# print("Best parameters: ", best)
# cv_results = lg.cv_results_
# print("CV results: ", cv_results)
# print("STD test: ", cv_results.get("std_test_score")[0])

import src.file_manager as fm
import src
import src.classify as classify
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from copy import deepcopy
from sklearn.model_selection import train_test_split

__all__ = ["benchmark"]

def benchmark(train_or_test="test", random=False, min=2, max=30, folds=5):
    print("Benchmarking " + train_or_test + " data from " + str(min) + " to " + str(max))

    dictionary = {
        "accuracy": [],
        "accuracy_std": [],
        "predict_proba": [],
        "predict_proba_std":[],
        "cross_entropy_loss": [],
        "cross_entropy_loss_std": []
    }
    techniques = {'fasttext': deepcopy(dictionary) , 'tfidf': deepcopy(dictionary)}

    data = src.file_manager.get_df("1_embedded_fasttext")
    wordcount = src.file_manager.get_df("details_min_wordcount")
    hyperparams = src.file_manager.get_df("4_benchmark_change_testing_data_train")

    for cur_technique in techniques.keys():
        train, test = train_test_split(data, random_state=1515, train_size=0.8)
        for min_wordcount in range(min, max):
            print("Min wordcount: ", min_wordcount)
            data_size = wordcount["wordcount"][train_or_test].get(min_wordcount)

            if train_or_test == "test":
                test = test[test["parsed"]["wordcount"] >= min_wordcount]
                C = hyperparams[cur_technique]["C"].get(min)
                max_iter = hyperparams[cur_technique]["max_iter"].get(min)
            if train_or_test == "train":
                train = train[train["parsed"]["wordcount"] >= min_wordcount]
                C = hyperparams[cur_technique]["C"].get(min_wordcount)
                max_iter = hyperparams[cur_technique]["max_iter"].get(min_wordcount)

            accuracies = []
            predict_probas = []
            cross_entropy_losses = []

            for cur_fold in range(0,folds):

                classified, lg = src.classify_std.classify(technique=cur_technique, train_data=train, test_data=test, C=C, max_iter=max_iter)

                accuracies.append(accuracy_score(classified["parsed"]["character"], classified["classified"]["character"]))
                cross_entropy_losses.append(log_loss(classified["parsed"]["character"], classified["predict_proba_"], labels=classified["predict_proba_"].columns))
                predict_probas.append(classified["predict_proba_specific"]["predicted_character"].mean())

            cur_details = techniques.get(cur_technique)
            cur_details.get("accuracy").append(np.mean(accuracies))
            cur_details.get("accuracy_std").append(np.std(accuracies))
            cur_details.get("predict_proba").append(np.mean(predict_probas))
            cur_details.get("predict_proba_std").append(np.std(predict_probas))
            cur_details.get("cross_entropy_loss").append(np.mean(cross_entropy_losses))
            cur_details.get("cross_entropy_loss_std").append(np.std(cross_entropy_losses))

            techniques.update({cur_technique:cur_details})
            print(techniques)
    print(techniques)
    fm.write_df(pd.DataFrame.from_dict(techniques), "STD")
    return techniques