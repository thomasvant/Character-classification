import src.file_manager as fm
import src
import src.classify as classify
from sklearn import metrics
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from copy import deepcopy
from sklearn.model_selection import train_test_split

sim_types = ['fasttext', 'tfidf']

__all__ = ["benchmark_change_data"]


def benchmark_change_data(train_or_test="test", random=False, grid=False, min=2, max=30):
    print("Benchmarking using new method from " + str(min) + " to " + str(max))

    if not grid:
        dictionary = {
            "accuracy": [],
            "cross_entropy": [],
            "predict_proba_predicted_character": [],
        }
    else:
        dictionary = {
            "accuracy": [],
            "cross_entropy": [],
            "predict_proba_predicted_character": [],
            "C":[],
            "max_iter":[]
        }
    fasttext_dict = deepcopy(dictionary)
    tfidf_dict = deepcopy(dictionary)

    data = src.file_manager.get_df("1_embedded_fasttext")

    train, test = train_test_split(data, random_state=1515, train_size=0.8)

    test_count = {}
    train_count = {}
    for i in range(min, max):
        test_count.update({i: test[test["parsed"]["wordcount"] > i].count()["parsed"]["wordcount"]})
        train_count.update({i: train[train["parsed"]["wordcount"] > i].count()["parsed"]["wordcount"]})

    # Train shrinks, data needs to be classified only once
    if train_or_test == "test":
        classified_data, params = src.classify(technique="fasttext", train_data=train, test_data=test, unique=False, C=10.0, max_iter=200, write=False)
    for min_wordcount in range(min, max):
        print(min_wordcount)
        if random:
            if train_or_test == "test":
                classified_data = classified_data.sample(n=test_count.get(min_wordcount))
            else:
                train = train.sample(n=train_count.get(min_wordcount))
                classified_data, params = src.classify(technique="fasttext", train_data=train, test_data=test, unique=False,
                                                       grid=grid, write=False)
        else:
            if train_or_test == "test":
                classified_data = classified_data[classified_data["parsed"]["wordcount"] >= min_wordcount]
            else:
                train = train[train["parsed"]["wordcount"] >= min_wordcount]
                classified_data, params = src.classify(technique="fasttext", train_data=train, test_data=test, unique=False,
                                                       grid=grid, write=False)
        labels = classified_data["predict_proba_"].columns
        fasttext_dict.get("accuracy").append(
            accuracy_score(classified_data["parsed"]["character"], classified_data["classified"]["character"]))
        fasttext_dict.get("cross_entropy").append(
            log_loss(classified_data["parsed"]["character"], classified_data["predict_proba_"], labels=labels))
        fasttext_dict.get("predict_proba_predicted_character").append(
            classified_data["predict_proba_specific"]["predicted_character"].mean())
        if grid:
            fasttext_dict.get("C").append(params.get("C"))
            fasttext_dict.get("max_iter").append(params.get("max_iter"))
        print(fasttext_dict)

    data = src.file_manager.get_df("0_parsed")

    train, test = train_test_split(data, random_state=1515, train_size=0.8)

    wordcount_range = range(min, max)
    if train_or_test == "test":
        classified_data, params = src.classify(technique="tfidf", train_data=train, test_data=test, unique=False, C=1.0, max_iter=500, write=False)
    for min_wordcount in wordcount_range:
        print(min_wordcount)
        if random:
            if train_or_test == "test":
                classified_data = classified_data.sample(n=test_count.get(min_wordcount))
            else:
                train = train.sample(n=train_count.get(min_wordcount))
                classified_data, params = src.classify(technique="tfidf", train_data=train, test_data=test, unique=False,
                                                       grid=grid, write=False)
        else:
            if train_or_test == "test":
                classified_data = classified_data[classified_data["parsed"]["wordcount"] >= min_wordcount]
            else:
                train = train[train["parsed"]["wordcount"] >= min_wordcount]
                classified_data, params = src.classify(technique="tfidf", train_data=train, test_data=test, unique=False, grid=grid, write=False)
        labels = classified_data["predict_proba_"].columns
        tfidf_dict.get("accuracy").append(
            accuracy_score(classified_data["parsed"]["character"], classified_data["classified"]["character"]))
        tfidf_dict.get("cross_entropy").append(
            log_loss(classified_data["parsed"]["character"], classified_data["predict_proba_"], labels=labels))
        tfidf_dict.get("predict_proba_predicted_character").append(
            classified_data["predict_proba_specific"]["predicted_character"].mean())
        if grid:
            tfidf_dict.get("C").append(params.get("C"))
            tfidf_dict.get("max_iter").append(params.get("max_iter"))
        print(tfidf_dict)

    print(wordcount_range)
    print()

    tfidf_df = pd.concat([pd.Series(v, index=wordcount_range) for k, v in tfidf_dict.items()],
                         keys=[k for k, v in tfidf_dict.items()], axis=1)
    fasttext_df = pd.concat([pd.Series(v, index=wordcount_range) for k, v in fasttext_dict.items()],
                            keys=[k for k, v in fasttext_dict.items()], axis=1)
    d = {
        "tfidf": tfidf_df,
        "fasttext": fasttext_df
    }

    df = pd.concat(d, axis=1)
    fm.write_df(df, "4_benchmark_change_testing_data_" + train_or_test + ("_random" if random else ""))
    return df