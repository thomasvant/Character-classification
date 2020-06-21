import src.file_manager as fm
import src
import src.classify as classify
from sklearn import metrics
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from copy import deepcopy
from sklearn.model_selection import train_test_split

sim_types = ['fasttext', 'tfidf']

__all__ = ["benchmark_per_wordcount", "benchmark_per_min_wordcount", "benchmark_change_data"]


def benchmark_per_wordcount(grid=False):
    print("Benchmarking per wordcount")

    dictionary = {
        "accuracy": [],
        "cross_entropy": [],
        "predict_proba_predicted_character": []
    }

    fasttext = deepcopy(dictionary)
    tfidf = deepcopy(dictionary)

    tfidf_data, _ = src.classify(technique="tfidf", unique=False, grid=grid, \
                                    C=1.0, max_iter=500, \
                                    write=False)
    fasttext_data, _ = src.classify(technique="fasttext", unique=False, grid=grid, \
                                 C=1.0, max_iter=500, \
                                 write=False)

    labels = tfidf_data["predict_proba_"].columns
    wordcounts = list(tfidf_data["parsed"]["wordcount"].unique())

    for count in wordcounts:
        cur_tfidf = tfidf_data[tfidf_data["parsed"]["wordcount"] == count]
        cur_fasttext = fasttext_data[fasttext_data["parsed"]["wordcount"] == count]

        tfidf.get("accuracy").append(
            accuracy_score(cur_tfidf["parsed"]["character"], cur_tfidf["classified"]["character"]))
        tfidf.get("cross_entropy").append(
            log_loss(cur_tfidf["parsed"]["character"], cur_tfidf["predict_proba_"], labels=labels))
        tfidf.get("predict_proba_predicted_character").append(
            cur_tfidf["predict_proba_specific"]["predicted_character"].mean())

        fasttext.get("accuracy").append(
            accuracy_score(cur_fasttext["parsed"]["character"], cur_fasttext["classified"]["character"]))
        fasttext.get("cross_entropy").append(
            log_loss(cur_fasttext["parsed"]["character"], cur_fasttext["predict_proba_"], labels=labels))
        fasttext.get("predict_proba_predicted_character").append(
            cur_fasttext["predict_proba_specific"]["predicted_character"].mean())

    tfidf_df = pd.concat([pd.Series(v, index=wordcounts) for k, v in tfidf.items()],
                         keys=[k for k, v in tfidf.items()], axis=1)
    fasttext_df = pd.concat([pd.Series(v, index=wordcounts) for k, v in fasttext.items()],
                            keys=[k for k, v in fasttext.items()], axis=1)
    d = {
        "tfidf": tfidf_df,
        "fasttext": fasttext_df
    }

    data = pd.concat(d, axis=1).sort_index()
    fm.write_df(data, "3_benchmark_per_wordcount")
    return data


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


def benchmark_per_min_wordcount(min=2, max=30, grid=False):
    print("Benchmarking per min wordcount from " + str(min) + " to " + str(max) + " with grid set to " + str(grid))

    if grid:
        dictionary = {
            "accuracy": [],
            "cross_entropy": [],
            "predict_proba_predicted_character": [],
            "C": [],
            "max_iter": []
        }
    else:
        dictionary = {
            "accuracy": [],
            "cross_entropy": [],
            "predict_proba_predicted_character": [],
        }
    fasttext = deepcopy(dictionary)
    tfidf = deepcopy(dictionary)

    wordcount_range = range(min, max + 1)
    print(wordcount_range)
    print(list(wordcount_range))
    for i in wordcount_range:
        print("Min wordcount: " + str(i))
        if grid:
            tfidf_data, tfidf_params = src.classify(technique="tfidf", min_wordcount=i, unique=False, grid=grid,\
                                         # C=1.0, max_iter=500, \
                                         write=False)
        else:
            tfidf_data, tfidf_params = src.classify(technique="tfidf", min_wordcount=i, unique=False, grid=grid, \
                                                    C=1.0, max_iter=500, \
                                                    write=False)
        tfidf_labels = tfidf_data["predict_proba_"].columns
        tfidf.get("accuracy").append(accuracy_score(tfidf_data["parsed"]["character"], tfidf_data["classified"]["character"]))
        tfidf.get("cross_entropy").append(log_loss(tfidf_data["parsed"]["character"], tfidf_data["predict_proba_"], labels=tfidf_labels))
        tfidf.get("predict_proba_predicted_character").append(tfidf_data["predict_proba_specific"]["predicted_character"].mean())
        if grid:
            tfidf.get("C").append(tfidf_params.get("C"))
            tfidf.get("max_iter").append(tfidf_params.get("max_iter"))
        print(tfidf)

        if grid:
            fasttext_data, fasttext_params = src.classify(technique="fasttext", min_wordcount=i, unique=False, grid=grid, \
                                            # C=0.1, max_iter=500, \
                                            write=False)
        else:
            fasttext_data, fasttext_params = src.classify(technique="fasttext", min_wordcount=i, unique=False, grid=grid, \
                                                          C=0.1, max_iter=500, \
                                                          write=False)
        fasttext_labels = fasttext_data["predict_proba_"].columns
        fasttext.get("accuracy").append(accuracy_score(fasttext_data["parsed"]["character"], fasttext_data["classified"]["character"]))
        fasttext.get("cross_entropy").append(log_loss(fasttext_data["parsed"]["character"], fasttext_data["predict_proba_"], labels=fasttext_labels))
        fasttext.get("predict_proba_predicted_character").append(fasttext_data["predict_proba_specific"]["predicted_character"].mean())
        if grid:
            fasttext.get("C").append(fasttext_params.get("C"))
            fasttext.get("max_iter").append(fasttext_params.get("max_iter"))
        print(fasttext)

    tfidf_df = pd.concat([pd.Series(v, index=wordcount_range) for k,v in tfidf.items()], keys=[k for k,v in tfidf.items()], axis=1)
    fasttext_df = pd.concat([pd.Series(v, index=wordcount_range) for k,v in fasttext.items()], keys=[k for k,v in fasttext.items()], axis=1)
    d = {
        "tfidf": tfidf_df,
        "fasttext": fasttext_df
    }

    data = pd.concat(d, axis=1)
    fm.write_df(data, "3_benchmark_per_min_wordcount" + ("_grid" if grid else ""))
    return data
