import src.file_manager as fm
import src
import src.classify as classify
from sklearn import metrics
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from copy import deepcopy

sim_types = ['fasttext', 'tfidf']

__all__ = ["benchmark_per_wordcount", "benchmark_per_min_wordcount"]


def benchmark_per_wordcount(grid=False):
    print("Benchmarking per wordcount")

    dictionary = {
        "accuracy": [],
        "cross_entropy": [],
        "predict_proba_predicted_character": []
    }

    fasttext = deepcopy(dictionary)
    tfidf = deepcopy(dictionary)

    tfidf_data, _ = src.classify(technique="tfidf", unique=True, grid=grid, \
                                    C=1.0, max_iter=500, \
                                    write=False)
    fasttext_data, _ = src.classify(technique="fasttext", unique=True, grid=grid, \
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


def benchmark_per_min_wordcount(grid=False, min=2, max=30):
    print("Benchmarking per min wordcount from " + str(min) + " to " + str(max) + " with grid set to " + str(grid))

    dictionary = {
        "accuracy": [],
        "cross_entropy": [],
        "predict_proba_predicted_character": [],
        "C": [],
        "max_iter": []
    }
    fasttext = deepcopy(dictionary)
    tfidf = deepcopy(dictionary)

    wordcount_range = range(min, max + 1)
    print(wordcount_range)
    print(list(wordcount_range))
    for i in wordcount_range:
        print("Min wordcount: " + str(i))
        tfidf_data, tfidf_params = src.classify(technique="tfidf", min_wordcount=i, unique=True, grid=grid,\
                                     # C=1.0, max_iter=500, \
                                     write=False)
        tfidf_labels = tfidf_data["predict_proba_"].columns
        tfidf.get("accuracy").append(accuracy_score(tfidf_data["parsed"]["character"], tfidf_data["classified"]["character"]))
        tfidf.get("cross_entropy").append(log_loss(tfidf_data["parsed"]["character"], tfidf_data["predict_proba_"], labels=tfidf_labels))
        tfidf.get("predict_proba_predicted_character").append(tfidf_data["predict_proba_specific"]["predicted_character"].mean())
        tfidf.get("C").append(tfidf_params.get("C"))
        tfidf.get("max_iter").append(tfidf_params.get("max_iter"))
        print(tfidf)

        fasttext_data, fasttext_params = src.classify(technique="fasttext", min_wordcount=i, unique=True, grid=grid, \
                                        # C=0.1, max_iter=500, \
                                        write=False)
        fasttext_labels = fasttext_data["predict_proba_"].columns
        fasttext.get("accuracy").append(accuracy_score(fasttext_data["parsed"]["character"], fasttext_data["classified"]["character"]))
        fasttext.get("cross_entropy").append(log_loss(fasttext_data["parsed"]["character"], fasttext_data["predict_proba_"], labels=fasttext_labels))
        fasttext.get("predict_proba_predicted_character").append(fasttext_data["predict_proba_specific"]["predicted_character"].mean())
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
    fm.write_df(data, "3_benchmark_per_min_wordcount")
    return data
