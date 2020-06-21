import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import src.file_manager as fm
import src.embed as embed

sim_types = ['fasttext', 'tfidf']

__all__ = ["classify"]


def classify(technique="tfidf", train_data=None, test_data=None, C=None, max_iter=None, write=False):
    if technique not in sim_types:
        raise ValueError("Invalid classification type " + technique + ". Expected one of: %s" % sim_types)
    print("Classifying lines using " + technique)
    if technique == "tfidf":
        data = fm.get_df("0_parsed")
    else:
        data = fm.get_df("1_embedded_fasttext")

    if train_data is not None and test_data is not None:
        print("Test and train data provided, using those instead")
        print("Test size: ", test_data.shape)
        print("Train size: ", train_data.shape)
        train = train_data
        test = test_data
    else:
        train, test = train_test_split(data, random_state=1515, train_size=0.8)

    y_train = train["parsed"]["character"]
    y_test = test["parsed"]["character"]

    if technique == "tfidf":
        tfidf = TfidfVectorizer()
        x_train = tfidf.fit_transform(train["parsed"]["line"])
        x_test = tfidf.transform(test["parsed"]["line"])
    else:
        x_train = train["embedded"]
        x_test = test["embedded"]

    lg = LogisticRegression(C=C, max_iter=max_iter)

    lg.fit(x_train, y_train)

    predict_proba_df = pd.DataFrame(lg.predict_proba(x_test), columns=lg.classes_, index=y_test.index)
    predict_df = pd.DataFrame(lg.predict(x_test), columns=["character"],index=y_test.index)

    predict_proba_character_series = pd.concat([y_test, predict_proba_df], axis=1)\
        .apply(lambda x: x[x["character"]], axis=1)
    predict_proba_predicted_series = pd.concat([predict_df, predict_proba_df], axis=1)\
        .apply(lambda x: x[x["character"]], axis=1)
    predict_proba_specific_df = pd.concat([predict_proba_character_series, predict_proba_predicted_series], keys=["actual_character", "predicted_character"], axis=1)

    is_correct_df = predict_df["character"].eq(y_test).to_frame(name="is_correct")

    d = {"parsed": test["parsed"],
         "classified": pd.concat([predict_df,is_correct_df], axis=1),
         "predict_proba_specific": predict_proba_specific_df,
         "predict_proba_": predict_proba_df}
    data = pd.concat(d, axis=1)

    if write:
        fm.write_df(data, "2_classified_" + technique)

    return data, lg
