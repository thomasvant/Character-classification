import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import src.file_manager as fm
import src.embed as embed

sim_types = ['fasttext', 'word2vec', 'elmo', 'tfidf']

__all__ = ["classify"]


def classify(data=None, technique="tfidf", grid=False, C=None, max_iter=None, cv=None, min_wordcount=None, verbose=0, unique=False, write=True):
    if technique not in sim_types:
        raise ValueError("Invalid classification type " + technique + ". Expected one of: %s" % sim_types)
    print("Classifying lines using " + technique)
    if data is None:
        if technique == "tfidf":
            data = fm.get_df("0_parsed", unique=unique)
        else:
            data = fm.get_df("1_embedded_" + technique, unique=unique)
            if data is None:
                data = embed.embed_transcripts(type=technique)

    if min_wordcount:
        data = data[data["parsed"]["wordcount"] > min_wordcount]

    train, non_train = train_test_split(data, random_state=1515, train_size=0.6)
    test, validate = train_test_split(non_train, random_state=1515, train_size=0.5)

    y_train = train["parsed"]["character"]
    y_test = test["parsed"]["character"]

    if technique == "tfidf":
        tfidf = TfidfVectorizer()
        x_train = tfidf.fit_transform(train["parsed"]["line"])
        x_test = tfidf.transform(test["parsed"]["line"])
    else:
        x_train = train["embedded"]
        x_test = test["embedded"]


    if grid:
        params = {"C": np.logspace(-10, 5, 16), 'max_iter': [500, 1000, 2000]}
        if C:
            params["C"] = C if technique is list else [C]
        if max_iter:
            params["max_iter"] = max_iter if technique is list else [max_iter]
        lg = GridSearchCV(LogisticRegression(multi_class="multinomial"), params, verbose=verbose, n_jobs=-1, cv=cv)
    else:
        lg = LogisticRegression(multi_class="multinomial", C=C if C else 1, max_iter=max_iter if max_iter else 500)

    lg.fit(x_train, y_train)

    if grid:
        best = lg.best_params_
        print("Best parameters: ", best)
    else:
        best = None

    predict_proba_df = pd.DataFrame(lg.predict_proba(x_test), columns=lg.classes_, index=y_test.index)
    decision_function_df = pd.DataFrame(lg.decision_function(x_test), columns=lg.classes_, index=y_test.index)

    predict_df = pd.DataFrame(lg.predict(x_test), columns=["character"],index=y_test.index)

    predict_proba_character_series = pd.concat([y_test, predict_proba_df], axis=1)\
        .apply(lambda x: x[x["character"]], axis=1)
    predict_proba_predicted_series = pd.concat([predict_df, predict_proba_df], axis=1)\
        .apply(lambda x: x[x["character"]], axis=1)
    predict_proba_specific_df = pd.concat([predict_proba_character_series, predict_proba_predicted_series], keys=["actual_character", "predicted_character"], axis=1)

    decision_function_character_series = pd.concat([y_test, decision_function_df], axis=1)\
        .apply(lambda x: x[x["character"]], axis=1)
    decision_function_predicted_series = pd.concat([predict_df, decision_function_df], axis=1) \
        .apply(lambda x: x[x["character"]], axis=1)
    decision_function_specific_df = pd.concat([decision_function_character_series, decision_function_predicted_series], keys=["actual_character", "predicted_character"], axis=1)


    is_correct_df = predict_df["character"].eq(y_test).to_frame(name="is_correct")

    # confidence = pd.DataFrame(lg.decision_function(x_test), columns=["confidence"],index=y_test.index)

    d = {"parsed": test["parsed"],
         "classified": pd.concat([predict_df,is_correct_df], axis=1),
         "predict_proba_specific": predict_proba_specific_df,
         "decision_function_specific": decision_function_specific_df,
         "predict_proba_": predict_proba_df,
         "decision_function": decision_function_df}
    data = pd.concat(d, axis=1)
    if write:
        fm.write_df(data, "2_classified_" + technique)
    return data, best
