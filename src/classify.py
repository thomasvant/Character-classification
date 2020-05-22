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

def classify(data=None, technique="tfidf", grid=False, C=None, max_iter=None, cv=None, min_wordcount=None, verbose=0, unique=False):
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
        params = {"C": np.logspace(-5, 5, 11), 'max_iter': [500, 1000, 2000]}
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

    predicted_probability_df = pd.DataFrame(lg.predict_proba(x_test), columns=lg.classes_, index=y_test.index)
    confidence_df = pd.DataFrame(lg.decision_function(x_test), columns=lg.classes_, index=y_test.index)

    predicted = pd.DataFrame(lg.predict(x_test), columns=["character"],index=y_test.index)
    confidence_prediction_series = pd.concat([predicted,confidence_df], axis=1)\
        .apply(lambda x: x[x["character"]], axis=1)
    confidence_prediction_df = pd.DataFrame(confidence_prediction_series, columns=["predicted"])
    confidence_character_series = pd.concat([y_test, confidence_df], axis=1) \
        .apply(lambda x: x[x["character"]], axis=1)
    confidence_character_df = pd.DataFrame(confidence_character_series, columns=["character"])

    is_correct = predicted["character"].eq(y_test).to_frame(name="is_correct")

    # confidence = pd.DataFrame(lg.decision_function(x_test), columns=["confidence"],index=y_test.index)

    d = {"parsed": test["parsed"],
         "classified": pd.concat([predicted,is_correct], axis=1),
         "confidence": pd.concat([confidence_prediction_df, confidence_character_df], axis=1),
         "confidence_per_character": confidence_df,
         "predicted_probabilities": predicted_probability_df}
    data = pd.concat(d, axis=1)
    fm.write_df(data, "2_classified_" + technique)
    return data, best
