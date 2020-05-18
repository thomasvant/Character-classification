import pathlib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import time
import numpy as np
import integration_src.file_manager as fm
import integration_src.embed as embed
import mpld3

sim_types = ['fasttext', 'word2vec', 'elmo', 'tfidf']


def classify_characters(data=fm.get_df("0_parsed"), type="tfidf", C=None, max_iter=None, cv=None):
    if type not in sim_types:
        raise ValueError("Invalid classification type. Expected one of: %s" % sim_types)
    print("Classifying lines using " + type)

    train, non_train = train_test_split(data, random_state=1515, train_size=0.6)
    test, validate = train_test_split(non_train, random_state=1515, train_size=0.5)

    y_train = train["parsed"]["character"]
    y_test = test["parsed"]["character"]

    if type == "tfidf":
        tfidf = TfidfVectorizer()
        x_train = tfidf.fit_transform(train["parsed"]["line"])
        x_test = tfidf.transform(test["parsed"]["line"])
    elif type == ("fasttext" or "word2vec" or "elmo"):
        x_train = train["embedded"]
        x_test = test["embedded"]

    params = {"C": np.logspace(-5, 5, 21), 'max_iter': [100, 500, 1000, 2000, 5000]}
    if C:
        params["C"] = C if type(C) is list else [C]
    if max_iter:
        params["max_iter"] = max_iter if type(max_iter) is list else [max_iter]
    lg = GridSearchCV(LogisticRegression(multi_class="multinomial"), params, verbose=3, n_jobs=-1, cv=cv)
    lg.fit(x_train, y_train)
    print("Best parameters: ", lg.best_params_)

    predicted_probability_df = pd.DataFrame(lg.predict_proba(x_test), columns=lg.classes_, index=y_test.index)
    predicted = lg.predict(x_test)

    d = {"parsed": x_test["parsed"],
         "classified": pd.DataFrame([predicted, y_test == predicted], columns=["character","is_correct"], index=y_test.index),
         "predicted_probabilities": predicted_probability_df}

    return pd.concat(d, axis=1)
