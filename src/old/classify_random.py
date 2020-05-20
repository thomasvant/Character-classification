import pathlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import random


def main():
    created_files_dir = pathlib.Path.cwd().parent.joinpath('created_files')
    parsed_path = created_files_dir.joinpath('parsed.csv')

    parsed_data = pd.read_csv(parsed_path, sep='|', index_col=False)
    parsed_lines = parsed_data['line']
    parsed_characters = parsed_data['character']
    seed = 19981515
    X_train, X_rest, y_train, y_rest = \
        train_test_split(parsed_lines, parsed_characters, test_size=0.4, random_state=seed)
    X_test, X_validate, y_test, y_validate = \
        train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

    characters = ['chandler', 'joey', 'monica', 'phoebe', 'rachel', 'ross']
    predicted = []
    for _ in y_test:
        predicted.append(random.choice(characters))

    # print(predicted)
    # print(type(y_test))

    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))


main()
