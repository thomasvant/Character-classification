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
import mpld3

created_files_dir = pathlib.Path.cwd().parent.joinpath('created_files')


def main():
    parsed_path = created_files_dir.joinpath('processed.csv')
    embedded_fasttext_path = created_files_dir.joinpath('embedded_fasttext.csv')
    embedded_word2vec_path = created_files_dir.joinpath('embedded_word2vec.csv')
    parsed_data = pd.read_csv(parsed_path, sep='|', index_col=0)

    # Word2Vec
    # Tuned hyperparameters: (best parameters)
    # {'C': 0.1, 'max_iter': 500, 'penalty': 'l2'}
    # Accuracy: 0.2181906218975002
    #
    # FastText
    # Tuned hyperparameters: (best parameters)
    # {'C': 0.1, 'max_iter': 500, 'penalty': 'l2'}
    # Accuracy: 0.2334271619341249
    #
    # TF-IDF
    # Tuned hyperparameters: (best parameters)
    # {'C': 0.0001, 'max_iter': 500, 'penalty': 'l2'}
    # Accuracy: 0.26580906989409814

    embedded_sisters_data = pd.read_csv(embedded_fasttext_path, sep='|',header=None, index_col=0)

    def tfidf():
        seed = 19981515
        X_train, X_rest, y_train, y_rest = \
            train_test_split(parsed_data['line'], parsed_data['character'], test_size=0.4, random_state=seed)
        X_test, X_validate, y_test, y_validate = \
            train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

        tfidf_transformer = TfidfVectorizer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train)
        X_test_tfidf = tfidf_transformer.transform(X_test)

        sc = StandardScaler(with_mean=False)
        scalar = sc.fit(X_train_tfidf)
        X_train_tfidf = scalar.transform(X_train_tfidf)
        X_test_tfidf = scalar.transform(X_test_tfidf)

        params = {'C': np.logspace(-20, 20, 41), 'penalty': ['l2'], 'max_iter': [500]}
        params = {'C': [0.0001], 'penalty': ['l2'], 'max_iter': [500]}

        lg = LogisticRegression(multi_class='multinomial')
        lg_grid = GridSearchCV(lg, params, verbose=3)
        lg_grid.fit(X_train_tfidf, y_train)
        print("Tuned hyperparameters :(best parameters) ", lg_grid.best_params_)
        print("Accuracy :", lg_grid.best_score_)

        classes = lg_grid.classes_

        predicted_probabilities = lg_grid.predict_proba(X_test_tfidf)
        predicted_label = lg_grid.predict(X_test_tfidf)

        df = pd.DataFrame({'line': X_test, 'character': y_test, 'predicted': predicted_label}, index=None)
        df_probabilities = pd.DataFrame(predicted_probabilities, columns=classes, index=df.index).round(3)
        result = pd.concat([df, df_probabilities], axis=1)
        result['failure'] = result.apply(lambda x: (x[x['predicted']] - x[x['character']]), axis=1).round(3)
        result['correct'] = result.apply(lambda x: x['predicted'] == x['character'], axis=1)
        result['line_length'] = result.apply(lambda x: len(x['line'].split(' ')), axis=1)

        cross_entropy_loss = log_loss(y_test, predicted_probabilities)
        print(cross_entropy_loss)
        # ordered_df = df_probabilities.apply(lambda x: (-x).argsort(), axis=1).apply(lambda x: df_probabilities.columns[x].to_list(), axis=1)
        # test = pd.concat([result['character'], ordered_df], axis=1)
        # number = test.apply(lambda x: x[0].index(x['character']), axis=1).to_frame(name="rank")
        # ordered_df = pd.DataFrame(ordered_df.to_list(), columns=list(range(1,7)), index=df.index)
        # result = pd.concat([result, number], axis=1)
        classification_tfidf_path = created_files_dir.joinpath('classification_tfidf.csv')

        result.sort_values(by=['failure'], ascending=False).to_csv(classification_tfidf_path, sep='|')
        print("TF-IDF")
        print("Test set:")
        print(metrics.classification_report(y_test, lg_grid.predict(X_test_tfidf)))
        print(metrics.confusion_matrix(y_test, lg_grid.predict(X_test_tfidf)))

        print("Train set:")
        print(metrics.classification_report(y_train, lg_grid.predict(X_train_tfidf)))
        print(metrics.confusion_matrix(y_train, lg_grid.predict(X_train_tfidf)))
        print()

    def sisters():
        seed = 19981515
        X_train, X_rest, y_train, y_rest = \
            train_test_split(embedded_sisters_data, parsed_data, test_size=0.4, random_state=seed)
        X_test, X_validate, y_test, y_validate = \
            train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

        params = {'C': np.logspace(-20, 0, 21), 'penalty': ['l2'], 'max_iter': [500]}
        params = {'C': [0.1], 'penalty': ['l2'], 'max_iter': [500]}

        lg = LogisticRegression(multi_class='multinomial')
        lg_grid = GridSearchCV(lg, params, verbose=3)
        lg_grid.fit(X_train, y_train['character'])
        print("Tuned hyperparameters :(best parameters) ", lg_grid.best_params_)
        print("Accuracy :", lg_grid.best_score_)

        classes = lg_grid.classes_

        predicted_probabilities = lg_grid.predict_proba(X_test)
        predicted_label = lg_grid.predict(X_test)

        df = pd.DataFrame({'line': y_test['line'], 'character': y_test['character'], 'predicted': predicted_label}, index=None)
        df_probabilities = pd.DataFrame(predicted_probabilities, columns=classes, index=df.index).round(3)
        result = pd.concat([df, df_probabilities], axis=1)
        result['failure'] = result.apply(lambda x: (x[x['predicted']] - x[x['character']]), axis=1).round(3)
        result['correct'] = result.apply(lambda x: x['predicted'] == x['character'], axis=1)
        result['line_length'] = result.apply(lambda x: len(x['line'].split(' ')), axis=1)
        # ordered_df = df_probabilities.apply(lambda x: (-x).argsort(), axis=1).apply(
        #     lambda x: df_probabilities.columns[x].to_list(), axis=1)
        # test = pd.concat([result['character'], ordered_df], axis=1)
        # number = test.apply(lambda x: x[0].index(x['character']), axis=1).to_frame(name="rank")
        # result = pd.concat([result, number], axis=1)
        cross_entropy_loss = log_loss(y_test['character'], predicted_probabilities)
        print(cross_entropy_loss)
        classification_sisters_path = created_files_dir.joinpath('classification_sisters.csv')

        result.sort_values(by=['failure'], ascending=False).to_csv(classification_sisters_path, sep='|')

        print("SISTERS")
        print("Test set:")
        print(metrics.classification_report(y_test['character'].tolist(), lg_grid.predict(X_test)))
        print(metrics.confusion_matrix(y_test['character'].tolist(), lg_grid.predict(X_test)))

        print("Train set:")
        print(metrics.classification_report(y_train['character'].tolist(), lg_grid.predict(X_train)))
        print(metrics.confusion_matrix(y_train['character'].tolist(), lg_grid.predict(X_train)))
        print()

    sisters()


# def sisters():
#     embedded_path = created_files_dir.joinpath('embedded_sisters.csv')
#     parsed_path = created_files_dir.joinpath('parsed.csv')
#
#     embedded_data = pd.read_csv(embedded_path, sep='|', header=None, index_col=False)
#     parsed_data = pd.read_csv(parsed_path, sep='|', index_col=False)
#
#     # Split data into train, test, validate (60:20:20)
#     seed = 19981515
#     X_train, X_rest, y_train, y_rest = \
#         train_test_split(embedded_data, parsed_data['character'], test_size=0.4, random_state=seed)
#     X_test, X_validate, y_test, y_validate = \
#         train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)
#     params = {'C': [0.1], 'max_iter': [500], 'penalty': ['l2']}
#     # params = {'C': np.logspace(-20, 20, 41), 'penalty': ['l2'], 'max_iter': [500, 2000]}
#
#     logreg = LogisticRegression()
#     logreg_cv = GridSearchCV(logreg, params, verbose=3)
#     logreg_cv.fit(X_train, y_train)
#     print("Tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
#     print("Accuracy :", logreg_cv.best_score_)
#     y_test_predicted = logreg_cv.predict(X_test)
#     y_train_predicted = logreg_cv.predict(X_train)
#
#     print("")
#     print(metrics.classification_report(y_test, y_test_predicted))
#     print(metrics.confusion_matrix(y_test, y_test_predicted))
#
#     print(metrics.classification_report(y_train, y_train_predicted))
#     print(metrics.confusion_matrix(y_train, y_train_predicted))

    # return y_test, y_test_predicted, y_train, y_train_predicted

    # clf = LogisticRegression(max_iter=100, C=1.0).fit(X_train, y_train['character'])
    #
    # # Predict class from scaled test dataset
    # y_test_predicted = clf.predict(X_test)
    #
    # tsne = TSNE(n_components=2, random_state=0)
    # tsne_object = tsne.fit_transform(X_test)
    #
    # tsne_df = pd.DataFrame({'X': tsne_object[:, 0],
    #                         'Y': tsne_object[:, 1],
    #                         'predicted_char': y_test_predicted,
    #                         'actual_char': y_test['character'],
    #                         'line': y_test['line']})
    #
    # tsne_path = created_files_dir.joinpath('tsne_sisters.csv')
    # # tsne_df.to_csv(tsne_path, sep='|', index=False)
    # tsne_df = pd.read_csv(tsne_path, sep='|', index_col=False)
    # #
    # sns.scatterplot(x="X", y="Y",
    #                 hue="actual_char",
    #                 legend='full',
    #                 data=tsne_df)
    # #
    # plt.show()
    #
    # return y_test['character'], y_test_predicted


# def tfidf():
#     parsed_path = created_files_dir.joinpath('parsed.csv')
#     parsed_data = pd.read_csv(parsed_path, sep='|', index_col=False)
#     parsed_data = parsed_data.head(10)
#
#     # Split data into train, test, validate (60:20:20)
#     seed = 19981515
#     X_train, X_rest, y_train, y_rest = \
#         train_test_split(parsed_data['line'], parsed_data['character'], test_size=0.4, random_state=seed)
#     X_test, X_validate, y_test, y_validate = \
#         train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)
#
#     # Create TFIDF from train data and scale
#     tfidf_transformer = TfidfVectorizer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train)
#     X_test_tfidf = tfidf_transformer.transform(X_test)
#
#     sc = StandardScaler(with_mean=False)
#     scalar = sc.fit(X_train_tfidf)
#     X_train_tfidf_scaled = scalar.transform(X_train_tfidf)
#     X_test_tfidf_scaled = scalar.transform(X_test_tfidf)
#     params = {'C': [0.0001], 'max_iter': [500]}
#     # C is [1.e-20, 1.e-19, 1.e-18 ... 1.e+19, 1.e+20]
#     # params = {'C': np.logspace(-20, 20, 41), 'penalty': ['l2'], 'max_iter': [500, 2000]}
#
#     logreg = LogisticRegression()
#     logreg_cv = GridSearchCV(logreg, params, verbose=3)
#     logreg_cv.fit(X_train_tfidf_scaled, y_train)
#     print("Tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
#     print("Accuracy :", logreg_cv.best_score_)
#     y_test_predicted = logreg_cv.predict(X_test_tfidf_scaled)
#     y_train_predicted = logreg_cv.predict(X_train_tfidf_scaled)
#     probabilities_of_prediction = logreg_cv.predict_proba(X_test_tfidf_scaled)
#     print(probabilities_of_prediction)

    # Train logistic regression model
    # clf = LogisticRegression(max_iter=1000, C=0.1).fit(X_train_tfidf_scaled, y_train)

    # Predict class from scaled test dataset
    # y_test_predicted = clf.predict(X_test_tfidf_scaled)

    # return y_test, y_test_predicted, y_train, y_train_predicted

    # selected_characters = ['phoebe', 'chandler']
    # two_characters_data = parsed_data[parsed_data['character'].isin(selected_characters)]

    # tsne = TSNE(n_components=6, random_state=0)
    # tsne_object = tsne.fit_transform(X_test_tfidf_scaled)
    #
    # tsne_df = pd.DataFrame({'X': tsne_object[:, 0],
    #                         'Y': tsne_object[:, 1],
    #                         'predicted_char': y_test_predicted,
    #                         'actual_char': y_test,
    #                         'line': X_test})
    # #
    # tsne_path = created_files_dir.joinpath('tsne_tfidf.csv')
    # tsne_df.to_csv(tsne_path, sep='|', index=False)
    # tsne_df = pd.read_csv(tsne_path, sep='|', index_col=False)
    # #
    # #
    # sns.scatterplot(x="X", y="Y",
    #                 hue="predicted_char",
    #                 legend='full',
    #                 data=tsne_df)
    #
    # plt.show()


main()
