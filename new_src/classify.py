import pathlib
import pandas as pd
from boto import sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import time


def main():
    # Read data
    created_files_dir = pathlib.Path.cwd().parent.joinpath('created_files')
    parsed_path = created_files_dir.joinpath('parsed.csv')

    parsed_data = pd.read_csv(parsed_path, sep='|', index_col=False)
    parsed_lines = parsed_data['line']
    parsed_characters = parsed_data['character']


    # Split data into train, test, validate (60:20:20)
    seed = 19981515
    X_train, X_rest, y_train, y_rest = \
        train_test_split(parsed_lines, parsed_characters, test_size=0.4, random_state=seed)
    X_test, X_validate, y_test, y_validate = \
        train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

    # Create TFIDF from train data and scale
    tfidf_transformer = TfidfVectorizer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)

    sc = StandardScaler(with_mean=False)
    scalar = sc.fit(X_train_tfidf)
    X_train_tfidf_scaled = scalar.transform(X_train_tfidf)


    # Train logistic regression model
    clf = LogisticRegression(max_iter=100, C=5.0).fit(X_train_tfidf_scaled, y_train)


    # Create TFIDF from test data and scale
    X_test_tfidf = tfidf_transformer.transform(X_test)
    X_test_tfidf_scaled = scalar.transform(X_test_tfidf)


    # Predict class from scaled test dataset
    y_test_predicted = clf.predict(X_test_tfidf_scaled)

    print(metrics.classification_report(y_test, y_test_predicted))
    print(metrics.confusion_matrix(y_test, y_test_predicted))

    # Create visualisation
    # https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    # print('Starting t-SNE')
    # time_start = time.time()
    # TSNE_results = TSNE(n_components=2).fit_transform(X_train_scaled)
    # print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    #
    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 10),
    #     data=TSNE_results,
    #     legend="full",
    #     alpha=0.3
    # )

main()
