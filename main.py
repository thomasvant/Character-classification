import src

features = {
    "tfidf": {},
    "fasttext": {}
}
options = {
    "unique": True,
    "only_wrong": False,
    "min_wordcount": {
        "enabled": True,
        "min": 2,
        "max": 30
    },
    "correct_spelling": False,
    "stemming": False,
    "remove_stopwords": False,
    "expand_contractions": True
}

benchmarks = {
    "accuracy":{},
    "cross_entropy":{},
    "predict_proba_predicted_character":{}
}

# src.parse()
# src.embed()
# src.classify(technique="fasttext", grid=True, verbose=3)
# src.classify(technique="tfidf", grid=True, verbose=3)
# src.confusion_matrix(technique="tfidf")
# src.confusion_matrix(technique="tfidf")
# src.benchmark_change_data(train_or_test="test")
src.benchmark_change_data(train_or_test="train")

for k, v in benchmarks.items():
    # src.display_changing_dataset_benchmark(benchmark=k, test_or_train="test")
    # src.display_changing_dataset_benchmark(benchmark=k, test_or_train="train")
    pass