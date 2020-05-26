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

for k, v in features.items():
    # src.classify(technique=k, unique=options.get("unique"))
    src.confusion_matrix(technique=k)

# src.benchmark_per_wordcount()
# src.benchmark_per_min_wordcount(grid=True)

# for k, v in benchmarks.items():
#     src.display_benchmark_per_wordcount(benchmark=k, min_wordcount=True)
