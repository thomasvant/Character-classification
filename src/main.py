import src

features = {
    "learned": {
        "word2vec": {},
        "fasttext": {}
    },
    "hand_crafted": {
        "tfidf": {}
    }
}
options = {
    "unique": True,
    "only_wrong": False,
    "min_wordcount": {
        "enabled": True,
        "min": 2,
        "max": 30
    }
}

benchmarks = {
    "confidence_character":{},
    "confidence_predicted":{},
    "cross_entropy_loss":{}
}

# src.parse()
# for feature in features.get("learned").keys():
#     src.embed(technique=feature, unique=options.get("unique"))
# for technique, features in features.items():
#     for feature, details in features.items():
#         src.classify(technique=feature, unique=options.get("unique"))
#         src.benchmark(technique=feature, only_wrong=options.get("only_wrong"))
# for benchmark in benchmarks.keys():
#     src.display_benchmark(benchmark_type=benchmark)

# src.accuracy_per_min_wordcount(2, 42, unique=True)
src.parse()
src.embed(technique="elmo", unique=True)