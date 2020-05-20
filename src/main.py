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
    "unique": False,
    "only_wrong": False
}

benchmarks = {
    "confidence_character":{},
    "confidence_predicted":{},
    "cross_entropy_loss":{}
}

src.parse()
for feature in features.get("learned").keys():
    src.embed(technique=feature, unique=options.get("unique"))
for technique, features in features.items():
    for feature, details in features.items():
        src.classify(technique=feature, unique=options.get("unique"))
        src.benchmark(technique=feature, only_wrong=options.get("only_wrong"))
for benchmark in benchmarks.keys():
    src.display_benchmark(benchmark_type=benchmark)