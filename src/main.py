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
for feature in features.learned.keys():
    src.embed(technique=feature, unique=options.unique)
for feature, details in features.values():
    src.classify(technique=feature, unique=options.unique)
    src.benchmark(technique=feature, only_wrong=options.only_wrong)
for benchmark in benchmarks.keys():
    src.display_benchmark(benchmark_type=benchmark)