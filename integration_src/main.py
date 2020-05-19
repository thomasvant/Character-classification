from integration_src.download import download_episodes
from integration_src.parse import parse_episodes
import integration_src.file_manager as fm
from integration_src.embed import embed_transcripts
from integration_src.classify import classify_characters
from integration_src.benchmark import benchmark
from integration_src.details import display_benchmark

extraction_types = ['fasttext', 'word2vec', 'tfidf']
benchmark_types = ["confidence_character", "confidence_predicted", "cross_entropy_loss"]

for type in extraction_types:
    classify_characters(class_type=type, grid=True)

# for type in extraction_types:
#     benchmark(class_type=type, only_wrong=True)
for type in benchmark_types:
    display_benchmark(benchmark_type=type)