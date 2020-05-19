from integration_src.download import download_episodes
from integration_src.parse import parse_episodes
import integration_src.file_manager as fm
from integration_src.embed import embed_transcripts
from sklearn.metrics import accuracy_score
from integration_src.classify import classify_characters
from integration_src.benchmark import benchmark
from integration_src.details import display_benchmark
from integration_src.details import confusion_matrix
import pandas as pd

extraction_types = {'fasttext':{'C': 0.1, 'max_iter': 500},
                    'word2vec':{'C': 0.1, 'max_iter': 500},
                    'tfidf':{'C': 1.0, 'max_iter': 2000}}
benchmark_types = ["confidence_character", "confidence_predicted", "cross_entropy_loss"]

# for cur_type in extraction_types:
#     classify_characters(class_type=cur_type, C=extraction_types.cur_type.C, max_iter=extraction_types.cur_type.max_iter)

accuracy_per_min_wordcount = {'fasttext':{},
                    'word2vec':{},
                    'tfidf':{}}
for i in range(2,50):
    for type in extraction_types.keys():
        data = classify_characters(class_type=type, min_wordcount=i, grid=True, verbose=0)
        type_dict = accuracy_per_min_wordcount.get(type)
        type_dict.update({i:accuracy_score(data["parsed"]["character"], data["classified"]["character"])})
        accuracy_per_min_wordcount.update({type: type_dict})
        print(accuracy_per_min_wordcount)

print(accuracy_per_min_wordcount)
fm.write_df(pd.concat({"accuracy":pd.DataFrame(accuracy_per_min_wordcount)}, axis=1), "3_accuracy_per_wordcount")