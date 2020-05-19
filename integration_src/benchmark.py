import integration_src.file_manager as fm
import integration_src.classify as classify
from sklearn import metrics
import pandas as pd
from sklearn.metrics import log_loss

sim_types = ['fasttext', 'word2vec', 'elmo', 'tfidf']


def benchmark(data=None, class_type="tfidf", only_wrong=False):
    if class_type not in sim_types:
        raise ValueError("Invalid benchmarking type. Expected one of: %s" % sim_types)
    print("Benchmarking lines using " + class_type)
    if data is None:
        data = fm.get_df("2_classified_" + class_type)
        if data is None:
            data = classify.classify_characters(class_type=class_type)

    if only_wrong:
        data = data[data["classified"]["is_correct"] == False]

    wordcount = data["parsed"]["wordcount"]
    wordcount_dict = dict.fromkeys(wordcount.to_list())

    cross_entropy_dict = wordcount_dict.copy()
    confidence_character = wordcount_dict.copy()
    confidence_predicted = wordcount_dict.copy()

    labels = data["predicted_probabilities"].columns
    for cur_number in wordcount_dict:
        parsed_character = data["parsed"][wordcount == cur_number]["character"]
        indexes = parsed_character.index

        # Cross entropy
        predicted_probabilities = data["predicted_probabilities"][data.index.isin(indexes)]
        cross_entropy_loss = log_loss(parsed_character, predicted_probabilities, labels=labels)
        cross_entropy_dict.update({cur_number:cross_entropy_loss})

        confidence = data["confidence"][data.index.isin(indexes)]
        confidence_character.update({cur_number:confidence["character"].mean()})
        confidence_predicted.update({cur_number: confidence["predicted"].mean()})

    probabilities_df = pd.DataFrame(cross_entropy_dict.values(), columns=["cross_entropy_loss"], index=wordcount_dict.keys())
    confidence_predicted_df = pd.DataFrame(confidence_predicted.values(), columns=["predicted"], index=wordcount_dict.keys())
    confidence_character_df = pd.DataFrame(confidence_character.values(), columns=["character"], index=wordcount_dict.keys())

    d = {"cross_entropy": probabilities_df,
         "average_confidence": pd.concat([confidence_predicted_df, confidence_character_df], axis=1)}
    new_data = pd.concat(d, axis=1).sort_index()
    fm.write_df(new_data, "3_benchmark_" + class_type)
    return new_data

