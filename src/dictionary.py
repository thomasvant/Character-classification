import pandas as pd
import pathlib
import time
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn import metrics


dir_transcripts_parsed = pathlib.Path.cwd().parent.joinpath('transcripts_parsed')
dir_dictionary = pathlib.Path.cwd().parent.joinpath('dictionary')
dir_dictionary.mkdir(parents=True, exist_ok=True)
path_dictionary = dir_dictionary.joinpath('dictionary_occurrence.csv')
path_details = dir_dictionary.joinpath('details.csv')
path_normalized = dir_dictionary.joinpath('dictionary_normalized.csv')
default = {'joey': 0, 'rachel': 0, 'monica': 0, 'phoebe': 0, 'chandler': 0, 'ross': 0, 'other': 0}


def create_dictionary():
    print("Creating dictionary")
    dict = {}
    details = {'word_count': default.copy(), 'sentence_count': default.copy()}

    for path_episode in dir_transcripts_parsed.iterdir():
        data = pd.read_csv(path_episode, sep='|').values.tolist()
        for line in data:
            character = line[0]
            sentence = line[1]
            details['sentence_count'][character] = details['sentence_count'][character] + 1
            for word in sentence.split(' '):
                word_dict = dict.get(word, default.copy())
                word_dict[character] = word_dict[character] + 1
                details['word_count'][character] = details['word_count'][character] + 1
                dict[word] = word_dict

    pd.DataFrame.from_dict(dict, orient='index').to_csv(path_dictionary, sep='|', header=list(default.keys()))
    pd.DataFrame.from_dict(details, orient='index').to_csv(path_details, sep='|', header=list(default.keys()))


def normalize_dictionary():
    print("Normalizing dictionary")
    dict = pd.read_csv(path_dictionary, sep='|', index_col=0).to_dict(orient='index')
    details = pd.read_csv(path_details, sep='|', index_col=0).to_dict(orient='index')
    for word in dict:
        total = 0
        for char in dict[word]:
            perc = dict[word][char] / details['word_count'][char]
            dict[word][char] = perc
            total = total + perc
        for char in dict[word]:
            dict[word][char] /= total
    pd.DataFrame.from_dict(dict, orient='index').to_csv(path_normalized, sep='|', header=list(default.keys()))


def classify():
    print("Classifying")
    dict = pd.read_csv(path_normalized, sep='|', index_col=0).to_dict(orient='index')
    classified_path = dir_dictionary.joinpath('classified.csv')
    chance_distribution_path = dir_dictionary.joinpath('chance_distribution.csv')
    classification = []
    chance_distribution = []

    for path_episode in dir_transcripts_parsed.iterdir():
        data = pd.read_csv(path_episode, sep='|').values.tolist()
        for line in data:
            true_character = line[0]
            sentence = line[1]
            chance_per_char = default.copy()
            for word in sentence.split(' '):
                word_dic = dict.get(word, default.copy())
                for char in chance_per_char:
                    chance_per_char[char] += word_dic[char]
            for char in chance_per_char:
                chance_per_char[char] /= sum(chance_per_char.values())
            chance_distribution.append(chance_per_char)
            classified_character = max(chance_per_char, key=chance_per_char.get)
            classification.append([true_character, classified_character])
    pd.DataFrame(classification).to_csv(classified_path, sep='|', header=['true', 'classified'], index=False)
    pd.DataFrame(chance_distribution).to_csv(chance_distribution_path, sep='|', header=None, index=False)

def check_correctness():
    print("Checking correctness")
    classified_path = dir_dictionary.joinpath('classified.csv')
    file = pd.read_csv(classified_path, sep='|', index_col=False)

    accuracy = metrics.accuracy_score(file['true'], file['classified'])
    confusion_matrix_result = confusion_matrix(file['true'], file['classified'])
    accuracy_percentage = 100 * accuracy
    print('Accuracy percentage is ' + str(accuracy_percentage))
    print(confusion_matrix_result)

create_dictionary()
normalize_dictionary()
classify()
check_correctness()