import pandas as pd
import pathlib
import time

dir_transcripts_parsed = pathlib.Path.cwd().parent.joinpath('transcripts_parsed')
dir_dictionary = pathlib.Path.cwd().parent.joinpath('dictionary')
dir_dictionary.mkdir(parents=True, exist_ok=True)

def create_dictionary():
    path_dictionary = dir_dictionary.joinpath('dictionary_occurrence.csv')
    path_details = dir_dictionary.joinpath('details.csv')
    dict = {}
    default = {'joey': 0, 'rachel': 0, 'monica': 0, 'phoebe': 0, 'chandler': 0, 'ross': 0, 'other': 0}
    details = {'word_count': default.copy(), 'sentence_count': default.copy()}

    for path_episode in dir_transcripts_parsed.iterdir():
        print(path_episode)
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