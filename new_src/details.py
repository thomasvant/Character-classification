import pathlib
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

created_files_dir = pathlib.Path.cwd().parent.joinpath('created_files')


def main():
    # parsed_path = created_files_dir.joinpath('parsed.csv')
    # parsed_data = pd.read_csv(parsed_path, sep='|')

    classification_tfidf_path = created_files_dir.joinpath('classification_tfidf.csv')
    classification_sisters_path = created_files_dir.joinpath('classification_sisters.csv')
    sisters = pd.read_csv(classification_sisters_path, sep='|', index_col=0)
    tfidf = pd.read_csv(classification_tfidf_path, sep='|', index_col=0)

    # display_failure_per_line_length(sisters_data)
    # display_failure_per_line_length(tfidf_data)
    display_failure_per_line_length([(sisters, "Sisters"), (tfidf, "TF-IDF")])
    display_rank([(sisters, "Sisters"), (tfidf, "TF-IDF")])
    # lines_per_character(sisters)
    plt.show()


def display_lines_length(data):
    dictionary = {}
    for line in data['lines']:
        length = len(line.split(' '))
        count = dictionary.get(length, 0) + 1
        dictionary[length] = count
    x = [pair[0] for pair in dictionary.items()]
    y = [pair[1] for pair in dictionary.items()]
    plt.bar(x, y)
    plt.show()


def lines_per_character(data):
    dictionary = {}
    for rank in data['character']:
        count = dictionary.get(rank, 0) + 1
        dictionary[rank] = count
    x = [pair[0] for pair in dictionary.items()]
    y = [pair[1] for pair in dictionary.items()]
    plt.xlabel("Character")
    plt.ylabel("Amount of lines")
    plt.bar(x, y)
    plt.show()

def display_failure_per_line_length(data_array):
    for data in data_array:
        dictionary = {}
        for index, line in data[0].iterrows():
            # print(line)
            failure = line['failure']
            line_length = line['line_length']
            length_failure = dictionary.get(line_length, [])
            length_failure.append(failure)
            dictionary[line_length] = length_failure
        # print(dictionary)
        # new_dictionary = {print(v) for k, v in dictionary.items()}

        new_dictionary = {k: np.mean(v) for k, v in dictionary.items()}
        # print(new_dictionary)
        x = [pair[0] for pair in new_dictionary.items()]
        y = [pair[1] for pair in new_dictionary.items()]
        plt.xlabel("Words per line")
        plt.ylabel("Average loss")
        plt.scatter(x, y)
    labels = [cur[1] for cur in data_array]
    plt.legend(labels)
    plt.show()


def display_rank(data_array):
    width = 1.0/len(data_array)-0.2 #create space around bars
    test = 0
    for data in data_array:
        dictionary = {}
        for rank in data[0]['rank']:
            count = dictionary.get(rank, 0) + 1
            dictionary[rank] = count
        x = [pair[0] for pair in dictionary.items()]
        y = [pair[1] for pair in dictionary.items()]
        r = [cur + width*(test-0.5) for cur in x]
        print(x)
        print(r)
        plt.xlabel("Rank of character")
        plt.ylabel("Amount")
        plt.bar(r, y, width=width)
        test += 1
    labels = [cur[1] for cur in data_array]
    plt.legend(labels)
    plt.show()


main()
