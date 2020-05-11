import pathlib
import pandas as pd


def main():
    created_files_dir = pathlib.Path.cwd().parent.joinpath('created_files')
    parsed_path = created_files_dir.joinpath('parsed.csv')
    preprocessed_path = created_files_dir.joinpath('preprocessed.csv')

    parsed_data = pd.read_csv(parsed_path, sep='|', index_col=False)
    preprocessed_data = []

    for char, line in parsed_data['character'], parsed_data['line']:
        preprocessed_data.append([char, preprocess_line('line')])

    pd.DataFrame(preprocessed_data).to_csv(preprocessed_path, sep='|', header=['character', 'line'], index=False)


def preprocess_line(string):
    string = expand_abbreviations(string)
    string = remove_stopwords(string)
    string = lemmatize(string)
    string = stem(string)
    return string


def expand_abbreviations(string):
    pass


def remove_stopwords(string):
    pass


def lemmatize(string):
    pass


def stem(string):
    pass


main()
