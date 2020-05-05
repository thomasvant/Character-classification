import pathlib
import pandas as pd


def main():
    created_files_dir = pathlib.Path.cwd().parent.joinpath('created_files')
    parsed_path = created_files_dir.joinpath('parsed.csv')
    preprocessed_path = created_files_dir.joinpath('preprocessed.csv')

    parsed_data = pd.read_csv(parsed_path, sep='|', index_col=False)
    preprocessed_data = []

    for line in parsed_data:
        preprocessed_data.append(line['character'], preprocess_line(line['line']))

    pd.DataFrame(preprocessed_data).to_csv(preprocessed_path, sep='|', header=['character', 'line'], index=False)


def preprocess_line(line):
    pass


main()
