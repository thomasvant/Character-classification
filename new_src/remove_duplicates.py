import pathlib
import pandas as pd

created_files_dir = pathlib.Path.cwd().parent.joinpath('created_files')


def main():
    parsed_path = created_files_dir.joinpath('parsed.csv')
    parsed_data = pd.read_csv(parsed_path, sep='|', index_col=0)
    processed_path = created_files_dir.joinpath('processed.csv')
    parsed_data.drop_duplicates(subset="line", keep=False, inplace=True)
    parsed_data.to_csv(processed_path, sep='|')
    # print(parsed_data)


main()
