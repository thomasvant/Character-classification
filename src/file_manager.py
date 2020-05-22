import pathlib
from pathlib import Path

import pandas as pd

__all__ = ['write_transcripts', 'get_transcripts', 'write_df', 'get_df']

transcripts_dir = pathlib.Path.cwd().joinpath('transcripts')
created_files_dir = pathlib.Path.cwd().joinpath('created_files')


def write_transcripts(transcript_files):
    from bs4 import BeautifulSoup as bs
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    for file in transcript_files:
        bs_file = bs(file, "html.parser")
        ep = bs_file.select('.topic')[0].string.split(' ')[0]
        with open(transcripts_dir.joinpath(ep + ".html"), 'wb') as f:
            f.write(file)


def get_transcripts():
    return [open(path, encoding="utf8") for path in transcripts_dir.iterdir()]


def write_df(df: pd.DataFrame, filename: str):
    created_files_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(created_files_dir.joinpath(filename + ".csv"), sep='|')


def get_df(filename, unique=False):
    path = created_files_dir.joinpath(filename + ".csv")
    if path.is_file():
        data = pd.read_csv(created_files_dir.joinpath(filename + ".csv"), header=[0,1], sep='|', index_col=0)
        if unique:
            data = data[data["parsed"]["is_unique"]]
        return data
    else:
        return None
