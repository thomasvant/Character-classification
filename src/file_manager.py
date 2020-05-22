import pathlib
from pathlib import Path
from typing import List, Optional

import pandas as pd

__all__ = ['write_transcripts', 'get_transcripts', 'write_df', 'get_df']

transcripts_dir: Path = pathlib.Path.cwd().parent.joinpath('transcripts')
transcripts_dir.mkdir(parents=True, exist_ok=True)

created_files_dir: Path = pathlib.Path.cwd().parent.joinpath('created_files')
created_files_dir.mkdir(parents=True, exist_ok=True)


def write_transcripts(transcript_files) -> None:
    from bs4 import BeautifulSoup as bs

    for file in transcript_files:
        ep = bs(file.content, "html.parser").select('.topic')[0].string.split(' ')[0]
        with open(transcripts_dir.joinpath(ep + ".html"), 'wb') as f:
            f.write(file.content)


def get_transcripts():
    return [open(path, encoding="utf8") for path in transcripts_dir.iterdir()]


def write_df(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(created_files_dir.joinpath(filename + ".csv"), sep='|')


def get_df(filename: str, unique=False) -> Optional[pd.DataFrame]:
    path = created_files_dir.joinpath(filename + ".csv")
    if path.is_file():
        data = pd.read_csv(created_files_dir.joinpath(filename + ".csv"), header=[0,1], sep='|', index_col=0)
        if unique:
            data = data[data["parsed"]["is_unique"]]
        return data
    else:
        return None