import pathlib
import pandas as pd
from bs4 import BeautifulSoup as bs
import re
import string


dir_transcript = pathlib.Path.cwd().parent.joinpath('transcripts')
dir_transcript_parsed = pathlib.Path.cwd().parent.joinpath('transcripts_parsed')
dir_transcript_parsed.mkdir(parents=True, exist_ok=True)
punctuation = string.punctuation.replace("'", "") + "—"

def remove_details(string):
    new_string = string.replace('\n', '')
    new_string = re.sub(r'\([^\)]*\)', '', new_string)  # remove scene directions
    new_string = re.sub(r'\[[^\]]*\]', '', new_string)  # remove scene explanations
    return new_string


def split_char_from_line(string):
    try:
        char, line = string.split(':', 1)
        return char, line
    except ValueError:
        return None, None

def clean_line(string):
    # remove punctuation except for '
    new_string = string
    for x in punctuation:
        new_string = new_string.replace(x, ' ')

    # remove duplicate white spaces
    new_string = re.sub('(  +)', ' ', new_string)
    return new_string

def return_multiple_characters(string):
    new_string = string.lower()
    banned_names = ['scene']
    for name in banned_names:
        if bool(re.search(name, new_string)):
            return None
    split_char = new_string.split('+')
    return split_char


def clean_character(string):
    if string in ['all', 'written by', 'author', 'post subject', 'teleplay by', 'story by', 'transcribed by']:
        return None
    if string not in ['joey', 'rachel', 'monica', 'phoebe', 'chandler', 'ross']:
        return 'other'
    else:
        return string


def return_multiple_lines(string):
    lines = re.split("[.?!]", string)  # split line on punctuation
    lines = [line.strip().lower() for line in lines]  # strip line from preceding and appending white space
    lines = list(filter(None, lines))
    return lines


def parse_episode(episode):
    episode_soup = bs(episode, 'html.parser')
    data = []
    for p_tag in episode_soup.select('p'):
        p = remove_details(str(p_tag.text))
        if not p:
            continue
        char, line = split_char_from_line(p)
        if not char or not line:
            continue
        chars_multiple = return_multiple_characters(char)
        if not chars_multiple:
            continue
        lines_multiple = return_multiple_lines(line)
        if not lines_multiple:
            continue
        for cur_char in chars_multiple:
            cleaned_char = clean_character(cur_char)
            if not cleaned_char:
                continue
            for cur_line in lines_multiple:
                cleaned_line = clean_line(cur_line)
                data.append([cleaned_char, cleaned_line])
    return data


def write_to_file(data, path):
    pd.DataFrame(data).to_csv(path, sep='|', header=False, index=False)


for path_episode in dir_transcript.iterdir():
    print("Parsing episode " + path_episode.stem)
    data = parse_episode(open(path_episode, encoding="utf8"))
    path_episode_parsed = dir_transcript_parsed.joinpath(path_episode.stem + '.csv')
    write_to_file(data, path_episode_parsed)
