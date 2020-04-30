import pathlib
import pandas as pd
from bs4 import BeautifulSoup as bs
import re

dir_transcript = pathlib.Path.cwd().parent.joinpath('transcripts')
dir_transcript_parsed = pathlib.Path.cwd().parent.joinpath('transcripts_parsed')
dir_transcript_parsed.mkdir(parents=True, exist_ok=True)


def remove_details(string):
    new_string = re.sub(r'\([^\)]*\)', '', string)  # remove scene directions
    new_string = re.sub(r'\[[^\]]*\]', '', new_string)  # remove scene explanations
    return new_string


def split_char_from_line(string):
    try:
        char, line = string.split(':', 1)
        return char, line
    except ValueError:
        return None, None


def return_multiple_characters(string):
    new_string = string.lower()
    banned_names = ['scene']
    for name in banned_names:
        if bool(re.search(name, new_string)):
            print("Name found: " + new_string)
            return None
    split_char = new_string.split('+')
    return split_char


def clean_character(string):
    if string == 'all':
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
            for cur_line in lines_multiple:
                data.append([cleaned_char, cur_line])
    return data


def write_to_file(data, path):
    pd.DataFrame(data).to_csv(path, sep='|', header=False, index=False)


for path_episode in dir_transcript.iterdir():
    print("Parsing episode " + path_episode.stem)
    data = parse_episode(open(path_episode))
    path_episode_parsed = dir_transcript_parsed.joinpath(path_episode.stem + '.csv')
    write_to_file(data, path_episode_parsed)
    break
