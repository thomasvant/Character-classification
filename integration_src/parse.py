from bs4 import BeautifulSoup as bs
import re
import pandas as pd
import integration_src.file_manager as fm
import integration_src.download as dl


def parse_episodes(episode_array=None):
    if episode_array is None:
        episode_array = fm.get_transcripts()
    if episode_array is None:
        dl.download_episodes()
        episode_array = fm.get_transcripts()
    print("Parsing episodes")
    p = {"parsed": pd.concat(
        [pd.DataFrame(parse_episode(episode), columns=['character', 'line', 'wordcount']) for episode in episode_array],
        ignore_index=True)}
    parsed = pd.concat(p, axis=1)
    fm.write_df(parsed, "0_parsed")
    return parsed


def parse_episode(file):
    episode_soup = bs(file, 'html.parser')
    string_array = []
    for p_tag in episode_soup.select('p'):
        string_array.extend(parse_string(str(p_tag.text)))
    return string_array


def parse_string(string):
    string = remove_new_lines(string)
    string = decapitalize(string)
    string = replace_words(string)
    string = remove_scene_directions(string)
    character, line = split_sentence_from_character(string)
    character_array = split_and_process_characters(character)
    line_array = split_and_process_lines(line)
    string_array = []
    if character_array and line_array is not None:
        for character in character_array:
            for line in line_array:
                string_array.append([character, line, words_per_line(line)])
    return string_array


def remove_new_lines(string):
    return string.replace('\n', ' ')


def replace_words(string):
    word_mappings = {
        's*x': 'sex',
        'c.h.e.e.s.e.': 'cheese',
        "t.g.i.friday's": "tgifridays",
        'dr.': 'doctor',
        'f.y.i.': 'fyi',
        'p.m.': '',
        'a.m.': '',
        's.a.t.s.': 'sats',
        "’" : "'"
    }
    for k, v in word_mappings.items():
        string = string.replace(k, v)
    return string


def decapitalize(string):
    return string.lower()


def remove_scene_directions(string):
    string = re.sub(r'\([^\)]*\)', '', string)
    string = re.sub(r'\[[^\]]*\]', '', string)
    return string


def split_sentence_from_character(string):
    try:
        char, line = string.split(':', 1)
        return char, line
    except ValueError:
        return None, None


def split_and_process_characters(string):
    if string is None:
        return None
    main_characters = ['chandler', 'joey', 'monica', 'phoebe', 'rachel', 'ross']
    banned_names = ['scene', 'all', 'written by', 'author', 'post subject', 'teleplay by', 'story by', 'transcribed by']
    for name in banned_names:
        if bool(re.search(name, string)):
            return None
    character_array = string.split('+')
    new_character_array = []
    for character in character_array:
        if character in main_characters:
            new_character_array.append(character)
    return new_character_array


def words_per_line(string):
    return len(string.split(' '))


def split_and_process_lines(string):
    if string is None:
        return None
    lines = []
    for line in re.split("[.?!]", string):
        line = remove_punctuation(line)
        line = remove_redundant_spaces(line)
        if words_per_line(line) > 1:
            lines.append(line)
    lines = list(filter(None, lines))
    return lines


def remove_punctuation(string):
    banned_punctuation = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~—…“”'  # all punctuation except '
    for punctuation in banned_punctuation:
        string = string.replace(punctuation, ' ')
    string = re.sub('[0-9]', '', string)
    string = re.sub(" '|' | ' ", ' ', string)
    return string


def remove_redundant_spaces(string):
    string = re.sub('( +)', ' ', string).strip()
    return string
