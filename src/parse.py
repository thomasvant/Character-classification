from bs4 import BeautifulSoup as bs
import re
import pandas as pd
import src.file_manager as fm
from nltk.corpus import stopwords
from autocorrect import Speller
from nltk.stem import PorterStemmer
from pycontractions import Contractions


__all__ = ['parse']
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
spell = Speller()
cont = Contractions(api_key="glove-twitter-25")

# General method to be called
def parse(correct_spelling=False, stemming=False, remove_stopwords=False, expand_contractions=True):
    print("Parsing episodes")

    episode_array = fm.get_transcripts()

    lines = []
    for episode in episode_array:
        lines.extend(parse_episode(episode, correct_spelling, stemming, remove_stopwords, expand_contractions))

    df = pd.DataFrame(lines, columns=["character", "line", "wordcount", "stopwordcount"])
    # Remove duplicate combinations of ["character] and ["line"] and leave only 1
    # Remove duplicate ["line"] since this would mean multiple characters say the same sentence
    df = df[~df.duplicated(subset=["character", "line"])]
    df = df[~df.duplicated(subset=["line"], keep=False)]
    ml = {'parsed': df}
    ml_df = pd.concat(ml, axis=1).reindex()
    fm.write_df(ml_df, "0_parsed")
    return ml_df

# Parse single episode
def parse_episode(file, correct_spelling=False, stemming=False, remove_stopwords=False, expand_contractions=True):
    episode_soup = bs(file, 'html.parser')
    string_array = []
    for p_tag in episode_soup.select('p'):
        string_array.extend(parse_string(str(p_tag.text), correct_spelling, stemming, remove_stopwords, expand_contractions))
    return string_array


def parse_string(string, correct_spelling=False, stemming=False, remove_stopwords=False, expand_contractions=True):
    string = remove_new_lines(string)
    string = decapitalize(string)
    string = replace_words(string)
    string = remove_scene_directions(string)
    character, line = split_sentence_from_character(string)
    character_array = split_and_process_characters(character)
    line_array = split_and_process_lines(line, correct_spelling, stemming, remove_stopwords, expand_contractions)
    string_array = []
    if character_array and line_array is not None:
        for character in character_array:
            for line in line_array:
                string_array.append([character, line, words_per_line(line), count_stopwords(line)])
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
        'p.m.': 'pm',
        'a.m.': 'am',
        's.a.t.s.': 'sats',
        "’" : "'",
        "c'mon": "come on",
        "o'clock": "o clock",
        "y'know": "you know",
        "‘em": "them"
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


def split_and_process_lines(string, correct_spelling=False, stemming=False, remove_stopwords=False, expand_contractions=False):
    if string is None:
        return None
    lines = []
    for line in re.split("[.?!]", string):
        line = remove_punctuation(line)
        line = remove_redundant_spaces(line)
        if expand_contractions:
            line = contractions(line)
        line_array = line.split(" ")
        if correct_spelling:
            line_array = autocorrect(line_array)
        if stemming:
            line_array = stem(line_array)
        if remove_stopwords:
            line_array = stopwords(line_array)
        if len(line_array) > 1:
            lines.append(" ".join(line_array).lower())
    return lines


def contractions(line):
    return list(cont.expand_texts([line]))[0]


def autocorrect(string_array):
    return [spell(w) for w in string_array]


def stem(string_array):
    return [ps.stem(w) for w in string_array]


def stopwords(string_array):
    return [w for w in string_array if not w in stop_words]


def count_stopwords(string):
    return len([w for w in string.split(" ") if w in stop_words])


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
