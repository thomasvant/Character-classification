import re
import csv
from bs4 import BeautifulSoup
import os.path as op
import os
import pandas as pd
from contractions import CONTRACTION_MAP
import string


def create_dir_if_not_exists(dir):
    if not op.exists(dir):
        os.makedirs(dir)





def parse_string(string):
    new_string = string.replace('\n', ' ').replace(u'\xa0', '')
    new_string = re.sub(r'\([^\)]*\)', '', new_string)  # remove scene directions
    new_string = re.sub(r'\[[^\]]*\]', '', new_string)  # remove scene explanations
    return new_string


def strip_tags(soup):
    invalid_tags = ['font', 'em', 'i', 'strong', 'b']
    for tag in invalid_tags:
        for match in soup.findAll(tag):
            match.replaceWithChildren()  # strip the line from all tags, necessary due to irregularities in the files


# From GitHub user dipanjanS
# https://github.com/dipanjanS/text-analytics-with-python/tree/master/Old-First-Edition/source_code/Ch03_Processing_and_Understanding_Text
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = expanded_contraction
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def preprocess_string(string):
    new_string = re.sub(r'\d+', '', string)  # remove numbers
    new_string = new_string.lower()
    new_string = expand_contractions(new_string)
    new_string = re.sub(r'[^\w\s]', ' ', new_string)
    new_string = re.sub(' +', ' ', new_string)
    new_string = new_string.strip()
    return new_string


def split_line(string):
    lines = re.split("[.?!]", string)  # split line on punctuation
    lines = [line.strip().lower() for line in lines]  # strip line from preceding and appending white space
    lines = list(filter(None, lines))
    return lines


def parse_episode(path_episode):
    data = []
    soup_doc = BeautifulSoup(open(path_episode), 'html.parser')
    for tag_p in soup_doc.select('p'):
        soup_p_parsed = BeautifulSoup(parse_string(str(tag_p)), 'html.parser')
        try:
            characters, line = soup_p_parsed.get_text().split(':',
                                                              1)  # split characters from line, only split once since the line can contain : as well (e.g. in 2:30am)
            character_array = re.split(', and |, | and ',
                                       characters)  # sometimes, multiple characters say the same sentence, so they should be split. too

            for character in character_array:
                if character == "All":  # remove lines that are said by all because it is not clear who is meant by all
                    continue
                id = main_characters.get(character,
                                         0)  # assign id to character, default (for non main characters) is 0
                for cur_line in split_line(line):
                    processed_string = preprocess_string(cur_line)
                    data.append([character, id, cur_line, processed_string])

        except ValueError:  # credits etc. are in bold tags, so we remove them here
            continue
    return data


def write_to_file(data, file_path):
    pd.DataFrame(data).to_csv(file_path, sep='|', header=["char_name", "char_id", "unprocessed_line", "processed_line"], index=False)

main_characters = {'Rachel': 1, 'Monica': 2, 'Joey': 3, 'Chandler': 4, 'Phoebe': 5, 'Ross': 6}

dir_main = op.abspath(op.join(__file__, op.pardir))
dir_transcript, dir_transcript_parsed = op.join(dir_main, 'transcripts_old'), op.join(dir_main, 'parsedTranscripts')

create_dir_if_not_exists(dir_transcript_parsed)
# parse_episode("C:\\Users\\Thomas\\Documents\\Studie\\Y4\\Q4\\Research Project\\python\\transcripts_old\\season01\\0116.html")

for dir in os.listdir(dir_transcript):
    dir_season, dir_season_parsed = op.join(dir_transcript, dir), op.join(dir_transcript_parsed, dir)

    create_dir_if_not_exists(dir_season_parsed)

    for file in os.listdir(dir_season):
        path_episode, path_episode_parsed = op.join(dir_season, file), op.join(dir_season_parsed,
                                                                               op.splitext(file)[0] + '.csv')

        print(op.abspath(path_episode))
        write_to_file(parse_episode(path_episode), path_episode_parsed)
        print("Parsed " + op.basename(path_episode_parsed))
