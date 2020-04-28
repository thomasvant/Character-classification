import re
import csv
from bs4 import BeautifulSoup
import os.path as op
import os
import pandas as pd

globalInvalidTags = ['font', 'em', 'i', 'strong', 'b']
mainCharacters = {'Rachel': 1, 'Monica': 2, 'Joey': 3, 'Chandler': 4, 'Phoebe': 5, 'Ross': 6}

mainDir = op.abspath(op.join(__file__,op.pardir))
transcriptDir, parsedTranscriptsDir = op.join(mainDir, 'transcripts'), op.join(mainDir, 'parsedTranscripts')

if not op.exists(parsedTranscriptsDir):
    os.makedirs(parsedTranscriptsDir)

for dir in os.listdir(transcriptDir):
    seasonDir, parsedSeasonDir = op.join(transcriptDir, dir), op.join(parsedTranscriptsDir, dir)

    if not op.exists(parsedSeasonDir):
        os.makedirs(parsedSeasonDir)

    for file in os.listdir(seasonDir):
        episodePath, parsedEpisodePath = op.join(seasonDir, file), op.join(parsedSeasonDir, op.splitext(file)[0] + '.csv')
        curFile = open(episodePath)
        print(episodePath)
        data = []

        documentSoup = BeautifulSoup(curFile, 'html.parser')
        for paragraphTag in documentSoup.select('p'):
            paragraphString = str(paragraphTag).replace('\n', ' ').replace(u'\xa0', '')
            paragraphStringNoDirections = re.sub(r'\([^\)]*\)', '', paragraphString) # remove scene directions
            paragraphStringNoScene = re.sub(r'\[[^\]]*\]', '', paragraphStringNoDirections) # remove scene explanations

            paragraphSoup = BeautifulSoup(paragraphStringNoScene,'html.parser')
            bold = paragraphSoup.select('b')
            strong = paragraphSoup.select('strong')

            if bold or strong: # characters are placed in either strong or bold tags, all other lines can be diregarded
                for tag in globalInvalidTags:
                    for match in paragraphSoup.findAll(tag):
                        match.replaceWithChildren() # strip the line from all tags, necessary due to irregularities in the files
                try:
                    characters, line = paragraphSoup.getText().split(':', 1)  # split characters from line, only split once since the line can contain : as well (e.g. in 2:30am)
                    multipleCharacters = re.split(', and |, | and ', characters) # sometimes, multiple characters say the same sentence, so they should be split. too

                    for curCharacter in multipleCharacters:
                        characterId = mainCharacters.get(curCharacter, 0) # assign id to character, default (for non main characters) is 0
                        data.append([paragraphStringNoScene, curCharacter, characterId, line])
                        parsedTranscriptsDataFrame = pd.DataFrame(data)
                        parsedTranscriptsDataFrame.to_csv(parsedEpisodePath, sep='|', header=False, index=False)
                except ValueError: # credits etc. are in bold tags, so we remove them here
                    continue