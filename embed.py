import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import os
import os.path as op
import csv
import pandas as pd

tf.disable_eager_execution()

elmo = hub.Module("modules/elmo3", trainable=True)
vecs = []

def elmo_vectors(x):
  embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

mainDir = op.abspath(op.join(__file__,op.pardir))
parsedTranscriptsDir, embeddedTranscriptsDir = op.join(mainDir, 'parsedTranscripts'), op.join(mainDir, 'embeddedTranscripts'),

if not op.exists(embeddedTranscriptsDir):
    os.makedirs(embeddedTranscriptsDir)

for dir in os.listdir(parsedTranscriptsDir):
    parsedSeasonDir, embeddedSeasonDir = op.join(parsedTranscriptsDir, dir), op.join(embeddedTranscriptsDir, dir)

    if not op.exists(embeddedSeasonDir):
        os.makedirs(embeddedSeasonDir)

    for file in os.listdir(parsedSeasonDir):
        parsedEpisodePath, embeddedEpisodePath = op.join(parsedSeasonDir, file), op.join(embeddedSeasonDir, file)
        parsedTranscript = pd.read_csv(parsedEpisodePath, sep='|', usecols=[0,1], names=['character', 'line'], header=None)
        print(parsedTranscript['line'].tolist())
        embeddedTranscriptLines = elmo_vectors(parsedTranscript['line'].tolist())
        embeddedTranscriptDataFrame = pd.DataFrame(embeddedTranscriptLines)
        embeddedTranscriptDataFrame.to_csv(embeddedEpisodePath, sep='|', header=False, index=False)
        break
    break
        # with open(parsedEpisodePath, 'w', newline='') as parsedCurFile, open(embeddedEpisodePath, 'w', newline='') as embeddedCurFile:
        #     fileReader = csv.reader(parsedCurFile, delimiter='|')
        #     fileWriter = csv.writer(embeddedCurFile, delimiter='|')