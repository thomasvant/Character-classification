import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import os
import csv

tf.disable_eager_execution()

elmo = hub.Module("modules/elmo3", trainable=True)
vecs = []

scriptPath = os.path.abspath(__file__)  # path to python script
scriptDir = os.path.split(scriptPath)[0]  # path to python script dir
embeddedPath = os.path.join(scriptDir, "embedded.csv")  # path to transcripts dir
csvfile = open(embeddedPath, 'w', newline='')
fileWriter = csv.writer(csvfile, delimiter='|')

test = ["the car is fast", "the cat is sad", "i am happy"]
def elmo_vectors(x):
  embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

vectors = elmo_vectors(test)
for item in vectors:
    fileWriter.writerow(item)