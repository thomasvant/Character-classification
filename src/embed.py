import pandas as pd
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import pathlib
import time
import sister


dir_transcript_parsed = pathlib.Path.cwd().parent.joinpath('transcripts_parsed')


def write_to_file(data, path):
    pd.DataFrame(data).to_csv(path, sep='|', header=False, index=False)

def elmo():
    tf.disable_eager_execution()
    dir_transcript_embedded = pathlib.Path.cwd().parent.joinpath('transcripts_embedded').joinpath('elmo')
    dir_transcript_embedded.mkdir(parents=True, exist_ok=True)
    elmo = hub.Module("../modules/elmo3", trainable=False)


    def embed(sentences):
        embeddings = elmo(sentences, signature="default", as_dict=True)["elmo"]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings, 1))


    for path_episode in dir_transcript_parsed.iterdir():
        data = pd.read_csv(path_episode, sep='|')
        start_time = time.time()
        embedded_data = embed(data['line'].tolist())
        path_episode_embedded = dir_transcript_embedded.joinpath(path_episode.stem + '.csv')
        print('Embedding ' + str(path_episode.stem) + ' took ' + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
        write_to_file(embedded_data, path_episode_embedded)

def sisters():
    sentence_embedding = sister.MeanEmbedding(lang="en")
    dir_transcript_embedded = pathlib.Path.cwd().parent.joinpath('transcripts_embedded').joinpath('sisters')
    dir_transcript_embedded.mkdir(parents=True, exist_ok=True)

    def embed(sentences):
        embedded_sentences = [sentence_embedding(sentence) for sentence in sentences]
        return embedded_sentences

    for path_episode in dir_transcript_parsed.iterdir():
        data = pd.read_csv(path_episode, sep='|')
        start_time = time.time()
        embedded_data = embed(data['line'].tolist())
        path_episode_embedded = dir_transcript_embedded.joinpath(path_episode.stem + '.csv')
        print('Embedding ' + str(path_episode.stem) + ' took ' + str(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
        write_to_file(embedded_data, path_episode_embedded)