import pandas as pd
import time
import sister
from sister import word_embedders
import numpy as np
import src.file_manager as fm
import src.parse as parse

techniques = ['fasttext', 'word2vec', 'elmo']

__all__ = ["embed"]

def embed(technique="fasttext", unique=False):
    if technique not in techniques:
        raise ValueError("Invalid embedding type. Expected one of: %s" % technique)
    print("Embedding transcripts using " + technique)
    data = fm.get_df(technique, unique=unique)

    if technique == "fasttext" or technique == "word2vec":
        embedded = sisters(data["parsed"]["line"], technique)
    else:
        embedded = elmo(data["parsed"]["line"])

    d = {"embedded": pd.DataFrame.from_records(embedded)}
    embedded = data.join(pd.concat(d, axis=1))

    fm.write_df(embedded, "1_embedded_" + type)
    return embedded


def sisters(data, technique="fasttext"):
    if technique == "fasttext":
        word_embedder = sister.word_embedders.FasttextEmbedding("en")
    else:
        word_embedder = sister.word_embedders.Word2VecEmbedding("en")

    sentence_embedding = sister.MeanEmbedding(lang="en", word_embedder=word_embedder)

    return data.apply(sentence_embedding)


def elmo(data):
    import tensorflow_hub as hub
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
    elmo = hub.Module("../modules/elmo3", trainable=False)

    def embed(sentences):
        embeddings = elmo(sentences, signature="default", as_dict=True)["elmo"]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings, 1))

    start_time = time.time()

    data_split = [data[i:i + 100] for i in range(0, data.shape[0], 100)]
    cur = 1
    total = len(data_split)
    embedded_data = []
    for x in data_split:
        split_time = time.time()
        embedded_data.append(embed(x))
        print("Embedded batch " + str(cur) + " of " + str(total) + " in " + str(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - split_time))))
        cur += 1
    embedded_data = np.concatenate(embedded_data, axis=0)

    print('Elmo embedding took ' + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
    return pd.DataFrame(embedded_data)
