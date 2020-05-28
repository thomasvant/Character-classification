import pandas as pd
import time
import numpy as np
import src.file_manager as fm
import sister
from sister import word_embedders

__all__ = ["embed"]

def embed():
    print("Embedding transcripts")
    data = fm.get_df("0_parsed")
    sentence_embedding = sister.MeanEmbedding(lang="en")
    embedded = data["parsed"]["line"].apply(sentence_embedding)
    d = {"embedded": pd.DataFrame.from_records(embedded, index=embedded.index)}
    embedded = data.join(pd.concat(d, axis=1))

    fm.write_df(embedded, "1_embedded_fasttext")
    return embedded

def sisters(data, technique="fasttext"):

    return


def elmo(data):
    import tensorflow_hub as hub
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
    elmo = hub.Module("modules/elmo3", trainable=False)

    def embed(sentences):
        embeddings = elmo(sentences, signature="default", as_dict=True)["elmo"]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings, 1))

    start_time = time.time()

    data_split = [data[i:i + 50] for i in range(0, data.shape[0], 50)]
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
