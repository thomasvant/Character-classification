import pandas as pd
# import tensorflow_hub as hub
# import tensorflow.compat.v1 as tf
import pathlib
import time
import sister
import numpy as np

dir_transcript_parsed = pathlib.Path.cwd().parent.joinpath('created_files')


def main():
    created_files_dir = pathlib.Path.cwd().parent.joinpath('created_files')
    parsed_path = created_files_dir.joinpath('processed.csv')
    embedded_sisters_path = created_files_dir.joinpath('embedded_sisters_word2vec.csv')
    embedded_elmo_path = created_files_dir.joinpath('embedded_elmo.csv')

    sisters_dataframe = sisters(parsed_path)
    print(type(sisters_dataframe))
    print(sisters_dataframe.shape)
    sisters_dataframe.to_csv(embedded_sisters_path, sep='|', header=None)
    # elmo_dataframe.to_csv(embedded_elmo_path, sep='|', header=None)


def sisters(path):
    sentence_embedding = sister.MeanEmbedding(lang="en")

    parsed_data = pd.read_csv(path, sep='|', index_col=0)
    print(type(parsed_data))
    print(parsed_data.shape)

    # Pandas series with numpy array values to dataframe
    return pd.DataFrame.from_records(parsed_data['line'].apply(sentence_embedding))


def elmo(path):
    parsed_data = pd.read_csv(path, sep='|', index_col=0)
    # parsed_data = parsed_data.head(100)
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

    data_split = [parsed_data['line'][i:i + 100] for i in range(0, parsed_data.shape[0], 100)]
    cur = 1
    total = len(data_split)
    embedded_data = []
    for x in data_split:
        split_time = time.time()
        embedded_data.append(embed(x))
        print("Embedded batch " + str(cur) + " of " + str(total) + " in " + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - split_time))))
        cur += 1
    embedded_data = np.concatenate(embedded_data, axis=0)
    # embedded_data = np.concatenate([embed(x) for x in data_split], axis=0)

    # embedded_data = embed(parsed_data['line'].tolist())
    print('Elmo embedding took ' + str(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))
    return pd.DataFrame(embedded_data)


main()
