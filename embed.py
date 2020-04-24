import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

tf.disable_eager_execution()

elmo = hub.Module("modules/elmo3", trainable=True)
vecs = []

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embeddings = session.run(
        elmo(
            ["the cat is on the mat", "dogs are in the fog"],
            signature="default",
            as_dict=True)["elmo"])
    vecs.extend(embeddings)

print(vecs)