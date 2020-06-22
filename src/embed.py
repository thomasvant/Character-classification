import pandas as pd
import time
import numpy as np
import src.file_manager as fm
import sister

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