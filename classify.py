import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import os.path as op

mainDir = op.abspath(op.join(__file__,op.pardir))
parsedTranscriptsDir, embeddedTranscriptsDir = op.join(mainDir, 'parsedTranscripts'), op.join(mainDir, 'embeddedTranscripts'),

for dir in os.listdir(parsedTranscriptsDir):
    parsedSeasonDir, embeddedSeasonDir = op.join(parsedTranscriptsDir, dir), op.join(embeddedTranscriptsDir, dir)

    for file in os.listdir(parsedSeasonDir):
        parsedEpisodePath, embeddedEpisodePath = op.join(parsedSeasonDir, file), op.join(embeddedSeasonDir, file)
        parsedTranscript = pd.read_csv(parsedEpisodePath, sep='|', header=None)
        embeddedTranscript = pd.read_csv(embeddedEpisodePath, sep='|', header=None)

        x_train, x_test, y_train, y_test = train_test_split(embeddedTranscript, parsedTranscript[2], random_state=10)
        logistic_regression = LogisticRegression()
        logistic_regression.fit(x_train, y_train)
        y_pred = logistic_regression.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracy_percentage = 100 * accuracy
        print(accuracy_percentage)
        break
    break