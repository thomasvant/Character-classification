import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pathlib
import time
from sklearn.metrics import confusion_matrix

dir_transcript_embedded = pathlib.Path.cwd().parent.joinpath('transcripts_embedded').joinpath('sisters')
dir_transcript_parsed = pathlib.Path.cwd().parent.joinpath('transcripts_parsed')

characters_array = []
for path_episode in dir_transcript_parsed.iterdir():
    characters_array.append(pd.read_csv(path_episode, sep='|', index_col=False)['char'])

characters_df = pd.concat(characters_array, ignore_index=True)
del characters_array
print("Chars: shape is " + str(characters_df.shape))

embedded_array = []
for path_episode in dir_transcript_embedded.iterdir():
    embedded_array.append(pd.read_csv(path_episode, sep='|', header=None, index_col=False))

embedded_df = pd.concat(embedded_array, ignore_index=True)
del embedded_array
print("Lines: shape is " + str(embedded_df.shape))

start_time = time.time()
x_train, x_test, y_train, y_test = train_test_split(embedded_df, characters_df, random_state=10)
x_train, x_test = preprocessing.scale(x_train), preprocessing.scale(x_test)
print('Train size: ' + str(y_train.shape) + '\nTest size: ' + str(y_test.shape))
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)
accuracy_percentage = 100 * accuracy
print('Accuracy percentage is ' + str(accuracy_percentage))
print(confusion_matrix_result)