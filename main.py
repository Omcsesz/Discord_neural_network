import collections
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import sys

# Importing data files
discord_gud = pd.read_csv("dc_hang_ossz.csv", sep=';', header=None)
discord_ctrl_gud = pd.read_csv("dc_ctrl_ossz.csv", sep=';', header=None)

if discord_gud.empty:
    print("empty1")
else:
    print("ok1")

if discord_ctrl_gud.empty:
    print("empty11")
else:
    print("ok11")

discord_bad = pd.read_csv("dc_hang_ossz.csv", sep=';', header=None)
discord_ctrl_bad = pd.read_csv("dc_ctrl_ossz.csv", sep=';', header=None)

if discord_bad.empty:
    print("empty2")
else:
    print("ok2")

if discord_ctrl_bad.empty:
    print("empty22")
else:
    print("ok22")

sys.setrecursionlimit(2000)

# Data preprocessing
src_array = ["discord_gud", "discord_ctrl_gud"]
asd = pd.DataFrame(data=src_array)
for file_no in range(len(src_array)):
    print(src_array[file_no])

    df = pd.DataFrame(data=discord_gud)
    data_help = df.to_numpy()
# average = (np.average(data_help, axis=1))
# shape = np.shape(data_help)
    TimeSinceLastPacket = [0, 0]
    TimeSinceLastPacket[file_no] = np.empty([np.shape(data_help)[0], np.shape(data_help)[1]])
# data = pd.DataFrame(average)
# For loop for filling in TimeSinceLastPacket ndarray
    for i, row in df.iterrows():
    # print(i, row)
        col_iter = 0
        TSLP = 0
        for j in row:
            if j < 30:
                TSLP += 1
                TimeSinceLastPacket[i, col_iter] = TSLP
            else:
                TSLP = 0
                TimeSinceLastPacket[i, col_iter] = TSLP
            col_iter += 1

print(discord_gud.info())
print(discord_bad.info())

discord_gud['type'] = 1

discord_bad['type'] = 0

# discord_full = discord_gud.append(discord_bad, ignore_index=True)
discord_full = pd.concat([discord_gud, discord_bad], ignore_index=True)

X = discord_full.iloc[:, 0:37]

y = np.ravel(discord_full.type)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(1, activation='relu', input_shape=(37,)))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.output_shape

model.summary()

model.get_config()

model.get_weights()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose=1)

print(score)
