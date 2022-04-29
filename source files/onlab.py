import collections
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import sys

# Importing data files
discord_voice_good = pd.read_csv("dc_hang_ossz.csv", sep=';', header=None)
discord_ctrl_good = pd.read_csv("dc_ctrl_ossz.csv", sep=';', header=None)
discord_ctrl_good = discord_ctrl_good.fillna(0) # fill NaN cells with 0

if discord_voice_good.empty:
    print("empty1")
else:
    print("ok1")

if discord_ctrl_good.empty:
    print("empty11")
else:
    print("ok11")

discord_voice_bad = pd.read_csv("dc_hang_ossz.csv", sep=';', header=None)
discord_ctrl_bad = pd.read_csv("dc_ctrl_ossz.csv", sep=';', header=None)
discord_ctrl_bad = discord_ctrl_bad.fillna(0) # fill NaN cells with 0

if discord_voice_bad.empty:
    print("empty2")
else:
    print("ok2")

if discord_ctrl_bad.empty:
    print("empty22")
else:
    print("ok22")

# Data preprocessing
data_help = discord_voice_good.to_numpy() #numpy array only used for getting size of array

df = pd.DataFrame(data=discord_voice_good)
TimeSinceLastPacket_good = np.empty([np.shape(data_help)[0], np.shape(data_help)[1]])

#TODO 4 for ciklus különböző táblázatot kap meg és abból 4 külöbözőt gyárt le ugyanazzal a folyamattal
#TODO Ha van rá mód lerövidíteni

# For loop for filling in TimeSinceLastPacket_bad ndarray
for i, row in df.iterrows(): # iterating in rows where i is the row number, row is the data in a whole row
    col_iter = 0 # column iterator, 0 at the beginning of each row
    TSLP = 0
    for cell in row: # iterating in columns where cell is one cell of the row
        if cell < 30: # if there are less than 30 packets/s TSLP counts up otherwise it's 0
            TSLP += 1
        else:
            TSLP = 0
        TimeSinceLastPacket_good[i, col_iter] = TSLP # filling the result table used later
        col_iter += 1

df = pd.DataFrame(data=discord_ctrl_good)
TimeSinceLastBit_good = np.empty([np.shape(data_help)[0], np.shape(data_help)[1]])
# For loop for filling in TimeSinceLastBit_good ndarray
for i, row in df.iterrows():
    col_iter = 0
    TSLB = 0
    for cell in row:
        if cell == 0:
            TSLB += 1
        else:
            TSLB = 0
        TimeSinceLastBit_good[i, col_iter] = TSLB
        col_iter += 1

df = pd.DataFrame(data=discord_voice_bad)
TimeSinceLastPacket_bad = np.empty([np.shape(data_help)[0], np.shape(data_help)[1]])
# For loop for filling in TimeSinceLastPacket_bad ndarray
for i, row in df.iterrows():
    col_iter = 0
    TSLP = 0
    for cell in row:
        if cell < 30:
            TSLP += 1
        else:
            TSLP = 0
        TimeSinceLastPacket_bad[i, col_iter] = TSLP
        col_iter += 1

df = pd.DataFrame(data=discord_ctrl_bad)
TimeSinceLastBit_bad = np.empty([np.shape(data_help)[0], np.shape(data_help)[1]])
# For loop for filling in TimeSinceLastBit_bad ndarray
for i, row in df.iterrows():
    col_iter = 0
    TSLB = 0
    for cell in row:
        if cell == 0:
            TSLB += 1
        else:
            TSLB = 0
        TimeSinceLastBit_bad[i, col_iter] = TSLB
        col_iter += 1

discord_good = pd.DataFrame() # final DataFrames into which the preprocessed data goes
discord_bad = pd.DataFrame()
for i in range(discord_voice_good.shape[0]):
    # 1 row of the original data put into a column from each of the inputs
    discord_good[0] = discord_voice_good.to_numpy()[i]
    discord_good[1] = TimeSinceLastPacket_good[i]
    discord_good[2] = discord_ctrl_good.to_numpy()[i]
    discord_good[3] = TimeSinceLastBit_good[i]
    discord_good.columns = ['voice', 'TSLP', 'ctrl', 'TSLB']

    discord_bad[0] = discord_voice_good.to_numpy()[i]
    discord_bad[1] = TimeSinceLastPacket_good[i]
    discord_bad[2] = discord_ctrl_good.to_numpy()[i]
    discord_bad[3] = TimeSinceLastBit_good[i]
    discord_bad.columns = ['voice', 'TSLP', 'ctrl', 'TSLB']

    discord_good['type'] = 1
    discord_bad['type'] = 0

    discord_full = pd.concat([discord_good, discord_bad], ignore_index=True)

    X = discord_full.iloc[:, 0:3]

    y = np.ravel(discord_full.type)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

#TODO kitalálni mi van a for ciklusban

model = Sequential()

model.add(Dense(4, activation='relu', input_shape=(37,)))

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
