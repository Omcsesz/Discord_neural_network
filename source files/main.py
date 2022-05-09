import collections

#import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras.layers import Conv1D
from keras.layers import Dense


def TimeSinceLastZero(input_table):
    data_help = discord_voice_good.to_numpy()  # numpy array only used for getting size of array

    df = pd.DataFrame(data=input_table)
    TimeSinceLastX = np.empty([np.shape(data_help)[0], np.shape(data_help)[1]])

    # For loop for filling in TimeSinceLastPacket_bad ndarray
    for row_no, row in df.iterrows(): # iterating in rows where row_no is the row number, row is the data in a whole row
        col_iter = 0  # column iterator, 0 at the beginning of each row
        TSLX = 0
        for cell in row:  # iterating in columns where cell is one cell of the row
            if input_table.equals(discord_ctrl_bad) or input_table.equals(discord_ctrl_good): # counting bits or packets
                if cell == 0: # if there are no bits/s TSLB counts up otherwise it's 0
                    TSLX += 1
                else:
                    TSLX = 0
            else:
                if cell < 40:  # if there are less than 40 packets/s TSLP counts up otherwise it's 0
                    TSLX += 1
                else:
                    TSLX = 0
            TimeSinceLastX[row_no, col_iter] = TSLX  # filling the result table used later
            col_iter += 1
    return TimeSinceLastX


# Importing data files
discord_voice_good = pd.read_csv("dc_voice_good.csv", sep=';', header=None)
discord_ctrl_good = pd.read_csv("dc_ctrl_good.csv", sep=';', header=None)
discord_ctrl_good = discord_ctrl_good.fillna(0) # fill NaN cells with 0

if discord_voice_good.empty:
    print("empty1")
else:
    print("ok1")

if discord_ctrl_good.empty:
    print("empty11")
else:
    print("ok11")

discord_voice_bad = pd.read_csv("dc_voice_bad.csv", sep=';', header=None)
discord_ctrl_bad = pd.read_csv("dc_ctrl_bad.csv", sep=';', header=None)
discord_ctrl_bad = discord_ctrl_bad.fillna(0) # fill NaN cells with 0

if discord_voice_bad.empty:
    print("empty2")
else:
    print("ok2")

if discord_ctrl_bad.empty:
    print("empty22")
else:
    print("ok22")

TimeSinceLastPacket_good = TimeSinceLastZero(discord_voice_good)
TimeSinceLastBit_good = TimeSinceLastZero(discord_ctrl_good)
TimeSinceLastPacket_bad = TimeSinceLastZero(discord_voice_bad)
TimeSinceLastBit_bad = TimeSinceLastZero(discord_ctrl_bad)

discord_good = pd.DataFrame() # final DataFrames into which the preprocessed data goes
discord_bad = pd.DataFrame()

for i in range(discord_voice_good.shape[0]):
    discord_help_g = pd.DataFrame()
    discord_help_g[0] = discord_voice_good.to_numpy()[i]
    discord_help_g[1] = TimeSinceLastPacket_good[i]
    discord_help_g[2] = discord_ctrl_good.to_numpy()[i]
    discord_help_g[3] = TimeSinceLastBit_good[i]
    discord_good = pd.concat([discord_good, discord_help_g], ignore_index=True)

    discord_help_b = pd.DataFrame()
    discord_help_b[0] = discord_voice_bad.to_numpy()[i]
    discord_help_b[1] = TimeSinceLastPacket_bad[i]
    discord_help_b[2] = discord_ctrl_bad.to_numpy()[i]
    discord_help_b[3] = TimeSinceLastBit_bad[i]
    discord_bad = pd.concat([discord_bad, discord_help_b], ignore_index=True)

discord_good.columns = ['voice', 'TSLP', 'ctrl', 'TSLB']
discord_bad.columns = ['voice', 'TSLP', 'ctrl', 'TSLB']

discord_good['type'] = 1
discord_bad['type'] = 0

discord_full = pd.concat([discord_good, discord_bad], ignore_index=True)

X = discord_full.iloc[:, 0:4]

y = np.ravel(discord_full.type)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=133, verbose=1)

y_pred = model.predict(X_test)

for i in range(100):
    print(y_pred[i])

score = model.evaluate(X_test, y_test, verbose=1)

print(score)
