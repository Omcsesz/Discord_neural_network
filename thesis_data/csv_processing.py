import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger(__name__)


def TimeSinceLastZero(input_table):
    data_help = input_table.to_numpy()  # numpy array only used for getting size of array
    TimeSinceLastX = np.empty([np.shape(data_help)[0], np.shape(data_help)[1]])
    df = pd.DataFrame(data=input_table)

    for row_no, row in df.iterrows(): # iterating in rows where row_no is the row number, row is the data in a whole row
        col_iter = 0  # column iterator, 0 at the beginning of each row
        TSLX = 0
        for cell in row:  # iterating in columns where cell is one cell of the row
            if cell == 0:
                TSLX += 1
            else:
                TSLX = 0
            # if input_table.equals(discord_ctrl_bad) or input_table.equals(discord_ctrl_good): # counting bits or packets
            #    if cell == 0: # if there are no bits/s TSLB counts up otherwise it's 0
            #        TSLX += 1
            #    else:
            #        TSLX = 0
            # else:
            #    if cell < 40:  # if there are less than 40 packets/s TSLP counts up otherwise it's 0
            #        TSLX += 1
            #    else:
            #        TSLX = 0
            TimeSinceLastX[row_no, col_iter] = TSLX  # filling the result table used later
            col_iter += 1
    return TimeSinceLastX


# Importing data files
def Import_Files():
    kbit = [80, 90, 100, 0]
    loss = [10, 0]
    udp_data = {}
    tcp_data = {}
    quality = 1.5
    current_dir = os.path.dirname(os.path.realpath(__file__))
    for i_kbit in kbit:
        for i_loss in loss:
            #            if i_kbit == 70 and i_loss == 10:
            #                continue
            udp_data[quality] = pd.read_csv(f"{current_dir}/csv/udp/{i_kbit}_{i_loss}_udp.csv", sep=';', header=None)
            tcp_data[quality] = pd.read_csv(f"{current_dir}/csv/tcp/{i_kbit}_{i_loss}_tcp.csv", sep=';', header=None)
            quality += 0.5
    return udp_data, tcp_data


def process():
    udp_data, tcp_data = Import_Files()
    TimeSinceLastPacket = {}
    TimeSinceLastByte = {}
    for key, input_table in udp_data.items():
        TimeSinceLastPacket[key] = TimeSinceLastZero(input_table)
    for key, input_table in tcp_data.items():
        TimeSinceLastByte[key] = TimeSinceLastZero(input_table)
    discord = pd.DataFrame()  # final DataFrames into which the preprocessed data goes
    # discord_bad = pd.DataFrame()

    for key in udp_data:
        for i in range(udp_data[key].shape[0]):
            discord_help = pd.DataFrame()
            discord_help[0] = udp_data[key].to_numpy()[i]
            discord_help[1] = TimeSinceLastPacket[key][i]
            discord_help[2] = tcp_data[key].to_numpy()[i]
            discord_help[3] = TimeSinceLastByte[key][i]
            discord_help[4] = key
            discord = pd.concat([discord, discord_help], ignore_index=True)

    discord.columns = ['voice', 'TSLP', 'ctrl', 'TSLB', 'quality']
    X = discord.iloc[:, 0:4]
    y = np.ravel(discord.quality)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

