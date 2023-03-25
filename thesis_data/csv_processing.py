import csv
import logging
import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.DEBUG) # format='%(levelname) [%(filename):%(lineno)] %(message)'
_logger = logging.getLogger(__name__)


def TimeSinceLastZero(input_table, s_type):
    data_help = input_table.to_numpy()  # numpy array only used for getting size of array
    TimeSinceLastX = np.empty([np.shape(data_help)[0], np.shape(data_help)[1]])
    df = pd.DataFrame(data=input_table)

    for row_no, row in df.iterrows(): # iterating in rows where row_no is the row number, row is the data in a whole row
        col_iter = 0  # column iterator, 0 at the beginning of each row
        TSLX = 0
        for cell in row:  # iterating in columns where cell is one cell of the row
            if s_type == "tcp": # counting bytes or packets
                if cell == 0: # if there are no bits/s TSLB counts up otherwise it's 0
                    TSLX += 1
                else:
                    TSLX = 0
            else:
                if cell < 20:  # if there are less than 40 packets/s TSLP counts up otherwise it's 0
                    TSLX += 1
                else:
                    TSLX = 0
            TimeSinceLastX[row_no, col_iter] = TSLX  # filling the result table used later
            col_iter += 1
    return TimeSinceLastX


# Importing data files
def Import_Files():
    kbit = [80, 90, 100, 0]
    loss = [10, 0]
    stream_type = ["udp", "tcp"]
    udp_data = {}
    tcp_data = {}
    quality = 1.5
    current_dir = os.path.dirname(os.path.realpath(__file__))
    for i_kbit in kbit:
        for i_loss in loss:
            #            if i_kbit == 70 and i_loss == 10:
            #                continue
            for s_type in stream_type:
                with open(f"{current_dir}/csv/{s_type}/{i_kbit}_{i_loss}_{s_type}.csv", 'r') as rf:
                    reader = csv.reader(rf, delimiter=';')
                    num_cols = len(reader.__next__())
                    num_keep_cols = min(num_cols, 130)
                    keep_cols = list(range(num_keep_cols))
                    rf.seek(0)

                    with open(f"{current_dir}/csv/{s_type}/{i_kbit}_{i_loss}_{s_type}_m.csv", 'w', newline='') as wf:
                        writer = csv.writer(wf, delimiter=';')
                        for row in reader:
                            writer.writerow([row[i] for i in keep_cols])

            udp_data[quality] = {}
            tcp_data[quality] = {}
            retran = pd.read_csv(f"{current_dir}/test/csv/tcp/{i_kbit}_{i_loss}_retran.csv", sep=';', header=None)
            retran = retran.fillna(0)
            tcp_len = pd.DataFrame(0, index=range(130), columns=range(1))
            retransmissions = pd.DataFrame(0, index=range(130), columns=range(1))
            for row_no, row in retran.iterrows():
                if int(row[0]) >= 130:
                    continue
                tcp_len[0][int(row[0])] += row[1]
                retransmissions[0][int(row[0])] += row[2]
            tcp_data[quality]['retransmission'] = retransmissions
            tcp_data[quality]['tcp_len'] = tcp_len
            udp_data[quality]['data_stream'] = pd.read_csv(f"{current_dir}/csv/udp/{i_kbit}_{i_loss}_udp_m.csv", sep=';', header=None)
            udp_data[quality]['kbit'] = i_kbit
            udp_data[quality]['loss'] = i_loss
            tcp_data[quality]['data_stream'] = pd.read_csv(f"{current_dir}/csv/tcp/{i_kbit}_{i_loss}_tcp_m.csv", sep=';', header=None)
            tcp_data[quality]['kbit'] = i_kbit
            tcp_data[quality]['loss'] = i_loss
            quality += 0.5
    return udp_data, tcp_data


def process():
    try:
        udp_data, tcp_data = Import_Files()
        TimeSinceLastPacket = {}
        TimeSinceLastByte = {}
        TimeSinceLastRetran = {}
        for key, input_table in udp_data.items():
            TimeSinceLastPacket[key] = TimeSinceLastZero(input_table['data_stream'], "udp")
        for key, input_table in tcp_data.items():
            TimeSinceLastByte[key] = TimeSinceLastZero(input_table['data_stream'], "tcp")
            TimeSinceLastRetran[key] = TimeSinceLastZero(input_table['retransmission'], "tcp")
        discord = pd.DataFrame()  # final DataFrames into which the preprocessed data goes

        for key in udp_data:
            for i in range(udp_data[key]['data_stream'].shape[0]):
                discord_help = pd.DataFrame()
                discord_help[0] = udp_data[key]['data_stream'].to_numpy()[i]
                discord_help[1] = TimeSinceLastPacket[key][i]
                discord_help[2] = tcp_data[key]['data_stream'].to_numpy()[i]
                discord_help[3] = TimeSinceLastByte[key][i]
                discord_help[4] = udp_data[key]['kbit']
                discord_help[5] = udp_data[key]['loss']
                discord_help[6] = tcp_data[key]['retransmission']
                discord_help[6] = TimeSinceLastRetran[key][i]
                discord_help[7] = tcp_data[key]['tcp_len']
                discord_help[8] = (key - 1) / 4
                discord = pd.concat([discord, discord_help], ignore_index=True)

        discord.columns = ['voice', 'TSLP', 'ctrl', 'TSLB', 'kbit', 'loss', 'retransmissions', 'tcp_len', 'quality']
        discord.to_csv(f'{os.path.dirname(os.path.realpath(__file__))}/neural_network_inputs.csv', sep=',')
        X = discord.iloc[:, 0:8]
        y = np.ravel(discord.quality)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        _logger.error(e)

