import csv
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.DEBUG) # format='%(levelname) [%(filename):%(lineno)] %(message)'
_logger = logging.getLogger(__name__)
pd.set_option('display.precision', 9)


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
            jit = pd.read_csv(f"{current_dir}/test/csv/udp/{i_kbit}_{i_loss}_jitter.csv", sep=';', header=None)
            retran = retran.fillna(0)
            jit = jit.fillna(0)
            tcp_len = pd.DataFrame(0, index=range(10), columns=range(130))
            retransmissions = pd.DataFrame(0, index=range(10), columns=range(130))
            max_jitter = pd.DataFrame(0, index=range(10), columns=range(130))
            avg_jitter = pd.DataFrame(0, index=range(10), columns=range(130))
            missing_packets = pd.DataFrame(0, index=range(10), columns=range(130))
            avg_missing_packets = pd.DataFrame(0, index=range(10), columns=range(130))
            row_no = 0
            prev_row_no = 0
            prev_time = 0
            avg_counter = 0
            avg_sum = 0
            max_delta = 0
            prev_seq_num = 0
            moving_avg_jitter = 0
            jitter_sum = 0
            sum_missing_packets = 0
            for feck_this, row in jit.iterrows():
                if int(row[0]) < prev_time:
                    row_no += 1
                    moving_avg_jitter = 0
                    sum_missing_packets = 0
                if int(row[0]) >= 130:
                    continue
                if row[2] != 0:
                    if int(row[2]) != prev_seq_num+1 and prev_seq_num != 0 and int(row[2]) > prev_seq_num and row_no == prev_row_no and int(row[2])-prev_seq_num < 100:
                        missing_packets[int(row[0])][row_no] += int(row[2])-prev_seq_num
                    prev_seq_num = int(row[2])
                if int(row[0]) == prev_time or int(row[0]) < prev_time:
                    avg_sum += row[1]
                    avg_counter += 1
                    moving_avg = avg_sum / avg_counter
                    jitter_sum += abs(row[1]-moving_avg)
                    if abs(row[1]) > abs(max_delta):
                        max_delta = abs(row[1])
                else:
                    if avg_counter != 0:
                        avg_delta = avg_sum / avg_counter
                        jitter = jitter_sum - avg_delta
                        moving_avg_jitter = abs((jitter + moving_avg_jitter) / avg_counter)
                    else:
                        avg_delta = row[1]
                    sum_missing_packets += missing_packets[int(row[0]-1)][row_no]
                    avg_missing_packets[int(row[0])-1][row_no] = sum_missing_packets / int(row[0])
                    max_jitter[int(row[0])-1][row_no] = max_delta-avg_delta
                    avg_jitter[int(row[0])-1][row_no] = moving_avg_jitter
                    avg_sum = 0
                    avg_counter = 0
                    jitter_sum = 0
                prev_row_no = row_no
                prev_time = int(row[0])

            row_no = 0
            prev_time = 0
            for _, row in retran.iterrows():
                if int(row[0]) < prev_time:
                    row_no += 1
                prev_time = int(row[0])
                if int(row[0]) >= 130:
                    continue
                tcp_len[int(row[0])][row_no] += row[1]
                retransmissions[int(row[0])][row_no] += row[2]
            tcp_data[quality]['retransmission'] = retransmissions
            tcp_data[quality]['tcp_len'] = tcp_len
            udp_data[quality]['max_jitter'] = max_jitter
            udp_data[quality]['avg_jitter'] = avg_jitter
            udp_data[quality]['missing_packets'] = missing_packets
            udp_data[quality]['avg_missing_packets'] = avg_missing_packets
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
                #discord_help[4] = udp_data[key]['kbit']
                #discord_help[5] = udp_data[key]['loss']
                discord_help[4] = tcp_data[key]['retransmission'].to_numpy()[i]
                discord_help[5] = TimeSinceLastRetran[key][i]
                #discord_help[6] = tcp_data[key]['tcp_len'].to_numpy()[i]
                discord_help[6] = udp_data[key]['max_jitter'].to_numpy()[i]
                discord_help[7] = udp_data[key]['avg_jitter'].to_numpy()[i]
                #discord_help[8] = udp_data[key]['missing_packets'].to_numpy()[i]
                discord_help[8] = udp_data[key]['avg_missing_packets'].to_numpy()[i]
                discord_help[9] = (key - 1) / 4
                discord = pd.concat([discord, discord_help], ignore_index=True)

        discord.columns = ['voice', 'TSLP', 'ctrl', 'TSLB', 'retransmissions', 'TSLR', 'max_jitter', 'avg_jitter', 'avg_missing_packets', 'quality']
        discord.to_csv(f'{os.path.dirname(os.path.realpath(__file__))}/neural_network_inputs.csv', sep=',')
        X = discord.iloc[:, 0:10]
        y = np.ravel(discord.quality)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        _logger.error(e)

