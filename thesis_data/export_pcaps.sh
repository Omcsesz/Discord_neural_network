#!/bin/bash
set -veu

KBIT=(80 90 100 0)
LOSS=(0 10)

rm test/csv/udp/*
rm test/csv/tcp/*

for I_KBIT in ${KBIT[@]}; do
    for I_LOSS in ${LOSS[@]}; do
        for FILE_NUM in {1..10}; do
            UDP_STREAM=$(tshark -nr caps/${I_KBIT}_${I_LOSS}_${FILE_NUM}.pcapng -q -z io,stat,1,ip.addr==66.22.244.132 | grep '<>' | sed 's/^|* ... <> ... | *//' | sed 's/ | ..... . |$//')
            TCP_STREAM=$(tshark -nr caps/${I_KBIT}_${I_LOSS}_${FILE_NUM}.pcapng -q -z 'io,stat,1,ip.addr==192.168.2.31&&ip.addr==162.159.0.0/16' | grep '<>' | sed 's/^|* ... <> ... | * .. | *//' | sed 's/ | * |$//')
            tshark -nr caps/${I_KBIT}_${I_LOSS}_${FILE_NUM}.pcapng -Y 'ip.addr==192.168.2.31&&ip.addr==162.159.0.0/16&&tcp' -T fields -e frame.time_relative -e tcp.len -e tcp.analysis.retransmission >> test/csv/tcp/${I_KBIT}_${I_LOSS}_retran.csv
            tshark -nr caps/${I_KBIT}_${I_LOSS}_${FILE_NUM}.pcapng -d udp.port==50004,rtp -Y rtp -T fields -e frame.time_relative -e frame.time_delta_displayed -e rtp.seq >> test/csv/udp/${I_KBIT}_${I_LOSS}_jitter.csv
            echo $UDP_STREAM | tee -a test/csv/udp/${I_KBIT}_${I_LOSS}_udp.csv
            echo $TCP_STREAM | tee -a test/csv/tcp/${I_KBIT}_${I_LOSS}_tcp.csv
            #echo $RETRAN | tee -a test/csv/tcp/${I_KBIT}_${I_LOSS}_retran.csv
        done
        sed -i 's/\t/;/g' test/csv/tcp/${I_KBIT}_${I_LOSS}_retran.csv
        sed -i 's/\t/;/g' test/csv/udp/${I_KBIT}_${I_LOSS}_jitter.csv
        sed -i 's/ /;/g' test/csv/udp/${I_KBIT}_${I_LOSS}_udp.csv
        sed -i 's/ /;/g' test/csv/tcp/${I_KBIT}_${I_LOSS}_tcp.csv
    done
done

# tshark -nr caps/80_10_10.pcapng -Y 'ip.addr==192.168.2.31&&ip.addr==162.159.0.0/16' -T fields -e frame.time_relative -e tcp.analysis.retransmission
