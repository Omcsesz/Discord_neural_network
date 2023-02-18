#!/bin/bash
set -veu

KBIT=(80 90 100 0)
LOSS=(0 10)

for I_KBIT in ${KBIT[@]}; do
    for I_LOSS in ${LOSS[@]}; do
        for FILE_NUM in {1..10}; do
            UDP_STREAM=$(tshark -nr caps/${I_KBIT}_${I_LOSS}_${FILE_NUM}.pcapng -q -z io,stat,1,ip.addr==66.22.244.132 | grep '<>' | sed 's/^|* ... <> ... | *//' | sed 's/ | ..... . |$//')
            TCP_STREAM=$(tshark -nr caps/${I_KBIT}_${I_LOSS}_${FILE_NUM}.pcapng -q -z 'io,stat,1,ip.addr==192.168.2.31&&ip.addr==162.159.0.0/16' | grep '<>' | sed 's/^|* ... <> ... | * .. | *//' | sed 's/ | * |$//')
            echo $UDP_STREAM | tee -a csv/udp/${I_KBIT}_${I_LOSS}_udp.csv
            echo $TCP_STREAM | tee -a csv/tcp/${I_KBIT}_${I_LOSS}_tcp.csv
        done
        sed -i 's/ /;/g' csv/udp/${I_KBIT}_${I_LOSS}_udp.csv
        sed -i 's/ /;/g' csv/tcp/${I_KBIT}_${I_LOSS}_tcp.csv
    done
done
