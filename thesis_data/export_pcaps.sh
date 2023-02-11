#!/bin/bash
set -vexu

KBIT=(80 90 100 0)
LOSS=(0 10)

for I_KBIT in ${KBIT[@]}; do
    for I_LOSS in ${LOSS[@]}; do
        for FILE_NUM in {1..10}; do
            STREAM=$(tshark -nr caps/0_10_${FILE_NUM}.pcapng -q -z io,stat,1,ip.addr==66.22.244.132 | grep '<>' | sed 's/^|* ... <> ... | *//' | sed 's/ | ..... . |$//')
            echo $STREAM | tee 'csv/udp/0_10.xlsx'
        done
    done
done

#ASD=$(tshark -nr caps/0_10_6.pcapng -q -z io,stat,1,ip.addr==66.22.244.132 | grep '<>' | sed 's/^|* ... <> ... | *//' | sed 's/ | ..... . |$//')

#echo $ASD >> valami.xlsx

#sed 's/ /,/' valami.xlsx