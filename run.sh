#!/bin/bash

for i in `seq 1 23`;
do
    echo "------------------------------------"
    python main.py --model=RN      --epochs=20 --problem "$i"
    python main.py --model=CNN_MLP --epochs=20 --problem "$i"
done
