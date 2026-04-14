#!/bin/bash

if [ ! -f main.exe ]; then
    make -j
fi

blocksize_exp=1
num_div_exp=20
num_div_sub=0

echo "{" > results.json

for i in {1..10}; do
    echo "\"Test $i\": " >> results.json
    ./main.exe $blocksize_exp $num_div_exp $num_div_sub >> results.json
    if [ $i -lt 10 ]; then
        echo "," >> results.json
    fi
done

echo "}" >> results.json

num_div_sub=17

echo "{" > results_uneven.json

for i in {1..10}; do
    echo "\"Test $i\": " >> results_uneven.json
    ./main.exe $blocksize_exp $num_div_exp $num_div_sub >> results_uneven.json
    if [ $i -lt 10 ]; then
        echo "," >> results_uneven.json
    fi
done

echo "}" >> results_uneven.json