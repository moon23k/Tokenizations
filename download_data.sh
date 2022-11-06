#!/bin/bash
mkdir -p data/ende
cd data/ende

splits=(train val test_2016_flickr)
langs=(en de)

for split in "${splits[@]}"; do
    for lang in "${langs[@]}"; do
        wget https://github.com/multi30k/dataset/raw/master/data/task1/raw/${split}.${lang}.gz
        gunzip ${split}.${lang}.gz
    done
done

