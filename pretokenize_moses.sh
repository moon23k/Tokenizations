#!/bin/bash

datasets=(translate dialogue)
splits=(train valid test)
extensions=(src trg)

#Pre tokenize with moses
echo "Pretokenize with moses"
python3 -m pip install -U sacremoses
for data in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        for ext in "${extensions[@]}"; do
            sacremoses -l en -j 8 tokenize < ${data}/seq/${split}.${ext} > ${data}/tok/${split}.${ext}
        done
    done
done