#!/bin/bash
mkdir -p mecab
cd mecab

#pre-requisites
sudo apt install curl git
python3 -m pip install konlpy JPype1


#install Mecab
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
make install
sudo ldconfig
cd ..


#install mecab-ko-dic
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./configure
make
make install
cd ..

bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
python3 -m pip install mecab-python

ls | grep -E "*.gz" | xargs rm