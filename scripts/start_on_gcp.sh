#!/bin/sh

cd /usr/local/vast || exit
mkdir src
mkdir data

# gitclone
curl -sSL https://raw.githubusercontent.com/alkorolyov/vast/main/start -o start
curl -sSL https://raw.githubusercontent.com/alkorolyov/vast/main/src/tables.py -o src/tables.py
curl -sSL https://raw.githubusercontent.com/alkorolyov/vast/main/src/utils.py -o src/utils.py
curl -sSL https://raw.githubusercontent.com/alkorolyov/vast/main/src/preprocess.py -o src/preprocess.py

python3 main.py