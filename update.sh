#!/bin/bash

INSTALL_DIR="/opt/vast-stats"
DATA_DIR="/var/lib/vast-stats"

USER="vast-stats"
GROUP="vast-stats"

if [[ $UID -ne 0 ]]; then
    echo "Installation should be run as root."
    exit
fi

echo "=> Git clone sources to /tmp"
cd /tmp
git clone https://github.com/alkorolyov/vast-stats/
cd vast-stats/

echo "=> Copy sources to $INSTALL_DIR"
cp main.py $INSTALL_DIR
cp -r src $INSTALL_DIR
chown -R $USER:$GROUP $INSTALL_DIR
chown -R $USER:$GROUP $DATA_DIR