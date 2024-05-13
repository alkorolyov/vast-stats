#!/bin/bash

# Define ANSI escape codes for colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

INSTALL_DIR="/opt/vast-stats"
DATA_DIR="/var/lib/vast-stats"

USER="vast-stats"
GROUP="vast-stats"

SERVICE_NAME='vast-stats'

if [[ $UID -ne 0 ]]; then
    echo "Installation should be run as root."
    exit
fi

echo "=> Git clone sources to /tmp"
cd /tmp
git clone https://github.com/alkorolyov/vast-stats/
cd vast-stats/

echo "=> Stop service"
systemctl stop $SERVICE_NAME

echo "=> Copy sources to $INSTALL_DIR"
\cp -rf main.py $INSTALL_DIR
\cp -rf src $INSTALL_DIR
chown -R $USER:$GROUP $INSTALL_DIR
chown -R $USER:$GROUP $DATA_DIR

echo "=> Restart service"
systemctl start $SERVICE_NAME

sleep 5

# check service status
status=$(systemctl is-active $SERVICE_NAME)
if [[ "$status" == "active" ]]; then
    status="${GREEN}$status${NC}"
else
    status="${RED}$status${NC}"
fi
echo -e "=> Service status: $status"

echo "=> Remove /tmp files"
rm -rf /tmp/vast-stats

echo "=> Install complete"
