#!/bin/bash

# Define ANSI escape codes for colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

INSTALL_DIR="/opt/vast-stats"
DB_DIR="/var/lib/vast-stats"
USER="vast-stats"
GROUP=$USER

echo -e "=> ${GREEN}Start installation of Vast Stats service${NC}"

if [[ $UID -ne 0 ]]; then
    echo "Installation should be run as root."
    exit
fi

echo "=> Git clone sources to /tmp"
cd /tmp
git clone https://github.com/alkorolyov/vast-stats/
cd vast-stats/

echo "=> Create project dirs: $INSTALL_DIR $DB_DIR"
mkdir $INSTALL_DIR
mkdir $DB_DIR

echo "=> Create $USER user/group"
useradd -rs /bin/false $USER -d $INSTALL_DIR

echo "=> Copy sources to $INSTALL_DIR"
cp -f requirements.txt $INSTALL_DIR

cp -f main.py $INSTALL_DIR
chmod -x $INSTALL_DIR/main.py

mkdir $INSTALL_DIR/src
cp -rf src $INSTALL_DIR/src
chown -R $USER:$GROUP $INSTALL_DIR
chown -R $USER:$GROUP $DB_DIR

echo "=> Apt update"
apt -qq update -y

echo "=> Install python3 and pip"
apt -qq install python3 python3-pip -y

echo "=> Install pip requirements"
sudo -u $USER python3 -m pip -q install -r requirements.txt

echo "=> Create service config"
SERVICE_CONTENT="
[Unit]
Description=VastAi Stats Service
After=network.target

[Service]
Type=simple
User=$USER
Group=$GROUP
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/main.py --db.path $DATA_DIR
Restart=on-failure

[Install]
WantedBy=multi-user.target
"
SERVICE_NAME='vast-stats'

echo -e "$SERVICE_CONTENT" > /etc/systemd/system/$SERVICE_NAME.service

echo "=> Start service"
systemctl daemon-reload
systemctl start $SERVICE_NAME

# check service status
status=$(systemctl is-active $SERVICE_NAME)
if [[ "$status" == "active" ]]; then
    status="${GREEN}$status${NC}"
else
    status="${RED}$status${NC}"
fi
echo -e "=> Service status: $status"

echo "=> Install complete"