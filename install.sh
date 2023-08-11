#!/bin/bash

DIR="/var/lib/vast-stats"
GROUP="vast-stats"
USER="vast"

# create folders and download sources
mkdir $DIR
cd $DIR
mkdir data
git clone https://github.com/alkorolyov/vast-stats/ .

# create user/group
addgroup $GROUP
adduser --system --ingroup $GROUP --disabled-password --no-create-home --home $DIR $USER
chown -R $USER:$GROUP $DIR

# pip
apt -qq update -y
apt -qq install python3 python3-pip -y
sudo -u vast python3 -m pip -q install -r $DIR/requirements.txt

# Create and run service
SERVICE_CONTENT="[Unit]\n
Description=VastAi Stats Service\n
After=network.target\n
\n
[Service]\n
Type=simple\n
User=vast\n
WorkingDirectory=$DIR\n
ExecStart=$DIR/main.py\n
Restart=on-failure\n
\n
[Install]\n
WantedBy=multi-user.target\n
"

SERVICE_FILE="/etc/systemd/system/vast.service"
echo -e "$SERVICE_CONTENT" > "$SERVICE_FILE"

chmod +x $DIR/main.py
systemctl daemon-reload
#systemctl enable vast
systemctl start vast