#!/bin/bash

DIR="/var/lib/vast-stats"
GROUP="vast-stats"
USER="vast"

# git clone
mkdir $DIR
cd $DIR
git clone https://github.com/alkorolyov/vast-stats/ .

# create user/group
addgroup $GROUP
adduser --system --ingroup $GROUP --disabled-password --no-create-home --home $DIR $USER
chown -R $USER:$GROUP $DIR

# pip
sudo -u vast python3 -m pip install -r $DIR/requirements.txt

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
echo -e $SERVICE_CONTENT > $SERVICE_FILE

chmod +x $DIR/main.py
systemctl daemon-reload
#systemctl enable vast
systemctl start vast





































INSTALL_DIR="/var/lib/vast-stats"
DAEMON_GROUP="vast-stats"
DAEMON_USER="vast"

# Specify the content of the vast.service file
SERVICE_CONTENT="[Unit]\n
Description=VastAi Stats Service\n
After=network.target\n
\n
[Service]\n
Type=simple\n
User=vast\n
ExecStart=/usr/bin/python3 \$INSTALL_DIR/main.py\n
Restart=on-failure\n
\n
[Install]\n
WantedBy=multi-user.target\n
"

SERVICE_FILE="/etc/systemd/system/vast.service"
echo -e $SERVICE_CONTENT > $SERVICE_FILE


apt-get update -y
apt-get install curl git python3-pip -y

# git clone
cd /var/lib || exit
git clone https://github.com/alkorolyov/vast-stats/

# create vast user
groupadd $DAEMON_GROUP
adduser --system --home $INSTALL_DIR --disabled-password --ingroup $DAEMON_GROUP --shell /bin/bash $DAEMON_GROUP
chown -R $DAEMON_GROUP:$DAEMON_GROUP $INSTALL_DIR

# pip
sudo -u vast python3 -m pip install -r $INSTALL_DIR/requirements.txt

# install service
cp $INSTALL_DIR/serv/vast.service /etc/systemd/system
chown $DAEMON_GROUP:$DAEMON_GROUP /etc/systemd/system/vast.service
chmod +x $INSTALL_DIR/serv/start.sh
systemctl daemon-reload
systemctl enable vast
systemctl start vast



# install vast-stats
mkdir vast
cd vast || exit
curl -sSL https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -o vast
chmod +x vast

# gitclone vast-stats stats
curl -sSL https://raw.githubusercontent.com/alkorolyov/vast/main/scripts/start_on_gcp.sh -o start_on_gcp.sh
chmod +x start_on_gcp.sh

# register as a service
curl -sSL https://raw.githubusercontent.com/alkorolyov/vast/main/scripts/vast.service -o /etc/systemd/system/vast.service
systemctl daemon-reload
systemctl enable vast
systemctl start vast
