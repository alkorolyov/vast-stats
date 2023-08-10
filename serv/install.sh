#!/bin/bash

INSTALL_DIR="/var/lib/vast-stats"
DAEMON_GROUP="vast-stats"
DAEMON_USER="vast"


apt-get update -y
apt-get install curl git python3-pip -y


# git clone
cd /var/lib
git clone https://github.com/alkorolyov/vast-stats/

# create vast user
groupadd $DAEMON_GROUP
adduser --system --home $INSTALL_DIR --disabled-password --ingroup $DAEMON_GROUP --shell /bin/bash $DAEMON_USER
chown -R $DAEMON_USER:$DAEMON_GROUP $INSTALL_DIR

# pip
sudo -u vast python3 -m pip install -r $INSTALL_DIR/requirements.txt

# install service
cp $INSTALL_DIR/serv/vast.service /etc/systemd/system
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
