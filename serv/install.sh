#!/bin/bash

export VAST_INSTALL_DIR=/var/lib/vast-stats
export VAST_DAEMON_GROUP=vast-stats
export VAST_DAEMON_USER=vast


apt-get update -y
apt-get install curl git python3-pip -y


# git clone
cd /var/lib
git clone https://github.com/alkorolyov/vast-stats/

# create vast user
groupadd $VAST_DAEMON_GROUP
adduser --system --home $VAST_INSTALL_DIR --disabled-password --ingroup $VAST_DAEMON_GROUP --shell /bin/bash $VAST_DAEMON_GROUP
chown -R $VAST_DAEMON_GROUP:$VAST_DAEMON_GROUP $VAST_INSTALL_DIR

# pip
sudo -u vast python3 -m pip install -r $VAST_INSTALL_DIR/requirements.txt

# install service
cp $VAST_INSTALL_DIR/serv/vast.service /etc/systemd/system
chown $VAST_DAEMON_GROUP:$VAST_DAEMON_GROUP /etc/systemd/system/vast.service
chmod +x $VAST_INSTALL_DIR/serv/start.sh
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
