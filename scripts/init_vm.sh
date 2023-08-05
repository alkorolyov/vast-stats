#!/bin/sh

sudo apt-get update -y
sudo apt-get install mc -y

# minimal pip
mkdir /usr/local/vast
cd /usr/local/vast || exit
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
export PATH=$PATH:/home/ergot/.local/bin

# requirements
pip install pandas

# install vast-stats
mkdir vast
cd vast || exit
curl -sSL https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -o vast
chmod +x vast

# gitclone vast-stats stats
curl -sSL https://raw.githubusercontent.com/alkorolyov/vast/main/scripts/start_on_gcp.sh -o start_on_gcp.sh
chmod +x start_on_gcp.sh

# register as a service
curl -sSL https://raw.githubusercontent.com/alkorolyov/vast/main/scripts/vast.service -o vast.service
cp vast.service /etc/systemd/system/vast.service
systemctl daemon-reload
systemctl enable vast
systemctl start vast
