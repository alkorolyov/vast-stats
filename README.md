Stats for vast.ai meant to be run of Google VM

git clone https://github.com/alkorolyov/vast-stats; cd vast-stats; sudo bash ./install.sh

curl -sSL https://raw.githubusercontent.com/alkorolyov/vast-stats/master/install -o install; sudo python3 install

sudo -u vast curl -sSL https://raw.githubusercontent.com/alkorolyov/vast-stats/master/src/utils.py -o /var/lib/vast-stats/src/utils.py
sudo -u vast curl -sSL https://raw.githubusercontent.com/alkorolyov/vast-stats/master/main.py -o /var/lib/vast-stats/main.py