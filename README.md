Stats for vast.ai meant to be run of Google VM

git clone https://github.com/alkorolyov/vast-stats; cd vast-stats; sudo bash ./install.sh

[//]: # (curl -sSL https://raw.githubusercontent.com/alkorolyov/vast-stats/master/install -o install; sudo python3 install)

[//]: # (sudo -u vast curl -sSL https://raw.githubusercontent.com/alkorolyov/vast-stats/master/src/utils.py -o /var/lib/vast-stats/src/utils.py)

[//]: # (sudo -u vast curl -sSL https://raw.githubusercontent.com/alkorolyov/vast-stats/master/main.py -o /var/lib/vast-stats/main.py)

# download from gcloud

# gcloud compute scp {vm_instance_name}:/var/lib/vast-stats/vast.db vast.db

cd C:\Users\ergot\DataspellProjects\vast-stats\data
gcloud compute scp free-ubn20:/var/lib/vast-stats/vast.db vast.db

gcloud compute scp free-ubn20:/home/ergot/vast-stats/vast.db vast_gc.db