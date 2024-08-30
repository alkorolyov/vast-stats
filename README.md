Fetches information about vast.ai every 1min and stores it in local SQLite database.

```
git clone https://github.com/alkorolyov/vast-stats; cd vast-stats; sudo bash ./install.sh
```


[//]: # (curl -sSL https://raw.githubusercontent.com/alkorolyov/vast-stats/master/install -o install; sudo python3 install)

[//]: # (sudo -u vast curl -sSL https://raw.githubusercontent.com/alkorolyov/vast-stats/master/src/utils.py -o /var/lib/vast-stats/src/utils.py)

[//]: # (sudo -u vast curl -sSL https://raw.githubusercontent.com/alkorolyov/vast-stats/master/main.py -o /var/lib/vast-stats/main.py)

### install gcloud cli
```
https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe
```

### download from gcloud

[//]: # (gcloud compute scp {vm_instance_name}:/var/lib/vast-stats/vast.db vast.db)

```
tail -f /var/lib/vast-stats/vast.log
```

```
pv /var/lib/vast-stats/vast.db > vast.db
```

```
gcloud compute scp --port 22222 free-ubn20:/home/ergot/vast.db vast.db
```


### free google cloud vm
* Machine type: e2-micro
* Zone: us-central1-a
* Networking: Standard Tier
* Disk: 30GB Standard persistent disk
