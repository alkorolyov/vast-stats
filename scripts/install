#!/usr/bin/python

import os
import errno
import subprocess
from subprocess import PIPE, CalledProcessError
import sys
import inspect
import datetime as dt
import logging


INSTALL_DIR = "/var/lib/vast-stats"
DAEMON_USER = 'vast'

logging.basicConfig(filename="vast_stats_install.log",
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    level=logging.INFO)

maybe_sudo = ["sudo"] if os.geteuid() != 0 else []


def set_locale():
    for x in ["LANG", "LC_ADDRESS", "LC_COLLATE", "LC_CTYPE",
              "LC_IDENTIFICATION", "LC_MONETARY", "LC_MEASUREMENT",
              "LC_NAME", "LC_NUMERIC", "LC_PAPER", "LC_TELEPHONE",
              "LC_TIME", "LC_ALL", "LANGUAGE"]:
        os.environ[x] = "C.UTF-8"
    os.environ["LC_MESSAGES"] = "C"


def format_process_args(*a, **kw):
    res = []
    for x in a:
        res.append(repr(x))
    for k, v in kw.items():
        # skip default args
        if k in ["stdout", "stderr"] and v is PIPE:
            continue
        if k in ["universal_newlines", "check", "capture_output"] and v is True:
            continue

        res.append(f"{k}={repr(v)}")
    return ", ".join(res)


def process_run(*a, **kw):
    kw.setdefault("capture_output", True)
    # kw.setdefault("stdout", PIPE)
    # kw.setdefault("stderr", PIPE)
    kw.setdefault("universal_newlines", True)
    kw.setdefault('check', True)
    # optional = kw.pop("optional", False)
    kw["env"] = os.environ
    try:
        logging.info(f"subprocess.run({format_process_args(*a, **kw)})")
        res = subprocess.run(*a, **kw)
        logging.info(res.stdout)
        return res.returncode
    except CalledProcessError as e:
        logging.warning(e.stderr)
        return e.returncode
    except OSError as e:
        logging.error(e)
        raise


def mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        logging.error(e)
        raise


def create_daemon_user():
    logging.info('=> Create vast-stats daemon user')
    process_run(["groupadd", "vast"])
    process_run(["adduser",
                    "--system",
                    "--gecos", "",
                    "--home", INSTALL_DIR,
                    "--no-create-home",
                    "--disabled-password",
                    "--ingroup", "docker",
                    "--shell", "/bin/bash",
                    DAEMON_USER])
    process_run(["chown", f"vast:{DAEMON_USER}", INSTALL_DIR, "-R"])


def download_vast_stats():
    pass


def install_daemon_service():
    pass


if __name__ == '__main__':
    logging.info('=> Begin vast-stats software install')

    # apt update and install basic packages
    process_run(['sudo', 'apt-get', 'update', '-y'])
    process_run(['sudo', 'apt-get', 'install', 'mc', '-y'])

    mkdir(INSTALL_DIR)
    os.chdir(INSTALL_DIR)

    # check if daemon user exists
    if process_run(['id', '-u', DAEMON_USER]) != 0:
        create_daemon_user()

    logging.info('=> Downloading vast-stats')
    download_vast_stats()


    install_daemon_service()




