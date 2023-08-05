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
        if k in ["universal_newlines", "check"] and v is True:
            continue

        res.append(f"{k}={repr(v)}")
    return ", ".join(res)


def process_run(*a, **kw):
    kw.setdefault("stdout", PIPE)
    kw.setdefault("stderr", PIPE)
    kw.setdefault("universal_newlines", True)
    kw.setdefault('check', True)
    # optional = kw.pop("optional", False)
    kw["env"] = os.environ
    try:
        logging.info(f"subprocess.run({format_process_args(*a, **kw)})")
        res = subprocess.run(*a, **kw)
        logging.info(res.stdout)
        return res.returncode
    # except OSError as e:
    #     if e.errno == errno.ENOENT and optional:
    #         logging.warning(e)
    #         return None
    #     logging.error(e)
    #     raise
    except CalledProcessError as e:
        logging.error(e.stderr)
        raise

if __name__ == '__main__':
    logging.info('=> Begin vast-stats software install')

    # check if daemon user exists
    process_run(['id', '-u', DAEMON_USER])
    logging.info('=> Create vast-stats daemon user')

