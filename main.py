#!/usr/bin/python3
import requests
import argparse
import sqlite3
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from time import sleep, time
# from memory_profiler import profile

from src.tables import get_offers, get_machines, get_machines_offers, df_to_tmp_table, COST_COLS, HARDWARE_COLS, \
    EOD_COLS, AVG_COLS, \
    Timeseries, MapTable, Table, OnlineTS, MachineSplit, AverageStd
from src.preprocess import preprocess, split_raw
from src.utils import time_ms, time_utc_now

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)

TIMEOUT = 70  # Timeout between requests
RETRY_TIMEOUT = 20  # Timeout between unsuccessful attempts
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'
MAX_LOGSIZE = 1024 * 1024  # 1Mb
LOG_COUNT = 3


def main():
    # args parsing
    parser = argparse.ArgumentParser(description='Vast Stats Service')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Print to console or logfile')
    parser.add_argument('--db_path', default='.', help='Database store path')
    parser.add_argument('--log_path', default='.', help='Log file store path')

    args = vars(parser.parse_args())
    db_file = f"{args.get('db_path')}/vast.db"
    log_file = f"{args.get('log_path')}/vast.log"
    verbose = args.get('verbose')

    # Single Value Tables
    host_machine = MapTable('host_machine_map', 'machines', ['machine_id', 'host_id'])
    online = OnlineTS('online', 'host_machine_map')
    # new_online = NewOnlineTS('new_online', 'host_machine_map')
    machine_split = MachineSplit('machine_split', ['machine_id', 'num_gpus'], 'offers')
    cpu_ram = Timeseries('cpu_ram', ['cpu_ram'])
    disk = Timeseries('disk', ['disk_space'])
    reliability = Timeseries('reliability', ['reliability'])
    rent = Timeseries('rent', ['machine_id', 'rented'], 'offers')

    # Aggregated Tables
    hardware = Timeseries('hardware', HARDWARE_COLS)
    eod = Timeseries('eod', EOD_COLS)
    cost = Timeseries('cost', COST_COLS)
    avg = AverageStd('avg', AVG_COLS, period='1 d')
    ts = Table('timestamp_tbl')

    tables = [
        host_machine, online,
        # new_online,
        machine_split,
        hardware,
        cpu_ram, disk,
        eod,
        reliability,
        cost,
        rent,
        avg,
        ts
    ]

    # for col in HARDWARE_COLS:
    #     tables.append(Timeseries(col, [col]))
    # for col in EOD_COLS:
    #     tables.append(Timeseries(col, [col]))
    # for col in COST_COLS:
    #     tables.append(Timeseries(col, [col]))
    # for col in AVG_COLS:
    #     tables.append(Timeseries(col, [col]))
    #
    # tables.append(Timeseries('inet_up', ['inet_up']))
    # tables.append(Timeseries('inet_down', ['inet_down']))

    # logging
    log_handler = None
    if not verbose:
        rotating = RotatingFileHandler(log_file,
                                       maxBytes=MAX_LOGSIZE,
                                       backupCount=LOG_COUNT)
        log_handler = [rotating]

    logging.basicConfig(format=LOG_FORMAT,
                        handlers=log_handler,
                        level=logging.INFO,
                        datefmt='%d-%m-%Y %I:%M:%S')

    logging.info('[MAIN] init tables')

    conn = sqlite3.connect(db_file)

    for table in tables:
        table.init_db(conn)

    conn.commit()
    conn.close()

    while True:
        start = time()
        try:
            logging.info('[API] request started')
            machines, offers = get_machines_offers()

            # offers = get_offers()
            # preprocess(offers)
            #
            # # check for duplicates
            # dup = offers.id.duplicated(keep=False)
            # if dup.any():
            #     logging.warning(f'duplicated id: \n{offers[dup]}')
            #
            # machines = get_machines()
            # preprocess(machines)
            #
            # # check for duplicates
            # dup = machines.machine_id.duplicated(keep=False)
            # if dup.any():
            #     logging.warning(f'duplicated machine_id: \n{machines[dup]}')

        except requests.exceptions.Timeout:
            logging.exception("[API] CONNECTION TIMEOUT")
            sleep(RETRY_TIMEOUT)
            continue
        except requests.exceptions.RequestException:
            logging.exception("[API] REQUEST ERROR")
            sleep(RETRY_TIMEOUT)
            continue
        except Exception:
            logging.exception("[API] GENERAL EXCEPTION")

        start_total_db = time()

        conn = sqlite3.connect(db_file)

        output = conn.execute('SELECT timestamp FROM timestamp_tbl ORDER BY ROWID DESC LIMIT 1').fetchall()
        last_timestamp = output[0][0] if output else 0

        # timestamp = time_utc_now()

        if offers.timestamp.iloc[0] == last_timestamp:
            logging.warning(f'[API] snapshot already saved {time() - start:.2f}s')
            conn.close()
            sleep(RETRY_TIMEOUT)
            continue

        logging.info(f'[API] request completed in {time() - start:.2f}s')

        start = time()

        df_to_tmp_table(offers, 'tmp_offers', conn)
        df_to_tmp_table(machines, 'tmp_machines', conn)
        logging.info(f'[TMP_TABLES] created in {time_ms(time() - start)}ms')

        for table in tables:
            start = time()
            rowcount = table.write_db(conn)
            # if '_bw' in table.name or table.name == 'dlperf' or table.name == 'score':
            logging.info(f'[{table.name.upper()}] {rowcount} rows updated in {time_ms(time() - start)}ms')

        conn.commit()
        conn.close()

        logging.info(f'[TOTAL_DB] database updated in {time_ms(time() - start_total_db)}ms')
        logging.info('=' * 80)

        # break
        sleep(TIMEOUT)


if __name__ == '__main__':
    main()
