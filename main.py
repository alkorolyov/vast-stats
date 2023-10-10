#!/usr/bin/python3
import requests
import argparse
import sqlite3
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from time import sleep, time
# from memory_profiler import profile

from src import const
from src.const import TIMEOUT, RETRY_TIMEOUT, LOG_FORMAT, MAX_LOGSIZE, LOG_COUNT
from src.tables import get_machines, df_to_tmp_table, \
     \
    Timeseries, MapTable, Table, OnlineTS, AverageStd
from src.utils import time_ms

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)


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
    # machine_split = MachineSplit('machine_split', ['machine_id', 'num_gpus'], 'offers')
    cpu_ram = Timeseries('cpu_ram', ['cpu_ram'])
    disk = Timeseries('disk', ['disk_space'])
    reliability = Timeseries('reliability', ['reliability'])
    rent = Timeseries('rent', ['num_gpus_rented'])

    # Aggregated Tables
    hardware = Timeseries('hardware', const.HARDWARE_COLS)
    eod = Timeseries('eod', const.EOD_COLS)
    cost = Timeseries('cost', const.COST_COLS)
    avg = AverageStd('avg', const.AVG_COLS, period='5 min')
    ts = Table('timestamp_tbl')

    tables = [
        host_machine, online,
        # new_online,
        # machine_split,
        hardware,
        cpu_ram, disk,
        eod,
        reliability,
        cost,
        rent,
        avg,
        ts
    ]

    # logging
    log_handler = None
    if not verbose:
        rotating = RotatingFileHandler(log_file,
                                       maxBytes=MAX_LOGSIZE,
                                       backupCount=LOG_COUNT)
        log_handler = [rotating]

    logging.basicConfig(format=LOG_FORMAT,
                        handlers=log_handler,
                        # level=logging.DEBUG,
                        level=logging.INFO,
                        datefmt='%d-%m-%Y %I:%M:%S')

    logging.info('[MAIN] init tables')

    conn = sqlite3.connect(db_file)

    for table in tables:
        logging.debug(table.name)
        table.init_db(conn)

    conn.commit()
    conn.close()

    while True:
        start = time()
        try:
            logging.info('[API] request started')
            machines = get_machines()
            # machines, offers = get_machines_offers()


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

        if machines.timestamp.iloc[0] == last_timestamp:
            logging.warning(f'[API] snapshot already saved {time() - start:.2f}s')
            conn.close()
            sleep(RETRY_TIMEOUT)
            continue

        logging.info(f'[API] request completed in {time() - start:.2f}s')

        start = time()

        # df_to_tmp_table(offers, 'tmp_offers', conn)
        df_to_tmp_table(machines, 'tmp_machines', conn)
        logging.info(f'[TMP_TABLES] created in {time_ms(time() - start)}ms')

        total_rows = 0
        for table in tables:
            start = time()
            rowcount = table.write_db(conn)
            total_rows += rowcount
            if rowcount:
                logging.debug(f'[{table.name.upper()}] {rowcount} rows updated in {time_ms(time() - start)}ms')

        conn.commit()
        conn.close()

        logging.info(f'[TOTAL_DB] {total_rows} rows updated in {time_ms(time() - start_total_db)}ms')
        logging.info('=' * 50)

        # break
        sleep(TIMEOUT)


if __name__ == '__main__':
    main()
