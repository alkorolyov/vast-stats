#!/usr/bin/python3

import argparse
import sqlite3
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from time import sleep, time
# from memory_profiler import profile

from src.tables import get_offers, get_machines, df_to_tmp_table, COST_COLS, HARDWARE_COLS, EOD_COLS, AVG_COLS, \
    Timeseries, MapTable, OnlineTS, MachineTS, AverageStd, NewOnlineTS
from src.preprocess import preprocess
from src.utils import time_ms, time_utc_now

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)

TIMEOUT = 70  # Timeout between requests
RETRY_TIMEOUT = 20  # Timeout between unsuccessful attempts
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'
MAX_LOGSIZE = 1024 * 1024  # 1Mb
LOG_COUNT = 3


def main():
    # Single Value Tables
    host_machine = MapTable('host_machine_map', 'machines', ['machine_id', 'host_id'])
    online = OnlineTS('online', 'host_machine_map')
    # new_online = NewOnlineTS('new_online', 'host_machine_map')
    machine_split = MachineTS('machine_split', 'offers', ['machine_id', 'num_gpus'])
    cpu_ram = Timeseries('cpu_ram', 'machines', ['cpu_ram'])
    disk = Timeseries('disk', 'machines', ['disk_space'])
    reliability = Timeseries('reliability', 'machines', ['reliability'])
    rent = Timeseries('rent', 'offers', ['machine_id', 'rented'])

    # Aggregated Tables
    # hardware = Timeseries('hardware', 'machines', HARDWARE_COLS)
    # eod = Timeseries('eod', 'machines', EOD_COLS)
    # inet = Timeseries('inet', 'machines', ['inet_down', 'inet_up'])
    # cost = Timeseries('cost', 'machines', COST_COLS)
    avg = AverageStd('avg', 'machines', AVG_COLS, period='1 h')

    tables = [
        host_machine, online,
        # new_online,
        machine_split,
        # hardware,
        cpu_ram, disk,
        # eod,
        reliability,
        # cost,
        # inet,
        rent,
        avg,
    ]

    for col in HARDWARE_COLS:
        tables.append(Timeseries(col, 'machines', [col]))
    for col in EOD_COLS:
        tables.append(Timeseries(col, 'machines', [col]))
    for col in COST_COLS:
        tables.append(Timeseries(col, 'machines', [col]))
    tables.append(Timeseries('inet_up', 'machines', ['inet_up']))
    tables.append(Timeseries('inet_down', 'machines', ['inet_down']))


    # args parsing
    parser = argparse.ArgumentParser(description='Vast Stats Service')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Print output to console')
    parser.add_argument('--no_log', action='store_true', default=False, help='Dont store log files')
    parser.add_argument('--db_path', default='.', help='Database store path')
    parser.add_argument('--log_path', default='.', help='Log file store path')


    args = vars(parser.parse_args())
    db_file = f"{args.get('db_path')}/vast.db"
    log_file = f"{args.get('log_path')}/vast.log"
    verbose = args.get('verbose')
    no_log = args.get('no_log')

    # logging
    rotating = RotatingFileHandler(log_file,
                                   maxBytes=MAX_LOGSIZE,
                                   backupCount=LOG_COUNT)
    logging.basicConfig(format=LOG_FORMAT,
                        handlers=[rotating],
                        level=logging.INFO,
                        datefmt='%d-%m-%Y %I:%M:%S')
    if no_log:
        logging.disable(logging.CRITICAL)

    conn = sqlite3.connect(db_file)

    for table in tables:
        table.init_db(conn)

    conn.commit()
    conn.close()

    while True:
        start = time()

        try:
            offers = get_offers()
            preprocess(offers)

            # check for duplicates
            dup = offers.id.duplicated(keep=False)
            if dup.any():
                logging.warning(f'duplicated id: \n{offers[dup]}')

            machines = get_machines()
            preprocess(machines)

            # check for duplicates
            dup = machines.machine_id.duplicated(keep=False)
            if dup.any():
                logging.warning(f'duplicated machine_id: \n{machines[dup]}')

        except Exception as e:
            logging.exception("[API] connection")
            sleep(RETRY_TIMEOUT)
            continue

        start_total_db = time()

        conn = sqlite3.connect(db_file)

        output = conn.execute('SELECT timestamp FROM reliability_ts ORDER BY ROWID DESC LIMIT 1').fetchall()
        last_timestamp = output[0][0] if output else 0

        # timestamp = time_utc_now()

        if offers.timestamp.iloc[0] == last_timestamp:
            msg = f'[API] snapshot already saved {time() - start:.2f}s'
            logging.warning(msg)
            if verbose:
                print(msg)
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
            msg = f'[{table.name.upper()}] {rowcount} rows updated in {time_ms(time() - start)}ms'
            logging.info(msg)
            if verbose:
                print(msg)

        conn.commit()
        conn.close()

        msg = f'[TOTAL_DB] database updated in {time_ms(time() - start_total_db)}ms'
        logging.info(msg)
        logging.info('=' * 80)
        if verbose:
            print(msg)
            print('=' * 80)

        # break
        sleep(TIMEOUT)


if __name__ == '__main__':
    main()

