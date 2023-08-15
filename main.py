#!/usr/bin/python3
import argparse
import sqlite3
import pandas as pd
import logging
from time import sleep, time


from src.tables import get_offers, get_machines, df_to_tmp_table, COST_COLS, HARDWARE_COLS, EOD_COLS, AVG_COLS, Timeseries, MapTable, OnlineTS, MachineTS
from src.preprocess import preprocess
from src.utils import time_ms, time_utc_now

TIMEOUT = 20

# Tables
host_machine    = MapTable('host_machine_map', 'machines', ['machine_id', 'host_id'])
online          = OnlineTS('online', 'host_machine_map')
machine_split   = MachineTS('machine_split', 'offers', ['machine_id', 'num_gpus'])
hardware        = Timeseries('hardware', 'machines', HARDWARE_COLS)
cpu_ram         = Timeseries('cpu_ram', 'machines', ['cpu_ram'])
disk            = Timeseries('disk', 'machines', ['disk_space'])
eod             = Timeseries('eod', 'machines', EOD_COLS)
avg             = Timeseries('avg', 'machines', AVG_COLS)
reliability     = Timeseries('reliability', 'machines', ['reliability'])
cost            = Timeseries('cost', 'machines', COST_COLS)
rent            = Timeseries('rent', 'offers', ['machine_id', 'rented'])

tables = [
    host_machine, online, machine_split,
    hardware, cpu_ram, disk,
    eod, reliability,
    cost, rent,
    # avg,
]

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)

parser = argparse.ArgumentParser(description='Vast Stats Service')
parser.add_argument('--db_path', default='.', help='Database store path')

FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%m-%d-%Y %I:%M:%S')

if __name__ == '__main__':

    args = vars(parser.parse_args())
    db_file = f"{args.get('db_path')}/vast.db"
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
                logging.error('duplicated id:')
                print('\t', offers[dup])

            machines = get_machines()
            preprocess(machines)

            # check for duplicates
            dup = machines.machine_id.duplicated(keep=False)
            if dup.any():
                logging.error('duplicated machine_id:')
                print('\t', machines[dup])

        except Exception as e:
            logging.exception("[API] connection")
            sleep(TIMEOUT)
            continue

        start_total_db = time()

        conn = sqlite3.connect(db_file)

        output = conn.execute('SELECT timestamp FROM reliability_ts ORDER BY ROWID DESC LIMIT 1').fetchall()
        last_timestamp = output[0][0] if output else 0

        timestamp = time_utc_now()

        if offers.timestamp.iloc[0] == last_timestamp:
            logging.warning(f'[API] snapshot already saved {time() - start:.2f}s')
            conn.close()
            sleep(TIMEOUT)
            continue

        logging.info(f'[API] request completed in {time() - start:.2f}s')

        start = time()

        df_to_tmp_table(offers, 'tmp_offers', conn)
        df_to_tmp_table(machines, 'tmp_machines', conn)
        logging.info(f'[TMP_TABLES] created in {time_ms(time() - start)}ms')

        for table in tables:
            start = time()
            rowcount = table.write_db(conn)
            # if rowcount > 0:
            logging.info(f'[{table.name.upper()}] {rowcount} rows updated in {time_ms(time() - start)}ms')

        conn.commit()
        conn.close()

        logging.info(f'[TOTAL_DB] database updated in {time_ms(time() - start_total_db)}ms')
        print('=' * 80)

        sleep(70)
