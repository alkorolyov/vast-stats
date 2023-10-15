#!/usr/bin/python3
import argparse
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

from time import sleep, time
# from memory_profiler import profile

from src.fetch import fetch_sources
from src.utils import time_ms
from src.vastdb import VastDB
from src.const import TIMEOUT, LOG_FORMAT, MAX_LOGSIZE, LOG_COUNT

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)


def main():
    # args parsing
    parser = argparse.ArgumentParser(description='Vast Stats Service')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Print to console instead of logfile')
    parser.add_argument('--debug', '-d', action='store_true', default=False, help='Print debug information')
    parser.add_argument('--db_path', default='.', help='Database store path')
    parser.add_argument('--log_path', default='.', help='Log file store path')

    args = vars(parser.parse_args())
    db_path = f"{args.get('db_path')}/vast.db"
    log_path = f"{args.get('log_path')}/vast.log"
    verbose = args.get('verbose')
    debug = args.get('debug')

    # logging
    log_handler = None
    if not verbose:
        rotating = RotatingFileHandler(log_path,
                                       maxBytes=MAX_LOGSIZE,
                                       backupCount=LOG_COUNT)
        log_handler = [rotating]

    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=LOG_FORMAT,
                        handlers=log_handler,
                        level=log_level,
                        datefmt='%d-%m-%Y %I:%M:%S')

    logging.info('[MAIN] Script started')
    vast = VastDB(db_path)
    vast.connect()
    vast.init_tables()
    vast.create_tables()
    vast.close()
    logging.info('[MAIN] Tables created')

    while True:
        start = time()

        vast.connect()
        last_timestamp = vast.get_last_ts()
        vast.close()

        try:
            machines = fetch_sources(last_timestamp)
            if machines is None:
                sleep(TIMEOUT)
                continue

        except Exception as e:
            logging.error(f"[API] General error {e}")
            raise

        start_total_db = time()

        logging.debug(f'[API] Request completed in {time() - start:.2f}s')

        start = time()
        vast.connect()
        vast.create_tmp_tables(machines)
        logging.debug(f'[TMP_TABLES] created in {time_ms(time() - start)}ms')

        rows = vast.update_tables()

        vast.commit()
        vast.close()

        logging.info(f'[TOTAL_DB] {rows} rows updated in {time_ms(time() - start_total_db)}ms')
        logging.debug('=' * 50)

        # break
        sleep(TIMEOUT)


if __name__ == '__main__':
    main()
