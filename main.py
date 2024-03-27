#!/usr/bin/python3
import math
import argparse
import pandas as pd
import datetime as dt
import logging
import traceback
from logging.handlers import RotatingFileHandler

from time import sleep, time
# from memory_profiler import profile

from src.fetch import fetch_sources
from src.utils import time_ms, next_timeout, read_last_n_lines
from src.vastdb import VastDB
from src.const import TIMEOUT, LOG_FORMAT, MAX_LOGSIZE, LOG_COUNT
from src.email import send_error_email

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


    # create db
    logging.info('[MAIN] Script started')
    vast = VastDB(db_path)
    vast.connect()
    vast.create_tables()
    vast.close()
    logging.info('[MAIN] Tables created')

    # main cycle
    while True:
        start = time()

        vast.connect()
        last_ts = vast.get_last_ts()
        vast.close()
        logging.debug(f"[DB] last timestamp {dt.datetime.fromtimestamp(last_ts)}")

        try:
            machines = fetch_sources(last_ts)

            if machines is None:
                sleep(next_timeout(TIMEOUT))
                continue

            dt_source = dt.datetime.fromtimestamp(machines.timestamp.iloc[0])
            logging.debug(f"[API] source_ts: [{dt_source.time()}]")
            logging.debug(f"[API] ts - now : {(dt.datetime.now().replace(microsecond=0) - dt_source)}")
            # if dt_source < dt_last:
            #     logging.info(f"[API] last_ts-ts: {dt_last - dt_source}")

        except Exception as e:
            # msg = f"[API] General error {e}"
            msg = '\n'.join(traceback.format_exception(type(e), e, e.__traceback__))
            logs = read_last_n_lines(log_path, 10)

            logging.error(msg)
            send_error_email('VAST-STATS error', f"Error message:\n{msg}\nLogs:\n{logs}")
            raise

        logging.debug(f'[API] Request completed in {time() - start:.2f}s')

        start_total_db = time()

        vast.connect()

        start = time()
        vast.create_tmp_tables(machines)
        logging.debug(f'[TMP_TABLES] created in {time_ms(time() - start)}ms')

        rows = vast.update_tables()
        vast.commit()
        logging.info(f'[TOTAL_DB] {rows} rows updated in {time_ms(time() - start_total_db)}ms')

        if vast.avg_updated:
            start = time()
            vast.vacuum()
            logging.info(f'[TOTAL_DB] Vacuum in {time_ms(time() - start)}ms')

        vast.close()
        logging.debug('=' * 50)

        # break
        sleep(next_timeout(TIMEOUT))
        # sleep(TIMEOUT)


if __name__ == '__main__':
    main()
