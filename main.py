#!/usr/bin/python3
import sys
import argparse
import pandas as pd
import datetime as dt
import logging
from logging.handlers import RotatingFileHandler

from time import sleep, time
# from memory_profiler import profile

from src import const
from src.fetch import fetch_raw
from src.preprocess import preprocess
from src.utils import time_ms, next_timeout, read_last_n_lines, get_error_info, setqueue
from src.vastdb import VastDB
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
                                       maxBytes=const.MAX_LOGSIZE,
                                       backupCount=const.LOG_COUNT)
        log_handler = [rotating]

    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=const.LOG_FORMAT,
                        handlers=log_handler,
                        level=log_level,
                        datefmt='%d-%m-%Y %I:%M:%S')

    # create db
    logging.info('[MAIN] Script started')

    with VastDB(db_path) as vast:
        vast.create_tables()

        # TODO temporary add isp column
        info = vast.dbm.get_tbl_info('eod_snp')
        if 'isp' not in info.name.values:
            vast.dbm.execute("ALTER TABLE eod_snp ADD isp TEXT DEFAULT ''")
            vast.dbm.execute("ALTER TABLE eod_ts ADD isp TEXT DEFAULT ''")
            vast.dbm.commit()

    logging.info('[MAIN] Tables created')

    # main cycle
    while True:

        start = time()

        with VastDB(db_path) as vast:
            last_ts = vast.get_last_ts()
            logging.debug(f"[DB] last timestamp {dt.datetime.fromtimestamp(last_ts)}")

        # fetch
        try:
            raw = fetch_raw(last_ts)

            if raw is None:
                sleep(next_timeout(const.TIMEOUT))
                continue

            dt_source = dt.datetime.fromtimestamp(raw.timestamp.iloc[0])
            logging.debug(f"[API] source_ts: [{dt_source.time()}]")
            logging.debug(f"[API] ts - now : {(dt.datetime.now().replace(microsecond=0) - dt_source)}")
            # if dt_source < dt_last:
            #     logging.info(f"[API] last_ts-ts: {dt_last - dt_source}")

        except Exception as e:
            logging.error("[FETCH] Fetching error", get_error_info(e))
            send_error_email('VAST-STATS CRUSHED', get_error_info(e))
            raise

        logging.debug(f'[API] Request completed in {time() - start:.2f}s')

        # preprocessing
        start_preprocess = time()

        try:
            machines = preprocess(raw)

        except Exception as e:
            logging.error("[PROCESS] Preprocessing failed", get_error_info(e))
            send_error_email('VAST-STATS CRUSHED', get_error_info(e))
            raise


        logging.debug(f"[PREPROCESS] Completed in {time_ms(time() - start_preprocess)}ms")
        # exit(0)

        start_total_db = time()

        # update db
        with VastDB(db_path) as vast:
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

        logging.debug('=' * 50)

        # break
        sleep(next_timeout(const.TIMEOUT))


if __name__ == '__main__':
    main()
