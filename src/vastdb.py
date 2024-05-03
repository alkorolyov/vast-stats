from __future__ import annotations

import signal
import sqlite3
import logging
from time import time
import pandas as pd

from src.const import HARDWARE_COLS, COST_COLS, EOD_COLS, AVG_COLS
from src.manager import DbManager
from src.tables import _Table, SingleTbl, MapTable, OnlineTS, Timeseries, AverageStd
from src.utils import time_ms, get_error_info
from src.email import send_error_email


class VastDB:
    """ Class responsible for the Vast database structure """
    dbm: DbManager
    tables: dict[str, _Table]
    avg_updated: bool = False

    def __init__(self, db_path: str):
        self.dbm = DbManager(db_path)
        # self.ts_idx = 'ts_idx'
        self.tables = {}
        self.init_tables()
        signal.signal(signal.SIGTERM, self.handle_sigterm)

    def __enter__(self):
        try:
            self.dbm.connect()
            return self
        except Exception as e:
            msg = f"[DB] Connection error: {get_error_info(e)}"
            logging.error(msg)
            send_error_email('VAST-STATS CRUSHED', msg)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.dbm.close()
        except Exception as e:
            msg = f"[DB] Close error: {get_error_info(e)}"
            logging.error(msg)
            send_error_email('VAST-STATS CRUSHED', msg)
            raise

        if exc_type:
            msg = f"[DB] Error during Database Operations: {exc_type}:{exc_val}\n{exc_tb}"
            logging.error(msg)
            send_error_email('VAST-STATS CRUSHED', msg)
            raise

    def handle_sigterm(self, signum, frame):
        logging.warning('[OS] Received SIGTERM signal')
        if self.dbm.conn:
            self.dbm.close()
        exit(0)

    def connect(self):
        self.dbm.connect()

    def close(self):
        self.dbm.close()

    def commit(self):
        self.dbm.commit()

    def vacuum(self):
        self.dbm.vacuum()

    def create_tmp_tables(self, machines):
        try:
            self.dbm.df_to_tmp_table(machines, 'tmp_machines')
        except sqlite3.Error as e:
            logging.error(f"Error creating tmp tables: {e}")
            raise

    def _init_ts_index(self):
        self.tables['ts_idx'] = SingleTbl('ts_idx', ['timestamp'])
        # self.tables.append(ts_idx)

    def _init_map_tables(self):
        mach_host_map = MapTable('machine_host_map',
                                 ['machine_id', 'host_id'])
        self.tables['machine_host_map'] = mach_host_map
        # self.tables.append(mach_host_map)

    def _init_timeseries(self):
        self.tables['online'] = OnlineTS('online', 'machine_host_map')
        self.tables['cpu_ram'] = Timeseries('cpu_ram', ['cpu_ram'])
        self.tables['disk'] = Timeseries('disk', ['disk_space'])
        self.tables['reliability'] = Timeseries('reliability', ['reliability'])
        self.tables['rent'] = Timeseries('rent', ['num_gpus_rented'])

        # Aggregated Tables
        self.tables['hardware'] = Timeseries('hardware', HARDWARE_COLS)
        self.tables['eod'] = Timeseries('eod', EOD_COLS)
        self.tables['cost'] = Timeseries('cost', COST_COLS)
        self.tables['avg'] = AverageStd('avg', AVG_COLS, period='1 day')

        # online = OnlineTS('online', 'machine_host_map')
        # cpu_ram = Timeseries('cpu_ram', ['cpu_ram'])
        # disk = Timeseries('disk', ['disk_space'])
        # reliability = Timeseries('reliability', ['reliability'])
        # rent = Timeseries('rent', ['num_gpus_rented'])
        #
        # # Aggregated Tables
        # hardware = Timeseries('hardware', HARDWARE_COLS)
        # eod = Timeseries('eod', EOD_COLS)
        # cost = Timeseries('cost', COST_COLS)
        # avg = AverageStd('avg', AVG_COLS, period='1 day')
        #
        # self.tables += [
        #     online,
        #     hardware,
        #     cpu_ram, disk,
        #     eod, reliability,
        #     cost, rent, avg
        # ]

    def init_tables(self):
        self._init_ts_index()
        self._init_map_tables()
        self._init_timeseries()

    def create_tables(self):
        for tbl in self.tables.values():
            tbl.create(self.dbm)

    def update_tables(self) -> int:
        total_rows = 0
        self.avg_updated = False
        for tbl in self.tables.values():
            start = time()
            rows = tbl.update(self.dbm)
            if isinstance(tbl, AverageStd) and rows > 0:
                self.avg_updated = True
            total_rows += rows
            logging.debug(f'[{tbl.name.upper()}] {rows} rows updated in {time_ms(time() - start)}ms')
        return total_rows

    def get_last_ts(self) -> int:
        return self.dbm.get_last_ts(self.tables['ts_idx'].name)

    def get_first_ts(self) -> int:
        return self.dbm.get_first_ts(self.tables['ts_idx'].name)


