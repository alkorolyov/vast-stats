from __future__ import annotations

import sqlite3
import logging
from time import time

import pandas as pd

from src.const import HARDWARE_COLS, COST_COLS, EOD_COLS, AVG_COLS
from src.manager import DbManager
from src.tables import _Table, SingleTbl, MapTable, OnlineTS, Timeseries, AverageStd
from src.utils import time_ms


class VastDB:
    """ Class responsible for the Vast database structure """
    dbm: DbManager
    tables: list[_Table]
    avg_updated: bool = False

    def __init__(self, db_path: str):
        self.dbm = DbManager(db_path)
        self.ts_idx = 'ts_idx'
        self.tables = []
        self.init_tables()

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
        ts_idx = SingleTbl(self.ts_idx, ['timestamp'])
        self.tables.append(ts_idx)

    def _init_map_tables(self):
        mach_host_map = MapTable('machine_host_map',
                                 ['machine_id', 'host_id'])
        self.tables.append(mach_host_map)

    def _init_timeseries(self):
        online = OnlineTS('online', 'machine_host_map')
        cpu_ram = Timeseries('cpu_ram', ['cpu_ram'])
        disk = Timeseries('disk', ['disk_space'])
        reliability = Timeseries('reliability', ['reliability'])
        rent = Timeseries('rent', ['num_gpus_rented'])

        # Aggregated Tables
        hardware = Timeseries('hardware', HARDWARE_COLS)
        eod = Timeseries('eod', EOD_COLS)
        cost = Timeseries('cost', COST_COLS)
        avg = AverageStd('avg', AVG_COLS, period='1 day')   # TODO change to 1 day for prod

        self.tables += [
            online,
            hardware,
            cpu_ram, disk,
            eod, reliability,
            cost, rent, avg
        ]

    def init_tables(self):
        self._init_ts_index()
        self._init_map_tables()
        self._init_timeseries()

    def create_tables(self):
        for tbl in self.tables:
            tbl.create(self.dbm)

    def update_tables(self) -> int:
        total_rows = 0
        self.avg_updated = False
        for tbl in self.tables:
            start = time()
            rows = tbl.update(self.dbm)
            if isinstance(tbl, AverageStd) and rows > 0:
                self.avg_updated = True
            total_rows += rows
            logging.debug(f'[{tbl.name.upper()}] {rows} rows updated in {time_ms(time() - start)}ms')
        return total_rows

    def get_last_ts(self) -> int:
        return self.dbm.get_last_ts(self.ts_idx)




