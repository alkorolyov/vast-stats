from __future__ import annotations

import sqlite3
import logging

import pandas as pd

from src.manager import DbManager
from src.new_tables import _Table, SingleCol, Unique


class VastDB:
    """ Class responsible for the Vast database structure """
    dbm: DbManager
    tables: list[_Table]

    def __init__(self, db_path: str):
        self.dbm = DbManager(db_path)
        self.ts_col = 'timestamp'
        self.ts_idx = 'timestamp_idx'
        self.tmp_mach = 'tmp_machines'
        self.tmp_offer = 'tmp_offers'
        self.tables = []

    def connect(self):
        self.dbm.connect()

    def close(self):
        self.dbm.disconnect()

    def create_tmp_tables(self, machines, offers):
        self.dbm.df_to_tmp_table(machines, self.tmp_mach)
        self.dbm.df_to_tmp_table(offers, self.tmp_offer)
        # debug
        # self.dbm.df_to_table(machines, self.tmp_mach)
        # self.dbm.df_to_table(offers, self.tmp_offer)

    def create_ts_index(self):
        ts_idx = SingleCol(self.ts_idx, [self.ts_col], self.tmp_mach)
        ts_idx.create(self.dbm)
        self.tables.append(ts_idx)

    def create_map_tables(self):
        host_mach_map = Unique('host_mach_map',
                               ['machine_id', 'host_id'],
                               self.tmp_mach)
        mach_offer_map = Unique('mach_offer_map',
                                ['id', 'machine_id'],
                                self.tmp_offer)
        chunks = Unique('chunks',
                        ['id', 'num_gpus'],
                        self.tmp_offer)
        self.tables.append(host_mach_map)
        self.tables.append(mach_offer_map)
        self.tables.append(chunks)
        for tbl in self.tables:
            tbl.create(self.dbm)

    def update_tables(self):
        rows = 0
        for tbl in self.tables:
            rows += tbl.update(self.dbm)
        return rows

    def get_last_ts(self) -> int:
        output = self.dbm.execute(f'''
            SELECT timestamp
            FROM {self.ts_idx}
            ORDER BY ROWID
            DESC LIMIT 1
        ''').fetchall()
        return output[0][0] if output else 0




