from __future__ import annotations

import sqlite3
import logging

from src.manager import DbManager
from src.new_tables import _Table, Table, MapTable


class VastDB:
    """ Class responsible for the Vast database structure """
    tables: list[_Table]

    def __init__(self, db_path: str):
        self.manager = DbManager(db_path)
        self.ts_idx = 'timestamp_idx'
        self.tmp_mach = 'tmp_machines'
        self.tmp_offer = 'tmp_offers'
        self.tables = []

    def connect(self):
        self.manager.connect()

    def close(self):
        self.manager.disconnect()

    def create_tmp_tables(self, machines, offers):
        # self.manager.df_to_tmp_table(machines, self.tmp_mach)
        # self.manager.df_to_tmp_table(offers, self.tmp_offr)
        # debug
        self.manager.df_to_table(machines, self.tmp_mach)
        self.manager.df_to_table(offers, self.tmp_offer)

    def create_ts_index(self):
        ts_idx = Table(self.ts_idx, ['timestamp'], self.tmp_mach)
        ts_idx.create(self.manager)
        self.tables.append(ts_idx)

    def create_map_tables(self):
        host_mach_map = MapTable('host_mach_map',
                                 ['host_id', 'machine_id'],
                                 self.tmp_mach)
        mach_offer_map = MapTable('mach_offer_map',
                                  ['machine_id', 'id'],
                                  self.tmp_offer)
        host_mach_map.create(self.manager)
        mach_offer_map.create(self.manager)
        self.tables.append(host_mach_map)
        self.tables.append(mach_offer_map)

    def update_tables(self):
        rows = 0
        for tbl in self.tables:
            rows += tbl.update(self.manager)
        return rows

    def get_last_ts(self) -> int:
        output = self.manager.execute(f'''
            SELECT timestamp
            FROM {self.ts_idx}
            ORDER BY ROWID 
            DESC LIMIT 1
        ''').fetchall()
        return output[0][0] if output else 0




