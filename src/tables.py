import logging
import math
from time import time
from typing import List


import pandas as pd

from src import const
from src.manager import DbManager
from src.utils import time_ms


def _get_dtype(col_name: str) -> str:
    if col_name in const.INT_COLS:
        return 'INTEGER'
    elif col_name in const.STR_COLS:
        return 'TEXT'
    elif col_name in const.FLOAT_COLS:
        return 'REAL'
    else:
        return ''
    # raise ValueError(f'Error getting column dtype: {col_name} not found')


class _Table:
    """
    Base Class for SQL tables.
    Extracts data from the temporary source table from memory
    and writes it to database. Each table must implement two methods:
    create(), update().
    """
    name: str                   # table name
    cols: List[str]             # list of column names
    source: str                 # name of the temporary source table

    def __init__(self, name: str, cols: list = None, source: str = None):
        self.name = name
        self.cols = cols
        self.source = source

    def create(self, dbm: DbManager):
        """
        General method to create new table in database
        """
        pass

    def update(self, dbm: DbManager):
        """
        General method to update data in database
        """
        pass


class SingleTbl(_Table):
    """
    A Simple table with one column as a primary key.
    First row from the source tmp column is taken.
    """

    def __init__(self, name, cols, source='tmp_machines'):
        super().__init__(name, cols, source)
        if len(cols) != 1:
            raise ValueError(f'Error creating Table {name}: single column required')
        self.col_name = self.cols[0]

    def create(self, dbm):
        # dbm.create_table(self.name, self.cols)
        dbm.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.name} (
            {self.col_name} {_get_dtype(self.col_name)} PRIMARY KEY
            )
        ''')

    def update(self, dbm) -> int:
        rowcount = dbm.execute(f'''
            INSERT OR IGNORE INTO {self.name}
            SELECT DISTINCT {self.col_name} FROM {self.source}
        ''').rowcount
        return rowcount


class MapTable(_Table):
    def __init__(self, name,  cols, source='tmp_machines'):
        super().__init__(name, cols, source)
        if len(cols) != 2:
            raise ValueError(f'Table error: two columns expected, got {len(cols)}')
        self.key_col = cols[0]
        self.val_col = cols[1]

    def create(self, dbm):
        dbm.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.name} (
            {self.key_col} {_get_dtype(self.key_col)} PRIMARY KEY, 
            {self.val_col} {_get_dtype(self.val_col)}
            )
        ''')

    def update(self, dbm) -> int:
        rowcount = dbm.execute(f'''
        INSERT OR IGNORE INTO {self.name} ({self.key_col}, {self.val_col})
        SELECT t.{self.key_col}, t.{self.val_col} FROM {self.source} t
        ''').rowcount
        return rowcount


class Unique(_Table):
    """
    Unique key table, which stores a history of unique keys with additional value columns.
    Each time a new key is found in source table it is stored with value columns and timestamp.
    If key is already present in table - nothing is done.
    """
    def __init__(self, name: str, cols: list, source: str, key_col: str):
        super().__init__(name, cols, source)
        self.key_col = key_col
        self.cols_str = f"{', '.join([c for c in cols])}"
        self.t_cols = f"{', '.join([f't.{c}' for c in cols])}"
        self.cols_dtypes = f"{', '.join([f'{c} {_get_dtype(c)}' for c in cols])}"

    def create(self, dbm):
        dbm.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.name} ( 
            {self.key_col} INTEGER,
            {self.cols_dtypes},            
            timestamp INTEGER
            )
        ''')

    def update(self, dbm) -> int:
        # rowcount = dbm.execute(f'''
        #     INSERT OR IGNORE INTO {self.name} ({self.key_col}, {self.val_col})
        #     SELECT t.{self.key_col}, t.{self.val_col} FROM {self.source} t
        # ''').rowcount
        rowcount = dbm.execute(f'''
            INSERT OR IGNORE INTO {self.name}
            ({self.key_col}, {self.cols_str}, timestamp)
            SELECT t.{self.key_col}, {self.t_cols}, t.timestamp 
            FROM {self.source} AS t
            LEFT JOIN {self.name} AS t2 ON t.{self.key_col} = t2.{self.key_col}
            WHERE t2.{self.key_col} IS NULL
        ''').rowcount
        return rowcount


class Timeseries(_Table):
    """
    Main class designed to efficiently store and manage time-series data in SQLite database.
    It optimizes disk space utilization by retaining only updated values while omitting
    unchanged ones. This optimization is achieved through the utilization of two distinct SQL tables:

    - tablename_ts: This table captures the time-series data containing only the altered values.
    - tablename_snp: This table holds the most recent snapshot of the time-series.

    The process involves invoking the update() method. When this method is called, the new values
    from the temporary table are compared against the latest snapshot. If any modifications are
    detected, only the changes are recorded within the timeseries table, and the snapshot is updated.

    This class presents an efficient solution for managing time-series data while minimizing the
    storage footprint by concentrating solely on altered data points.
    """

    # SQL helper strings
    cols_str: str  # value columns string: 'col1, col2, ...'
    t_cols: str  # 't.col1, t.col2, ...'
    cols_dtypes: str  # 'col1 INTEGER, col2 TEXT ...'
    key_col: str  # primary key column name
    source: str  # source temp table name
    from_altered: str  # sql expression to select only altered values

    timeseries: str  # table name for timeseries
    snapshot: str  # table name for snapshot

    def __init__(self, name, cols, source='tmp_machines', key_col='machine_id'):
        super().__init__(name, cols, source)

        self.key_col = key_col

        self.cols_str = f"{', '.join([c for c in cols])}"
        self.t_cols = f"{', '.join([f't.{c}' for c in cols])}"

        self.cols_dtypes = f"{', '.join([f'{c} {_get_dtype(c)}' for c in cols])}"

        self.timeseries = name + '_ts'
        self.snapshot = name + '_snp'

        self.from_altered = f'''         
        FROM {self.source} t
        WHERE 
            ({self.key_col}, {self.cols_str}) NOT IN 
            (SELECT {self.key_col}, {self.cols_str} FROM {self.snapshot})
        '''

    def create(self, dbm):
        dbm.execute(f''' 
        CREATE TABLE IF NOT EXISTS {self.timeseries} (
            {self.key_col} INTEGER, 
            {self.cols_dtypes}, 
            timestamp INTEGER
--             FOREIGN KEY (timestamp)
--                 REFERENCES timestamp_tbl (timestamp)
            ) 
        ''')
        dbm.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.snapshot} (
             {self.key_col} INTEGER PRIMARY KEY, 
             {self.cols_dtypes}
             )
        ''')

    def update(self, dbm) -> int:
        rowcount = self._insert_timeseries(dbm)
        self._update_snapshot(dbm)
        return rowcount

    def _insert_timeseries(self, dbm):
        rowcount = dbm.execute(f''' 
        INSERT INTO {self.timeseries} ({self.key_col}, {self.cols_str}, timestamp)
        SELECT t.{self.key_col}, {self.t_cols}, t.timestamp
        {self.from_altered}
        ''').rowcount
        return rowcount

        # rowcount = dbm.execute(f'''
        # WITH altered AS (
        #     SELECT t.{self.key_col}, {self.t_cols}, t.timestamp
        #     FROM {self.source} t
        #     WHERE
        #         ({self.key_col}, {self.cols_str}) NOT IN
        #         (SELECT {self.key_col}, {self.cols_str} FROM {self.snapshot})
        # )
        # INSERT INTO {self.timeseries}
        # SELECT * FROM altered
        # ''').rowcount
        # return rowcount

    def _update_snapshot(self, dbm):
        # dbm.clear_table(self.snapshot)
        # dbm.execute(f'''
        # INSERT INTO {self.snapshot} ({self.key_col}, {self.cols_str})
        # SELECT {self.key_col}, {self.cols_str} FROM {self.source}
        # ''')
        dbm.execute(f'''
        INSERT OR REPLACE INTO {self.snapshot} ({self.key_col}, {self.cols_str})
        SELECT t.{self.key_col}, {self.t_cols}
        {self.from_altered}
        ''')


class AverageStd(Timeseries):
    def __init__(self, name: str, cols: list, source: str = 'tmp_machines', period: str = '1 day'):
        super().__init__(name, cols, source)
        self.period = period
        self.cols_avg_std = f"{', '.join([f'{c}_avg, {c}_std' for c in cols])}"
        self.cols_avg_std_dtypes = f"{', '.join([f'{c}_avg INTEGER, {c}_std REAL' for c in cols])}"

    def create(self, dbm):
        dbm.execute(f''' 
        CREATE TABLE IF NOT EXISTS {self.timeseries} (
            {self.key_col} INTEGER, 
            {self.cols_avg_std_dtypes}, 
            timestamp INTEGER) 
        ''')

        dbm.execute(f''' 
        CREATE TABLE IF NOT EXISTS {self.snapshot} (
            {self.key_col} INTEGER, 
            {self.cols_dtypes}, 
            timestamp INTEGER) 
        ''')

    def update(self, dbm):
        ts = dbm.get_last_ts(self.source)
        last_ts = dbm.get_last_ts(self.snapshot)

        period = int(pd.to_timedelta(self.period).total_seconds())
        period_end = math.ceil(last_ts / period) * period

        rowcount = 0

        # check if end of period
        if ts > period_end:
            start = time()
            rowcount = self._write_mean_std(dbm, period_end)
            self._clear_snapshot(dbm)
            dbm.commit()
            dbm.vacuum()
            logging.info(f"[{self.name.upper()}] Average calculated over '{self.period}' {time_ms(time() - start)}")

        self._write_snapshot(dbm)

        return rowcount

    def _write_mean_std(self, dbm, period_end: int):
        dbm.conn.create_function('SQRT', 1, math.sqrt)

        avg_std_calc = f', \n\r'.join([
            f'ROUND(AVG({c})),\n\r'
            f'ROUND(SQRT(AVG({c} * {c}) - AVG({c}) * AVG({c})), 2)'
            for c in self.cols
        ])

        sql_expression = f'''
        INSERT INTO {self.timeseries} (
            {self.key_col},
            {self.cols_avg_std},
            timestamp
        )
        SELECT
            {self.key_col}, 
            {avg_std_calc},
            {period_end}
        FROM {self.snapshot}
        GROUP BY {self.key_col}
        '''

        # print(sql_expression)
        rowcount = dbm.execute(sql_expression).rowcount
        return rowcount

    def _write_snapshot(self, dbm):
        dbm.execute(f'''        
            INSERT INTO {self.snapshot}
            SELECT {self.key_col}, {self.cols_str}, timestamp FROM {self.source} AS t
        ''')

    def _clear_snapshot(self, dbm):
        dbm.execute(f'DELETE FROM {self.snapshot}')


class OnlineTS:
    def __init__(self, name: str, machines_tbl: str):
        self.name = name
        self.machines_tbl = machines_tbl
        self.timeseries = name + '_ts'
        self.snapshot = name + '_snp'

    def create(self, dbm):
        dbm.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.timeseries} (
            machine_id INTEGER, 
            online INTEGER, 
            timestamp INTEGER
        )
        ''')
        dbm.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.snapshot} (
            machine_id INTEGER, 
            online INTEGER, 
            timestamp INTEGER, 
            PRIMARY KEY (machine_id)
        )
        ''')

    def update(self, dbm) -> int:
        # update online machines
        rowcount = dbm.execute(f"""
        INSERT INTO {self.timeseries} (machine_id, online, timestamp)
        SELECT t.machine_id, 1, t.timestamp FROM tmp_machines t
        WHERE (t.machine_id, 1) NOT IN (SELECT machine_id, online FROM {self.snapshot});
        """).rowcount

        dbm.execute(f"""
        INSERT OR REPLACE INTO {self.snapshot} (machine_id, online, timestamp)
        SELECT t.machine_id, 1, timestamp FROM tmp_machines t
        WHERE (t.machine_id, 1) NOT IN (SELECT machine_id, online FROM {self.snapshot});
        """)

        # update offline machines
        dbm.execute(f"""
        WITH machines AS (
            SELECT t.machine_id
            FROM {self.machines_tbl} t
            WHERE t.machine_id NOT IN
                  (SELECT machine_id FROM tmp_machines)
        ),
        offline_machines AS (
            SELECT machine_id, 0, t.timestamp
            FROM machines
            CROSS JOIN (SELECT timestamp FROM tmp_machines LIMIT 1) t    
        )
        INSERT INTO {self.timeseries} (machine_id, online, timestamp)
        SELECT * FROM offline_machines
        WHERE (machine_id, 0) NOT IN (SELECT machine_id, online from {self.snapshot});
        """)

        dbm.execute(f"""
        WITH machines AS (
        SELECT t.machine_id
        FROM {self.machines_tbl} t
        WHERE t.machine_id NOT IN
              (SELECT machine_id FROM tmp_machines)
        ),
        offline_machines AS (
         SELECT machine_id, 0, t.timestamp
         FROM machines
                  CROSS JOIN (SELECT timestamp FROM tmp_machines LIMIT 1) t
        )
        INSERT OR REPLACE INTO {self.snapshot} (machine_id, online, timestamp)
        SELECT * FROM offline_machines
        WHERE (machine_id, 0) NOT IN (SELECT machine_id, online from {self.snapshot});
        """)

        return rowcount


class NewOnlineTS(OnlineTS):
    def write_db(self, dbm) -> int:
        # update online machines
        rowcount = dbm.execute(f"""
        INSERT INTO {self.timeseries} (machine_id, online, timestamp)
        SELECT t.machine_id, 1, t.timestamp
        FROM tmp_machines t
        LEFT JOIN {self.snapshot} s ON t.machine_id = s.machine_id AND s.online = 1
        WHERE s.machine_id IS NULL;
        """).rowcount

        dbm.execute(f"""
        INSERT OR REPLACE INTO {self.snapshot} (machine_id, online, timestamp)
        SELECT t.machine_id, 1, t.timestamp
        FROM tmp_machines t
        LEFT JOIN {self.snapshot} s ON t.machine_id = s.machine_id AND s.online = 1
        WHERE s.machine_id IS NULL;
        """)

        # update offline machines
        dbm.execute(f"""
        WITH machines AS (
            SELECT t.machine_id
            FROM {self.machines_tbl} t
            WHERE t.machine_id NOT IN
                  (SELECT machine_id FROM tmp_machines)
        ),
        offline_machines AS (
            SELECT machine_id, 0, t.timestamp
            FROM machines
            CROSS JOIN (SELECT timestamp FROM tmp_machines LIMIT 1) t    
        )
        INSERT INTO {self.timeseries} (machine_id, online, timestamp)
        SELECT * FROM offline_machines
        WHERE (machine_id, 0) NOT IN (SELECT machine_id, online from {self.snapshot});
        """)

        dbm.execute(f"""
        WITH machines AS (
        SELECT DISTINCT t.machine_id
        FROM {self.machines_tbl} t
        WHERE t.machine_id NOT IN
              (SELECT machine_id FROM tmp_machines)
        ),
        offline_machines AS (
         SELECT machine_id, 0, t.timestamp
         FROM machines
                  CROSS JOIN (SELECT timestamp FROM tmp_machines LIMIT 1) t
        )
        INSERT OR REPLACE INTO {self.snapshot} (machine_id, online, timestamp)
        SELECT * FROM offline_machines
        WHERE (machine_id, 0) NOT IN (SELECT machine_id, online from {self.snapshot});
        """)

        return rowcount