import math
import requests
import pandas as pd
from typing import List

from pandas import DataFrame
from src.utils import np_min_chunk, get_tbl_timespan, table_to_df

offers_url = 'https://500.farm/vastai-exporter/offers'
machines_url = 'https://500.farm/vastai-exporter/machines'

ALL_COLS = ['bundle_id', 'bw_nvlink', 'compute_cap', 'cpu_cores',
 'cpu_cores_effective', 'cpu_name', 'cpu_ram', 'credit_balance',
 'credit_discount', 'credit_discount_max', 'cuda_max_good',
 'direct_port_count', 'discount_rate', 'discounted_dph_total',
 'discounted_hourly', 'disk_bw', 'disk_name', 'disk_space', 'dlperf',
 'dlperf_per_dphtotal', 'dph_base', 'driver_version', 'duration',
 'end_date', 'flops_per_dphtotal', 'geolocation', 'gpu_display_active',
 'gpu_frac', 'gpu_lanes', 'gpu_mem_bw', 'gpu_name', 'gpu_ram', 'has_avx',
 'host_id', 'host_run_time', 'hosting_type', 'id', 'inet_down',
 'inet_down_cost', 'inet_up', 'inet_up_cost', 'machine_id', 'min_bid',
 'mobo_name', 'num_gpus', 'pci_gen', 'pcie_bw', 'public_ipaddr',
            'reliability2', 'rentable', 'score', 'start_date', 'storage_cost',
            'total_flops', 'verification', 'verified', 'timestamp']

INT32_COLS = ['has_avx', 'bw_nvlink', 'cpu_cores', 'cpu_ram', 'hosting_type', 'disk_space',
              'dlperf', 'score', 'verification', 'reliability',
              'dph_base', 'storage_cost', 'inet_up_cost', 'inet_down_cost', 'min_bid', 'credit_discount_max',
              'total_flops', 'disk_bw', 'gpu_mem_bw', 'inet_down',
              'inet_up', 'hosting_type', 'pcie_bw', 'rented', 'static_ip',
              'compute_cap', 'direct_port_count', 'end_date',
              'gpu_display_active', 'gpu_lanes', 'gpu_ram', 'host_id',
              'machine_id', 'id', 'min_chunk', 'num_gpus', 'pci_gen',
              'num_gpus_rented', 'timestamp', 'dph_base',
              ]

STR_COLS = ['cpu_name', 'cuda_max_good', 'disk_name', 'driver_version',
            'gpu_name', 'mobo_name', 'public_ipaddr', 'country']

DROP_COLS = ['credit_balance', 'credit_discount','location', 'geolocation', 'bundle_id',
             'discount_rate', 'discounted_dph_total', 'discounted_hourly',
             'dlperf_per_dphtotal', 'duration', 'flops_per_dphtotal', 'start_date',
             'verified', 'host_run_time', 'cpu_cores_effective', 'gpu_frac', 'chunks']

AVG_COLS = ['disk_bw', 'gpu_mem_bw', 'pcie_bw',
            'dlperf', 'score']

HARDWARE_COLS = ['compute_cap', 'total_flops',
                 'cpu_cores', 'cpu_name', 'has_avx',
                 'disk_name', 'hosting_type', 'mobo_name',
                 'gpu_name', 'num_gpus', 'pci_gen', 'gpu_lanes', 'gpu_ram', 'bw_nvlink']

EOD_COLS = ['cuda_max_good', 'driver_version', 'direct_port_count',
            'country', 'verification',
            'end_date', 'public_ipaddr', 'static_ip']

COST_COLS = ['dph_base', 'storage_cost', 'inet_up_cost',
             'inet_down_cost', 'min_bid', 'credit_discount_max']


def df_to_table(df, tbl_name, conn):
    cols = ", ".join([c for c in df.columns])
    values = ('?, ' * len(df.columns))[:-2]

    conn.execute(f'CREATE TABLE IF NOT EXISTS {tbl_name} ({cols});')
    conn.executemany(f"INSERT INTO {tbl_name} VALUES ({values})", df.values.tolist())


def df_to_tmp_table(df, tbl_name, conn):
    cols = ", ".join([c for c in df.columns])
    values = ('?, ' * len(df.columns))[:-2]

    conn.execute(f'CREATE TEMP TABLE IF NOT EXISTS {tbl_name} ({cols});')
    conn.executemany(f"INSERT INTO {tbl_name} VALUES ({values})", df.values.tolist())


def create_tmp_tables(offers: pd.DataFrame, conn):
    pass


def get_machines() -> DataFrame:
    return _get_raw(machines_url)


def get_offers() -> DataFrame:
    raw = _get_raw(offers_url)
    if raw is None:
        return None
    df = np_min_chunk(raw).reset_index(drop=True)
    return df


def _get_raw(url) -> pd.DataFrame:
    r = requests.get(url)

    ts = int(pd.to_datetime(r.json()['timestamp']).timestamp())
    raw = pd.DataFrame(r.json()["offers"])
    raw['timestamp'] = ts
    return raw


def _get_dtype(col_name: str) -> str:
    if col_name in INT32_COLS:
        return 'INTEGER'
    if col_name in STR_COLS:
        return 'TEXT'
    raise ValueError(f'Column name {col_name} not found in INTEGER or STRING column lists')


class Table:
    """
    Base Class for SQL tables.
    Each table has a name and must implement two methods,
    init_db, write_db.
    """
    def __init__(self, name: str, cols: list = None):
        self.name = name
        self.cols_list = cols

    def init_db(self, sql_conn):
        """
        General method to create new tables in database
        """
        pass

    def write_db(self, sql_conn):
        """
        General method to write data to database
        """
        pass


class MapTable(Table):
    def __init__(self, name: str, source: str, cols: list):
        super().__init__(name, cols)
        if len(cols) != 2:
            raise ValueError(f'Table error: two columns expected, got {len(cols)}')
        self.prim_key = cols[0]
        self.sec_key = cols[1]

        if source == 'machines':
            self.tmp_table = 'tmp_machines'
        elif source == 'offers':
            self.tmp_table = 'tmp_offers'
        else:
            raise ValueError(f'Table creation: source must be machines or offers, got {source}')

    def init_db(self, conn):
        conn.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.name} (
            {self.prim_key} INTEGER, 
            {self.sec_key} {_get_dtype(self.sec_key)}, 
            PRIMARY KEY ({self.prim_key}))'''
        )
    def write_db(self, conn) -> int:
        rowcount = conn.execute(f'''
        INSERT OR IGNORE INTO {self.name} ({self.prim_key}, {self.sec_key})
        SELECT t.{self.prim_key}, t.{self.sec_key} FROM {self.tmp_table} t
        ''').rowcount
        return rowcount


class Timeseries(Table):
    """
    A main class designed to efficiently store and manage time-series data in SQLite database.
    It optimizes disk space utilization by retaining only updated values while omitting
    unchanged ones. This optimization is achieved through the utilization of two distinct SQL tables:

    - tbl_name_ts: This table captures the time-series data containing only the altered values.
    - tbl_name_snp: This table holds the most recent snapshot of the time-series.

    The process involves invoking the write_db() method. When this method is called, the new values
    from the temporary table are compared against the latest snapshot. If any modifications are
    detected, only the changes are recorded within the timeseries table, and the snapshot is updated.

    This class presents an efficient solution for managing time-series data while minimizing the
    storage footprint by concentrating solely on altered data points.
    """

    # SQL helper strings
    cols: str           # value columns string: 'col1, col2, ...'
    t_cols: str         # 't.col1, t.col2, ...'
    cols_dtypes: str    # 'col1 INTEGER, col2 TEXT ...'
    key_col: str        # key column name
    tmp_table: str      # source temp table name
    select_altered: str # sql expression to select only altered values

    timeseries: str     # table name for timeseries
    snapshot: str       # table name for snapshot

    def __init__(self, name: str, source: str, cols: list):
        super().__init__(name, cols)
        if source == 'machines':
            self.key_col = 'machine_id'
            self.tmp_table = 'tmp_machines'
        elif source == 'offers':
            self.key_col = 'id'
            self.tmp_table = 'tmp_offers'
        else:
            raise ValueError(f'Table creation: source must be machines or offers, got {source}')

        self.cols = f"{', '.join([c for c in cols])}"
        self.t_cols = f"{', '.join([f't.{c}' for c in cols])}"

        self.cols_dtypes = f"{', '.join([f'{c} {_get_dtype(c)}' for c in cols])}"

        self.timeseries = name + '_ts'
        self.snapshot = name + '_snp'

        self.select_altered = f'''
        SELECT t.{self.key_col}, {self.t_cols}, t.timestamp FROM {self.tmp_table} t
        WHERE 
            ({self.key_col}, {self.cols}) NOT IN 
            (SELECT {self.key_col}, {self.cols} FROM {self.snapshot})        
        '''

    def init_db(self, conn):
        conn.execute(f''' 
        CREATE TABLE IF NOT EXISTS {self.timeseries} (
            {self.key_col} INTEGER, 
            {self.cols_dtypes}, 
            timestamp INTEGER) 
        ''')
        conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.snapshot} (
             {self.key_col} INTEGER, 
             {self.cols_dtypes},
             timestamp INTEGER,
             PRIMARY KEY ({self.key_col}))
        ''')

    def write_db(self, conn) -> int:
        rowcount = self._insert_timeseries(conn)
        self._update_snapshot(conn)
        return rowcount

    def _insert_timeseries(self, conn):
        rowcount = conn.execute(f''' 
        INSERT INTO {self.timeseries} ({self.key_col}, {self.cols}, timestamp)
        {self.select_altered}
        ''').rowcount
        return rowcount

    def _update_snapshot(self, conn):
        conn.execute(f'''
        INSERT OR REPLACE INTO {self.snapshot} ({self.key_col}, {self.cols}, timestamp)
        {self.select_altered}
        ''')


class AverageStd(Timeseries):
    def __init__(self, name: str, source: str, cols: list, period: str = '5 min'):
        super().__init__(name, source, cols)

        self.period = pd.to_timedelta(period)
        self.cols_avg_std = f"{', '.join([f'{c}_avg, {c}_std' for c in cols])}"
        self.cols_avg_std_dtypes = f"{', '.join([f'{c}_avg INTEGER, {c}_std REAL' for c in cols])}"

    def init_db(self, conn):
        conn.execute(f''' 
        CREATE TABLE IF NOT EXISTS {self.timeseries} (
            {self.key_col} INTEGER, 
            {self.cols_avg_std_dtypes}, 
            timestamp INTEGER) 
        ''')

        conn.execute(f''' 
        CREATE TABLE IF NOT EXISTS {self.snapshot} (
            {self.key_col} INTEGER, 
            {self.cols_dtypes}, 
            timestamp INTEGER) 
        ''')

    def write_db(self, conn):
        self._write_snapshot(conn)
        timespan = get_tbl_timespan(self.snapshot, conn)

        if timespan > self.period:
            rowcount = self._write_mean_std(conn)
            self._clear_snapshot(conn)
            return rowcount

        return 0

    def _write_mean_std(self, conn):
        conn.create_function('sqrt', 1, math.sqrt)
        avg_std_calc =f', \n\r'.join([
            f'ROUND(AVG({c})),\n\r'
            f'ROUND(sqrt(AVG({c} * {c}) - AVG({c}) * AVG({c})), 2)'
            for c in self.cols_list
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
            timestamp
        FROM {self.snapshot}
        GROUP BY {self.key_col}
        '''

        # print(sql_expression)
        rowcount = conn.execute(sql_expression)
        return rowcount


        # # calculate mean, std
        # df = table_to_df(self.snapshot, conn)
        # cols = self.cols_list + [self.key_col]
        # mean_df = df[cols].groupby(self.key_col).mean()
        # std_df = df[cols].groupby(self.key_col).std()
        #
        # for col in std_df:
        #     mask = std_df[col].isna()
        #     if mask.any():
        #         print(df.loc[std_df[mask].index])
        #         print(mean_df[mask])
        #         print(std_df[mask])
        #
        # df_ = pd.DataFrame(index=mean_df.index)
        # for col in self.cols_list:
        #     df_[col + '_avg'] = mean_df[col]
        #     df_[col + '_std'] = std_df[col]
        # df_['timestamp'] = df.timestamp.iloc[-1]
        #
        #
        #
        # # write to sql
        # df_.to_sql(self.timeseries, conn, if_exists='append')

    def _write_snapshot(self, conn) -> int:
        return conn.execute(f'''        
        INSERT INTO {self.snapshot}
        SELECT {self.key_col}, {self.cols}, timestamp FROM {self.tmp_table} AS t
        ''').rowcount

    def _clear_snapshot(self, conn):
        conn.execute(f'DELETE FROM {self.snapshot}')


class MachineTS(Timeseries):
    def init_db(self, conn):
        super().init_db(conn)
        conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_machine_id
            ON machine_split_snp (machine_id);
        ''')

    def _update_snapshot(self, conn):
        conn.execute(f'''
        DELETE FROM {self.snapshot}
        WHERE machine_id IN (SELECT machine_id FROM {self.tmp_table})
        AND id NOT IN (SELECT id FROM {self.tmp_table});
        ''')

        conn.execute(f'''
        INSERT INTO {self.snapshot} ({self.key_col}, {self.cols}, timestamp)
        SELECT t.{self.key_col}, {self.t_cols}, t.timestamp
        FROM {self.tmp_table} t
        LEFT JOIN {self.snapshot} ON {self.snapshot}.machine_id = t.machine_id
        WHERE {self.snapshot}.machine_id IS NULL;
        ''');


class OnlineTS:
    def __init__(self, name: str, machine_tbl: str):
        self.name = name
        self.machine_tbl = machine_tbl
        self.timeseries = name + '_ts'
        self.snapshot = name + '_snp'

    def init_db(self, conn):
        conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.timeseries} (
            machine_id INTEGER, 
            online INTEGER, 
            timestamp INTEGER
        )
        ''')
        conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.snapshot} (
            machine_id INTEGER, 
            online INTEGER, 
            timestamp INTEGER, 
            PRIMARY KEY (machine_id)
        )
        ''')

    def write_db(self, conn) -> int:
        # update online machines
        rowcount = conn.execute(f"""
        INSERT INTO {self.timeseries} (machine_id, online, timestamp)
        SELECT t.machine_id, 1, t.timestamp FROM tmp_machines t
        WHERE (t.machine_id, 1) NOT IN (SELECT machine_id, online FROM {self.snapshot});
        """).rowcount

        conn.execute(f"""
        INSERT OR REPLACE INTO {self.snapshot} (machine_id, online, timestamp)
        SELECT t.machine_id, 1, timestamp FROM tmp_machines t
        WHERE (t.machine_id, 1) NOT IN (SELECT machine_id, online FROM {self.snapshot});
        """)

        # update offline machines
        conn.execute(f"""
        WITH machines AS (
            SELECT t.machine_id
            FROM {self.machine_tbl} t
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

        conn.execute(f"""
        WITH machines AS (
        SELECT t.machine_id
        FROM {self.machine_tbl} t
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
    def write_db(self, conn) -> int:
        # update online machines
        rowcount = conn.execute(f"""
        INSERT INTO {self.timeseries} (machine_id, online, timestamp)
        SELECT t.machine_id, 1, t.timestamp
        FROM tmp_machines t
        LEFT JOIN {self.snapshot} s ON t.machine_id = s.machine_id AND s.online = 1
        WHERE s.machine_id IS NULL;
        """).rowcount


        conn.execute(f"""
        INSERT OR REPLACE INTO {self.snapshot} (machine_id, online, timestamp)
        SELECT t.machine_id, 1, t.timestamp
        FROM tmp_machines t
        LEFT JOIN {self.snapshot} s ON t.machine_id = s.machine_id AND s.online = 1
        WHERE s.machine_id IS NULL;
        """)

        # update offline machines
        conn.execute(f"""
        WITH machines AS (
            SELECT t.machine_id
            FROM {self.machine_tbl} t
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

        conn.execute(f"""
        WITH machines AS (
        SELECT DISTINCT t.machine_id
        FROM {self.machine_tbl} t
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

