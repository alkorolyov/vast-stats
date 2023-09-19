import sqlite3
import pandas as pd
import logging


class DbManager:
    """ Class responsible for basic database operations """
    db_path: str
    conn: sqlite3.Connection

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            logging.info(f"Connected to the database '{self.db_path}'")
        except sqlite3.Error as e:
            logging.exception(f"Error connecting to the database '{self.db_path}': {e}")

    def disconnect(self):
        if self.conn:
            self.conn.close()
            logging.info(f"Disconnected from the database '{self.db_path}'.")

    def execute(self, sql):
        return self.conn.execute(sql)

    def create_table(self, name, cols):
        try:
            cols_str = ', '.join(cols)
            query = f'CREATE TABLE IF NOT EXISTS {name} ({cols_str})'
            self.conn.execute(query)
            self.conn.commit()
            logging.info(f"Table '{name}' created")
        except sqlite3.Error as e:
            logging.exception(f"Error creating table '{name}': {e}")

    def delete_table(self, name):
        try:
            self.conn.execute(f'DROP TABLE IF EXISTS {name}')
            self.conn.commit()
            logging.info(f"Table '{name}' deleted")
        except sqlite3.Error as e:
            logging.exception(f"Error deleting table'{name}': {e}")

    def clear_table(self, name):
        try:
            self.conn.execute(f'DELETE FROM {name}')
            self.conn.commit()
            logging.info(f"Table '{name}' deleted")
        except sqlite3.Error as e:
            logging.exception(f"Error deleting table'{name}': {e}")

    def get_tables(self) -> list:
        res = self.conn.execute(f'''
        SELECT
            name
        FROM
               (SELECT * FROM sqlite_master UNION ALL
                SELECT * FROM sqlite_temp_master)
        WHERE
            type ='table' AND
            name NOT LIKE 'sqlite_%';
        ''').fetchall()
        return [x[0] for x in res]

    def get_tbl_info(self, name) -> pd.DataFrame:
        return pd.read_sql(f'PRAGMA table_info({name})', self.conn, index_col='cid')

    def get_db_stats(self) -> pd.DataFrame:
        query = '''
        WITH t AS (
            SELECT
                SUM(pageno) as total_pages,
                ROUND(CAST(sum(pgsize) AS FLOAT)/(1024 * 1024), 2) AS total_size
            FROM dbstat
            WHERE aggregate=TRUE
        )
        
        SELECT
            name,
            pageno as pages,
            ROUND(CAST(pgsize AS FLOAT)/(1024 * 1024), 2) as size,
            ROUND(pgsize * 100.0 / (t.total_size * 1024 * 1024), 2) AS percentage
        FROM
            (SELECT * FROM dbstat WHERE aggregate=TRUE)
        JOIN t
        
        UNION ALL
        
        SELECT
            'total' as name,
            total_pages as pages,
            total_size as size,
            100 as percentage
        FROM t
        ORDER BY percentage DESC;
        '''
        return pd.read_sql(query, self.conn)

    def get_tbl_timespan(self, name) -> pd.Timedelta:
        first = self.conn.execute(f'SELECT timestamp FROM {name} LIMIT 1').fetchall()[0][0]
        last = self.conn.execute(f'SELECT timestamp FROM {name} ORDER BY ROWID DESC LIMIT 1').fetchall()[0][0]
        return pd.to_timedelta((last - first) * 1e9)

    def table_to_df(self, name) -> pd.DataFrame:
        return pd.read_sql(f'SELECT * FROM {name}', self.conn)

    def df_to_table(self, df, tbl_name):
        return df.to_sql(tbl_name, self.conn, if_exists='append')

        # cols_str = ", ".join([c for c in df.columns])
        # values = ('?, ' * len(df.columns))[:-2]
        #
        # self.conn.execute(f'CREATE TABLE IF NOT EXISTS {tbl_name} ({cols_str});')
        # self.conn.executemany(f"INSERT INTO {tbl_name} VALUES ({values})", df.values.tolist())

    def df_to_tmp_table(self, df, tbl_name):
        cols_str = ", ".join([c for c in df.columns])
        values = ('?, ' * len(df.columns))[:-2]

        self.conn.execute(f'CREATE TEMP TABLE IF NOT EXISTS {tbl_name} ({cols_str});')
        self.conn.executemany(f"INSERT INTO {tbl_name} VALUES ({values})", df.values.tolist())


class DataBase:
    """ Class responsible for the Database structure """
    def __init__(self, db_manager: DbManager):
        self.db_manager = db_manager
        self.conn = db_manager.conn
        self.ts_idx = 'timestamp_idx'

    def create_ts_index(self):
        self.db_manager.create_table(self.ts_idx, ['timestamp INTEGER'])

    def get_last_ts(self) -> int:
        output = self.conn.execute(f'''
            SELECT timestamp
            FROM {self.ts_idx}
            ORDER BY ROWID 
            DESC LIMIT 1
        ''').fetchall()
        return output[0][0] if output else 0

