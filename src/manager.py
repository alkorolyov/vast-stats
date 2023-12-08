from __future__ import annotations

import logging
import sqlite3

import pandas as pd


class DbManager:
    """ Class responsible for basic database operations """
    db_path: str | None
    conn: sqlite3.Connection | None

    def __init__(self, db_path=None):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path) if self.db_path else sqlite3.connect(':memory:')
            # logging.debug(f"Connected to the database '{self.db_path}'")
        except sqlite3.Error as e:
            logging.error(f"Error connecting to the database '{self.db_path}': {e}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            # logging.debug(f"Closed connection to the database '{self.db_path}'.")
        else:
            logging.debug(f"Close connection to the database: no active connection")

    def vacuum(self):
        try:
            return self.conn.execute('VACUUM')
        except sqlite3.Error as e:
            logging.error(f"Error during VACUUM database: {e}")
            raise

    def execute(self, sql_query):
        try:
            return self.conn.execute(sql_query)
        except sqlite3.Error as e:
            logging.error(f"Error executing sql command: {sql_query}\n{e}")
            raise

    def commit(self):
        self.conn.commit()

    def create_table(self, name, cols):
        try:
            cols_str = ', '.join(cols)
            query = f'CREATE TABLE IF NOT EXISTS {name} ({cols_str})'
            self.conn.execute(query)
            self.conn.commit()
            logging.info(f"Table '{name}' created")
        except sqlite3.Error as e:
            logging.error(f"Error creating table '{name}': {e}")
            raise

    def delete_table(self, name):
        try:
            self.conn.execute(f'DROP TABLE IF EXISTS {name}')
            self.conn.commit()
            logging.info(f"Table '{name}' deleted")
        except sqlite3.Error as e:
            logging.error(f"Error deleting table'{name}': {e}")
            raise

    def clear_table(self, name):
        try:
            self.conn.execute(f'DELETE FROM {name}')
            self.conn.commit()
            logging.info(f"Table '{name}' deleted")
        except sqlite3.Error as e:
            logging.error(f"Error deleting table'{name}': {e}")
            raise

    def _insert(self, tbl_name, values: list):
        try:
            # single row
            val_str = ', '.join([str(x) for x in values])
            rcount = self.conn.execute(f'INSERT INTO {tbl_name} VALUES ({val_str})').rowcount
            self.conn.commit()
            logging.info(f"{rcount} Rows inserted into '{tbl_name}'")
        except sqlite3.Error as e:
            logging.error(f"Error inserting rows into '{tbl_name}': {e}")
            raise

    def insert(self, tbl_name, values: list):
        try:
            placeholder = ', '.join(['?'] * len(values[0]))
            rcount = self.conn.executemany(
                f'INSERT INTO {tbl_name} VALUES ({placeholder})',
                values).rowcount
            logging.info(f"{rcount} Rows inserted into '{tbl_name}'")
        except sqlite3.Error as e:
            logging.error(f"Error inserting rows into '{tbl_name}': {e}")
            raise

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


    def get_last_ts(self, name) -> int:
        output = self.execute(f'''
            SELECT timestamp
            FROM {name}
            ORDER BY ROWID
            DESC LIMIT 1
        ''').fetchall()
        return output[0][0] if output else 0

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

    def table_tolist(self, name) -> list:
        return self.table_to_df(name).values.tolist()

    def df_to_table(self, df, tbl_name):
        return df.to_sql(tbl_name, self.conn, if_exists='append', index=False)

        # cols_str = ", ".join([c for c in df.columns])
        # values = ('?, ' * len(df.columns))[:-2]
        #
        # self.conn.execute(f'CREATE TABLE IF NOT EXISTS {tbl_name} ({cols_str});')
        # self.conn.executemany(f"INSERT INTO {tbl_name} VALUES ({values})", df.values.tolist())

    def df_to_tmp_table(self, df, tbl_name):
        cols_str = ", ".join([c for c in df.columns])
        placeholder = ', '.join(['?'] * len(df.columns))
        self.conn.execute(f'CREATE TEMP TABLE IF NOT EXISTS {tbl_name} ({cols_str});')
        self.conn.executemany(f"INSERT INTO {tbl_name} VALUES ({placeholder})", df.values.tolist())
