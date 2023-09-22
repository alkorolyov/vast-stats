import sqlite3
import pytest
import pandas as pd

from src.tables import MapTable, df_to_tmp_table, get_tables, get_tbl_info, table_to_df

# Fixture to create and connect to the test database
@pytest.fixture()
def conn():
    test_conn = sqlite3.connect(':memory:')  # Use an in-memory database for testing
    # test_conn = sqlite3.connect('test.db')
    yield test_conn
    test_conn.commit()
    test_conn.close()


def test_df_to_table(conn):
    df = pd.DataFrame([{'id': 2, 'machine_id': 3}])
    df_to_tmp_table(df, 'tmp', conn)

    assert 'tmp' in get_tables(conn)
    records = conn.execute('SELECT * FROM tmp').fetchall()
    assert len(records) == 1
    assert records[0][0] == 2   # id
    assert records[0][1] == 3   # machine_id


def test_init_map_tbl(conn):
    conn.execute('DROP TABLE IF EXISTS map_tbl')
    map_tbl = MapTable(
            'map_tbl',
            'offers',
            ['id', 'machine_id']
        )
    map_tbl.init_db(conn)
    assert 'map_tbl' in get_tables(conn)
    info = get_tbl_info('map_tbl', conn)
    assert info.loc[info.name == 'id', 'pk'].all()
    assert (info.name == 'machine_id').any()
    # conn.close()


def test_write_map_tbl(conn):
    # conn.execute('DROP TABLE IF EXISTS tmp')
    conn.execute('DROP TABLE IF EXISTS tmp_offers')
    conn.execute('CREATE TABLE tmp_offers (id, machine_id)')
    conn.execute('INSERT INTO tmp_offers VALUES (1, 1)')
    conn.execute('INSERT INTO tmp_offers VALUES (2, 1)')

    conn.execute('DROP TABLE IF EXISTS map_tbl')

    map_tbl = MapTable(
        'map_tbl',
        'offers',
        ['id', 'machine_id']
    )

    map_tbl.init_db(conn)
    map_tbl.write_db(conn)
    records = conn.execute('SELECT * FROM map_tbl').fetchall()
    assert records[0] == (1, 1)
    assert records[1] == (2, 1)

    conn.execute('INSERT INTO tmp_offers VALUES (1, 2)')
    map_tbl.write_db(conn)
    assert conn.execute('SELECT * FROM map_tbl').fetchall()[0] == (1, 1)

