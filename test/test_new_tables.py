import sqlite3
import pytest
import pandas as pd
from src.new_tables import Table, MapTable
from src.manager import DbManager


# Fixture to create and connect to the test database
@pytest.fixture()
def dbm():
    # test_conn = sqlite3.connect(':memory:')  # Use an in-memory database for testing
    # conn = sqlite3.connect('test.db')
    dbm = DbManager()
    dbm.connect()
    yield dbm
    # test_conn.commit()
    dbm.disconnect()


def test_ts_table_create(dbm, capsys):
    t = Table('t', ['timestamp'], 'tmp')
    t.create(dbm)
    assert 't' in dbm.get_tables()

    info = dbm.get_tbl_info('t')
    assert info.name[0] == 'timestamp'
    assert info.type[0] == 'INTEGER'
    assert info.loc[0, 'notnull'] == 0
    assert info.pk[0] == 1


def test_ts_table_update(dbm, capsys):
    dbm.create_table('tmp', ['id', 'timestamp'])
    dbm.insert('tmp', [[1, 16], [2, 16]])
    t = Table('t', ['timestamp'], 'tmp')
    t.create(dbm)
    t.update(dbm)

    assert dbm.table_tolist('t') == [[16]]

    dbm.clear_table('tmp')
    dbm.insert('tmp', [[3, 16]])
    t.update(dbm)
    assert dbm.table_tolist('t') == [[16]]

    dbm.clear_table('tmp')
    dbm.insert('tmp', [[1, 32], [2, 32]])
    t.update(dbm)
    assert dbm.table_tolist('t') == [[16], [32]]


def test_map_table_create(dbm, capsys):
    t = MapTable('map', ['id', 'sid'], 'tmp')
    t.create(dbm)

    assert 'map' in dbm.get_tables()
    info = dbm.get_tbl_info('map')
    assert 'id' in info['name'].values
    assert 'sid' in info['name'].values
    assert 'timestamp' in info['name'].values
    # with capsys.disabled():
    #     print(info['name'].values)
    #     print()
    #     print(dbm.execute('SELECT * FROM map').fetchall())


def test_map_table_update(dbm, capsys):
    dbm.create_table('tmp', ['id', 'sid', 'timestamp'])
    dbm.insert('tmp', [
        [1, 1, 16],
        [2, 3, 16],
    ])
    t = MapTable('map', ['id', 'sid'], 'tmp')
    t.create(dbm)
    t.update(dbm)
    rows = dbm.table_tolist('map')
    assert rows == [
        [1, 1, 16],
        [2, 3, 16],
    ]


def test_map_table_update_new_ts(dbm, capsys):
    t = MapTable('t', ['id', 'sid'], 'tmp')
    t.create(dbm)
    dbm.insert('t', [
        [1, 1, 16],
        [2, 3, 16],
    ])
    # with capsys.disabled():
    #     print(dbm.get_tables())
    #     print(dbm.get_tbl_info('t'))
    #     print(dbm.table_tolist('t'))

    dbm.create_table('tmp', ['id', 'sid', 'timestamp'])
    dbm.insert('tmp', [
        [2, 3, 18],
    ])
    t.update(dbm)
    rows = dbm.table_tolist('t')
    assert rows == [
        [1, 1, 16],
        [2, 3, 16],
    ]


def test_map_table_update_new_value(dbm, capsys):
    t = MapTable('t', ['id', 'sid'], 'tmp')
    t.create(dbm)
    dbm.insert('t', [
        [1, 1, 16],
        [2, 3, 16],
    ])
    dbm.create_table('tmp', ['id', 'sid', 'timestamp'])
    dbm.insert('tmp', [
        [1, 1, 22],
        [2, 4, 22],
    ])
    t.update(dbm)
    rows = dbm.table_tolist('t')
    assert rows == [
        [1, 1, 16],
        [2, 3, 16],
        [2, 4, 22],
    ]








