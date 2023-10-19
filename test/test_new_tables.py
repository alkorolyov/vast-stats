import sqlite3
import pytest
import pandas as pd
from src.tables import SingleTbl, Unique
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
    dbm.close()


def test_ts_idx_create(dbm, capsys):
    t = SingleTbl('t', ['timestamp'], 'tmp')
    t.create(dbm)
    assert 't' in dbm.get_tables()

    info = dbm.get_tbl_info('t')
    assert info.name[0] == 'timestamp'
    assert info.type[0] == 'INTEGER'
    assert info.loc[0, 'notnull'] == 0
    assert info.pk[0] == 1


def test_ts_idx_update(dbm, capsys):
    dbm.create_table('tmp', ['id', 'timestamp'])
    dbm.insert('tmp', [[1, 16], [2, 16]])
    t = SingleTbl('t', ['timestamp'], 'tmp')
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


def test_uniq_tbl_create(dbm, capsys):
    t = Unique('map', ['col1', 'col2'], 'tmp', 'id')
    t.create(dbm)

    assert 'map' in dbm.get_tables()
    info = dbm.get_tbl_info('map')
    assert ['id', 'col1', 'col2', 'timestamp'] == list(info['name'])
    # with capsys.disabled():
    #     print(info['name'].values)
    #     print()
    #     print(dbm.execute('SELECT * FROM map').fetchall())


def test_uniq_tbl_update(dbm, capsys):
    dbm.create_table('tmp', ['id', 'col1', 'timestamp'])
    dbm.insert('tmp', [
        [1, 1, 16],
        [2, 3, 16],
    ])
    t = Unique('map', ['col1'], 'tmp', 'id')
    t.create(dbm)
    t.update(dbm)
    rows = dbm.table_tolist('map')
    assert rows == [
        [1, 1, 16],
        [2, 3, 16],
    ]


def test_uniq_tbl_update_new_ts(dbm, capsys):
    t = Unique('t', ['col1'], 'tmp', 'id')
    t.create(dbm)
    dbm.insert('t', [
        [1, 1, 16],
        [2, 3, 16],
    ])
    # with capsys.disabled():
    #     print(dbm.get_tables())
    #     print(dbm.get_tbl_info('t'))
    #     print(dbm.table_tolist('t'))

    dbm.create_table('tmp', ['id', 'col1', 'timestamp'])
    dbm.insert('tmp', [
        [2, 3, 18],
    ])
    t.update(dbm)
    rows = dbm.table_tolist('t')
    assert rows == [
        [1, 1, 16],
        [2, 3, 16],
    ]


def test_uniq_tbl_update_new_value(dbm, capsys):
    t = Unique('t', ['col1'], 'tmp', 'id')
    t.create(dbm)
    dbm.insert('t', [
        [1, 1, 16],
        [2, 4, 16],
    ])
    dbm.create_table('tmp', ['id', 'col1', 'timestamp'])
    dbm.insert('tmp', [
        [1, 1, 22],
        [3, 4, 22],
    ])
    t.update(dbm)
    rows = dbm.table_tolist('t')
    assert rows == [
        [1, 1, 16],
        [2, 4, 16],
        [3, 4, 22],
    ]


def test_uniq_tbl_update_multi_vals(dbm, capsys):
    t = Unique('t', ['col1'], 'tmp', 'id')
    t.create(dbm)
    dbm.insert('t', [
        [1, 1, 16],
        [2, 1, 16],
    ])
    dbm.create_table('tmp', ['id', 'col1', 'timestamp'])
    dbm.insert('tmp', [
        [3, 1, 22],
        [4, 1, 22],
    ])
    t.update(dbm)
    rows = dbm.table_tolist('t')
    assert rows == [
        [1, 1, 16],
        [2, 1, 16],
        [3, 1, 22],
        [4, 1, 22],
    ]






