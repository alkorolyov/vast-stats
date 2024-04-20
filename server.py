from __future__ import annotations

import json

import pandas as pd
from time import time

import http.server
import socketserver
import sqlite3
import argparse
import logging


import gzip
from urllib.parse import urlparse, parse_qs
from http import HTTPStatus

from src.utils import time_ms
from src import const
from src.vastdb import VastDB

logging.basicConfig(format=const.LOG_FORMAT,
                    level=logging.DEBUG,
                    datefmt='%d-%m-%Y %I:%M:%S')


def datetime_to_ts(date: str):
    return int(pd.to_datetime(date).timestamp())


def parse_params(params: dict) -> tuple:
    machine_id = params.get('machine_id', [None])[0]
    from_date, to_date = params.get('from', [None])[0], params.get('to', [None])[0]

    if machine_id is None:
        raise ValueError('machine_id is required')

    try:
        machine_id = int(machine_id)
    except ValueError:
        raise ValueError(f'machine_id should be an integer: {machine_id}')

    from_timestamp = None
    if from_date:
        from_timestamp = datetime_to_ts(from_date)

    to_timestamp = None
    if to_date:
        to_timestamp = datetime_to_ts(to_date)

    return machine_id, from_timestamp, to_timestamp


def get_dbrequest_sql(params: dict, db_table: str) -> str:
    # unpack parameters
    machine_ids, from_ts, to_ts = parse_params(params)

    sql_query = f"SELECT * FROM {db_table} WHERE"

    # add machine_id
    sql_query += f" machine_id IN ({','.join(machine_ids)})"

    # add 'from' and 'to' constraints
    if from_ts:
        sql_query += f" AND timestamp >= '{from_ts}'"
    if to_ts:
        sql_query += f" AND timestamp <= '{to_ts}'"

    return sql_query


def get_last_value_sql(params: dict, db_table: str) -> str:
    machine_ids, _, _ = parse_params(params)
    return f"SELECT * FROM {db_table} WHERE machine_id={machine_ids[0]} ORDER BY timestamp DESC LIMIT 1"


def compress_data(json_data):
    start = time()
    compressed = gzip.compress(json_data, compresslevel=1)
    logging.debug(f"compress json:     {time_ms(time() - start)} ms")
    logging.debug(f"compression ratio: {len(compressed) / len(json_data) * 100:.1f}%")
    logging.debug(f"data size:         {len(compressed)} bytes")
    return compressed


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        logging.debug(f"Received request: {self.request}")

        parsed_url = urlparse(self.path)
        logging.debug(f"parsed_url: {parsed_url.path}")
        query_params = parse_qs(parsed_url.query)
        logging.debug(f"query_params: {query_params}")

        if parsed_url.path == '/reliability':
            self.handle_db_request(query_params, 'reliability_ts')
        elif parsed_url.path == '/rent':
            self.handle_db_request(query_params, 'rent_ts')
        elif parsed_url.path == '/plot':
            self.handle_plot_request()
        elif parsed_url.path == '/stats':
            self.handle_stats_request(query_params)
        elif parsed_url.path == '/test':
            self.handle_test_request()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)


    def handle_test_request(self) -> None:
        vastdb = self.server.vastdb

        logging.getLogger().setLevel(logging.INFO)
        logging.info(f"Testing request 2 weeks data")



        vastdb.connect()
        machines_list = pd.read_sql('SELECT machine_id FROM machine_host_map', vastdb.dbm.conn).machine_id.sample(10)
        vastdb.close()

        start_time = time()
        for machine_id in machines_list:
        # for machine_id in [4557, 13058, 13528, 12910, 13539, 12110, 12951, 10520, 13641, 9977]:
            json_data = vastdb.get_machine_stats(machine_id, datetime_to_ts('2024-03-06'), None)
        #     json_data = vastdb.get_machine_stats(machine_id, datetime_to_ts('2024'), None)
            logging.info(f"machine_id: {machine_id} {time() - start_time:.1f} s")

        logging.getLogger().setLevel(logging.DEBUG)

        self.send_response(HTTPStatus.OK)

    def handle_stats_request(self, query_params: dict) -> dict | None:
        vastdb = self.server.vastdb
        try:
            machine_id, from_ts, to_ts = parse_params(query_params)
        except ValueError as e:
            self.send_error(HTTPStatus.BAD_REQUEST, f'Error parsing params {query_params} {e}', str(e))

        try:
            json_data = vastdb.get_machine_stats(machine_id, from_ts, to_ts)
        except pd.errors.DatabaseError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR,
                            f'Pandas DatabaseError {e}', str(e))

        try:
            compressed = compress_data(json_data)
            self.send_compressed_json(compressed)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f'Error compressing json {query_params}', str(e))
        return

    def handle_db_request(self, query_params: dict, db_table: str) -> None:
        try:
            with sqlite3.connect(self.server.db_path) as conn:
                sql_query = get_dbrequest_sql(query_params, db_table)

                start = time()
                df = pd.read_sql_query(sql_query, conn)
                logging.debug(f"sql request: {time_ms(time() - start)} ms")

            json_data = df.to_json(orient='records').encode('utf-8')

            start = time()
            compressed_data = gzip.compress(json_data, compresslevel=1)
            logging.debug(f"compress json:     {time_ms(time() - start)} ms")
            logging.debug(f"compression ratio: {len(compressed_data) / len(json_data) * 100:.1f}%")
            logging.debug(f"data size:         {len(compressed_data)} bytes")

            self.send_compressed_json(compressed_data)

        except sqlite3.DatabaseError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f'SQLite DatabaseError {query_params} {db_table}', str(e))
        except pd.errors.DatabaseError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f'Pandas DatabaseError {query_params} {sql_query} {db_table}', str(e))
        except ValueError as e:
            self.send_error(HTTPStatus.BAD_REQUEST, f'ValueError during Parsing {query_params}', str(e))
        except TypeError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f'GET TypeError', str(e))
        except OSError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"GET OSError", str(e))
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, 'GET General exception', str(e))

    def handle_plot_request(self):
        try:
            # Load HTML file
            with open('index.html', 'rb') as f:
                html_content = f.read()

            # Send HTML response to client
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content)

        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, 'Error serving HTML file', str(e))

    def send_compressed_json(self, compressed_data):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Encoding', 'gzip')
        self.end_headers()
        self.wfile.write(compressed_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Vast Stats WebServer')
    parser.add_argument('-p', '--port', type=int, default=8080, help='port to listen on')
    parser.add_argument('--db_path', type=str, default='./vast.db', help='path to database')

    args = vars(parser.parse_args())
    db_path = args.get('db_path')
    port = args.get('port')

    with socketserver.TCPServer(("", port), RequestHandler) as httpd:
        httpd.vastdb = VastDB(db_path)
        logging.debug(db_path)
        logging.debug(f"Server listening on port {port}")
        httpd.serve_forever()
