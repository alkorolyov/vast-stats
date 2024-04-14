from __future__ import annotations

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

logging.basicConfig(format=const.LOG_FORMAT,
                    level=logging.DEBUG,
                    datefmt='%d-%m-%Y %I:%M:%S')


def datetime_to_ts(date: str):
    return int(pd.to_datetime(date).timestamp())


def get_reliability_sql(params: dict) -> str:
    # unpack parameters
    machine_ids = params.get('machine_id')
    from_dt, to_dt = params.get('from', [None])[0], params.get('to', [None])[0]

    if machine_ids is None:
        raise ValueError('machine_id is required')

    sql_query = "SELECT * FROM reliability_ts WHERE"

    # add machine_id
    sql_query += f" machine_id IN ({','.join(machine_ids)})"

    # add 'from' and 'to' constraints
    if from_dt:
        sql_query += f" AND timestamp >= '{datetime_to_ts(from_dt)}'"
    if to_dt:
        sql_query += f" AND timestamp <= '{datetime_to_ts(to_dt)}'"

    return sql_query


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        logging.debug(self.server.db_path)
        logging.debug(f"Received request: {self.request}")

        parsed_url = urlparse(self.path)
        logging.debug(f"parsed_url: {parsed_url.path}")
        query_params = parse_qs(parsed_url.query)
        logging.debug(f"query_params: {query_params}")

        if parsed_url.path == '/reliability':
            self.handle_reliability_request(query_params)
        elif parsed_url.path == '/plot':
            self.handle_plot_request()

    def handle_reliability_request(self, query_params: dict) -> None:
        try:
            with sqlite3.connect(self.server.db_path) as conn:
                sql_query = get_reliability_sql(query_params)

                start = time()
                df = pd.read_sql_query(sql_query, conn)
                logging.debug(f"sql request: {time_ms(time() - start)} ms")

            json_data = df.to_json(orient='records').encode('utf-8')

            start = time()
            compressed_data = gzip.compress(json_data, compresslevel=1)
            logging.debug(f"compress json: {time_ms(time() - start)} ms")
            logging.debug(f"compression ratio: {len(compressed_data) / len(json_data) * 100:.1f}%")

            self.send_compressed_data(compressed_data)

        except sqlite3.DatabaseError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f'SQLite DatabaseError {query_params}', str(e))
        except pd.errors.DatabaseError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f'Pandas DatabaseError {query_params} {sql_query}', str(e))
        except ValueError as e:
            self.send_error(HTTPStatus.BAD_REQUEST, f'ValueError during Parsing {query_params}', str(e))
        except TypeError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f'Reliability GET TypeError', str(e))
        except OSError as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Reliability GET OSError", str(e))
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, 'Reliability GET General exception', str(e))

    def send_compressed_data(self, compressed_data):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-Encoding', 'gzip')
        self.end_headers()
        self.wfile.write(compressed_data)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Vast Stats WebServer')
    parser.add_argument('-p', '--port', type=int, default=8080, help='port to listen on')
    parser.add_argument('--db_path', type=str, default='./vast.db', help='path to database')

    args = vars(parser.parse_args())
    db_path = args.get('db_path')
    port = args.get('port')

    with socketserver.TCPServer(("", port), RequestHandler) as httpd:
        httpd.db_path = db_path
        print(f"Server listening on port {port}")
        httpd.serve_forever()
