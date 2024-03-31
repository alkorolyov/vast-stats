from __future__ import annotations

import json
import requests
import logging

import pandas as pd
import datetime as dt
from urllib.parse import quote_plus

from src import const
from src.utils import time_utc_now, ts_utc_now
from src.manager import DbManager
from src.preprocess import preprocess, split_raw
from time import sleep, time

logging.getLogger("urllib3").setLevel(logging.WARNING)

def _apiurl(base_url: str, subpath: str, query_args: dict = None) -> str:
    """
    Creates the endpoint URL for a given combination of parameters.

    @param base_url: base url of the API server
    @param query_args: specifics such as API key and search parameters that complete the URL.
    @param subpath: added to end of URL to further specify endpoint
    @return: full url for API request
    """
    if query_args is None:
        query_args = {}
    if query_args:
        # a_list      = [<expression> for <l-expression> in <expression>]
        '''
        vector result;
        for (l_expression: expression) {
            result.push_back(expression);
        }
        '''
        # an_iterator = (<expression> for <l-expression> in <expression>)
        return base_url + "/api/v0" + subpath + "?" + "&".join(
            "{x}={y}".format(x=x, y=quote_plus(y if isinstance(y, str) else json.dumps(y))) for x, y in
            query_args.items())
    else:
        return base_url + "/api/v0" + subpath


def _get_sources():
    sources = []

    vast_exp_url = const.VAST_EXPORTER_BASEURL + '/machines'
    sources.append({
        'name': '500.farm',
        'url': vast_exp_url,
        'timeout': const.VAST_EXPORTER_TIMEOUT,
        'to_split': False,
    })

    # query = {"external": {"eq": False},  "disable_bundling": {"eq": True}, "type": "on-demand"}
    # vast_api_url = _apiurl(const.VAST_API_BASEURL, '/bundles', {'q': query})
    # sources.append({
    #     'name': 'vast.ai',
    #     'url': vast_api_url,
    #     'timeout': const.VAST_API_TIMEOUT,
    #     'to_split': True,
    # })

    return sources


def _get_raw(url, timeout) -> pd.DataFrame | None:

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    time_now = pd.Timestamp.utcnow()

    try:
        r_dict = r.json()
    except ValueError as e:
        logging.warning(f'JSON decoding failed from {url}: {e}')
        return None

    if not r_dict:
        logging.warning(f'Empty response from {url}')
        return None

    if 'offers' not in r_dict:
        logging.warning(f"No 'offers' in response from {url}")
        return None

    _ts = r_dict.get('timestamp', time_now)
    ts = int(pd.to_datetime(_ts).timestamp())

    raw = pd.DataFrame(r_dict['offers'])
    raw['timestamp'] = ts
    return raw


def fetch_single_source(source, max_tries=3) -> pd.DataFrame | None:
    # unpack params
    src_name, url, timeout, to_split = source['name'], source['url'], source['timeout'], source['to_split']
    logging.info(f"[API] fetching data from '{src_name}'")

    for i in range(1, max_tries + 1):
        try:
            raw = _get_raw(url, timeout)
            assert raw is not None

            preprocess(raw)
            if to_split:
                machines, _ = split_raw(raw)
            else:
                machines = raw
            return machines

        except AssertionError as e:
            logging.warning(f"[API] Empty 'raw' fetching from '{src_name}': {e}")
            logging.debug(f'[API] Attempt #{i}, retrying ...')
            sleep(const.RETRY_TIMEOUT)

        except requests.exceptions.HTTPError as e:
            logging.warning(f"[API] HTTPError fetching from '{src_name}': {e}")
            logging.debug(f'[API] Attempt #{i}, retrying ...')
            sleep(const.RETRY_TIMEOUT)

        except requests.exceptions.ConnectionError as e:
            logging.warning(f"[API] ConnectionError fetching from '{src_name}': {e}")
            logging.debug(f'[API] Attempt #{i}, retrying ...')
            sleep(const.RETRY_TIMEOUT)

        except requests.exceptions.Timeout as e:
            logging.warning(f"[API] Connection timeout to '{src_name}': {e}")
            logging.debug(f'[API] Attempt #{i}, retrying ...')
            sleep(const.RETRY_TIMEOUT)

    logging.warning(f"[API] Max attempts '{max_tries}' reached for '{src_name}'")
    return None


def fetch_sources(last_ts: int = 0) -> pd.DataFrame | None:
    """
    default source: 500.farm, fetch only 500.farm
    when ts == ts_db: wait 1x timeout, repeat
    when ts == ts_db and now - ts > 1x timeout: shift to vast source

    source: vast, fetch botch sources
    when now - 500farm_ts < 1x timeout: wait 1x timeout, shift back to 500.farm

    Fetch machine data from the sources.

    @param last_ts: last recorded timestamp in the database
    @return: machines dataframe
    """
    sources = _get_sources()
    for source in sources:
        machines = fetch_single_source(source)

        if machines is None:
            continue

        ts = machines.timestamp.iloc[0]

        # timestamp is too old - switch to next source
        # if time() - ts > 2 * const.TIMEOUT:
        #     logging.warning(f"[API] '{source['name']}' Response is too old {time() - ts:.1f}s")
        #     continue

        # if timestamp is already recorded return None
        if ts <= last_ts:
            logging.warning(f"[API] '{source['name']}' Response is already recorded")
            return None

        return machines

    logging.warning(f'[API] Failed to fetch data from any source')
    return None
