from __future__ import annotations

import json
import pandas as pd
import requests
import logging
from urllib.parse import quote_plus
from src import const
from src.utils import time_utc_now
from src.manager import DbManager
from time import sleep


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

    vast_exp_url = const.VAST_EXPORTER_BASEURL + '/offers'
    sources.append({
        'name': '500.farm',
        'url': vast_exp_url,
        'timeout': const.VAST_EXPORTER_TIMEOUT
    })

    query = {"external": {"eq": False},  "disable_bundling": {"eq": True}, "type": "on-demand"}
    vast_api_url = _apiurl(const.VAST_API_BASEURL, '/bundles', {'q': query})
    sources.append({
        'name': 'vast.ai',
        'url': vast_api_url,
        'timeout': const.VAST_API_TIMEOUT
    })

    return sources


def _get_raw(url, timeout) -> pd.DataFrame | None:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

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

    _ts = r_dict.get('timestamp', time_utc_now())   # if time not present in raw, insert utc_now()
    ts = int(pd.to_datetime(_ts).timestamp())

    raw = pd.DataFrame(r_dict["offers"])
    raw['timestamp'] = ts
    return raw


def fetch_single(source, max_tries=3) -> pd.DataFrame | None:
    # unpack params
    src_name, url, timeout = source['name'], source['url'], source['timeout']
    logging.info(f"[API] fetching data from '{src_name}'")

    for i in range(1, max_tries + 1):
        try:
            return _get_raw(url, timeout)
        except requests.exceptions.HTTPError as e:
            logging.warning(f"[API] failed to fetch data from '{src_name}': {e}")
            logging.info(f'[API] attempt #{i}, retrying ...')
            sleep(const.RETRY_TIMEOUT)

    logging.warning(f'[API] Max attempts {max_tries} reached for {src_name}')
    return None


def fetch(last_ts: int = 0) -> pd.DataFrame | None:
    sources = _get_sources()
    for source in sources:
        raw = fetch_single(source)
        if raw is not None:
            return raw
    return None
