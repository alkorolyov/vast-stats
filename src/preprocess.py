import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_integer_dtype, is_float_dtype, is_string_dtype

from src import const
from src.utils import round_day


def _round_ram(cpu_ram: pd.Series):
    """ Round cpu_ram to the nearest fraction of power of two """

    power_of_two = np.power(2, np.log2(cpu_ram).round())

    mask_lo = np.abs(cpu_ram - power_of_two) > np.abs(cpu_ram - power_of_two / 4 * 3)
    mask_hi = np.abs(cpu_ram - power_of_two) > np.abs(cpu_ram - power_of_two / 2 * 3)
    mask_ex = ~mask_lo & ~mask_hi

    res = cpu_ram.copy()

    res[mask_lo] = power_of_two[mask_lo] / 4 * 3
    res[mask_hi] = power_of_two[mask_hi] / 2 * 3
    res[mask_ex] = power_of_two[mask_ex]

    return res.astype(int)


def _set_isp(raw: pd.DataFrame):
    # pop isp data from 'location' dictionary
    raw['isp'] = raw.location.apply(lambda x: x.pop('isp', None) if not pd.isna(x) else None)


def _set_country(raw: pd.DataFrame):
    # try to get country from geomax 'location' dict
    raw.loc[raw.location.isna(), 'location'] = None
    country_loc = raw.location.apply(lambda x: x['country'] if x else None)
    raw['country'] = country_loc

    # fill missing values from vast provided 'geolocation' field
    mask = raw.country.isna()
    country_code = raw.geolocation.str.split(',').apply(lambda x: x[-1] if x else None).str.strip().replace('Sweden', 'SE')
    raw.loc[mask, 'country'] = country_code[mask]


def _fillna(raw: pd.DataFrame):
    """ Fill NA's:  0 for numerical types
                    '' for string types
    """

    for col in raw.columns:
        dtype = raw[col].dtype
        if is_integer_dtype(dtype) or is_float_dtype(dtype):
            raw.fillna({col: 0}, inplace=True)
        elif is_string_dtype(dtype):
            raw.fillna({col: ''}, inplace=True)


def _conv_to_int(raw: pd.DataFrame, cols: list):
    for col in cols:
        if raw.index.name == col:
            raw.index = np.round(raw.index).astype('uint32')
        if col not in raw:
            continue
        raw[col] = raw[col].round().astype('uint32')


def _conv_to_str(raw: pd.DataFrame, cols: list):
    for col in cols:
        if raw.index.name == col:
            raw.index = raw.index.astype(str).str.strip()
        if col not in raw:
            continue
        raw[col] = raw[col].astype(str).str.strip()


def _rename_cols(raw):
    if 'rentable' in raw:
        raw.drop(columns=['rented'], inplace=True, errors='ignore')
        raw.rentable = ~raw.rentable
        raw.rename(columns={'rentable': 'rented'}, inplace=True)

    if 'reliability2' in raw:
        raw.drop(columns=['reliability'], inplace=True, errors='ignore')
        raw.rename(columns={'reliability2': 'reliability'}, inplace=True)


def preprocess(raw: pd.DataFrame):
    _rename_cols(raw)

    _set_isp(raw)
    _set_country(raw)
    raw.verification = raw.verification.map(const.VERIFIED_ENUM)

    _fillna(raw)

    # Hardware
    raw.bw_nvlink = raw.bw_nvlink.round(-1)
    raw.cpu_ram = raw.cpu_ram / 1024  # RAM in Gb
    # raw['cpu_ram_rnd'] = _round_ram(raw.cpu_ram)
    raw.disk_space = raw.disk_space.round(-1).replace(0, 10)

    # Average cols
    raw.pcie_bw = raw.pcie_bw * 10
    # raw.gpu_mem_bw = round_base(raw.gpu_mem_bw, base=50)
    # raw.disk_bw = raw.disk_bw.round(-2)
    # raw.dlperf = round_base(raw.dlperf, base=50)
    # raw.score = raw.score.round()

    # End Of Day data
    raw.end_date = round_day(raw.end_date)

    # Reliability * 1e4
    raw.reliability = raw.reliability * 1e4

    # All costs * 1000 as integer
    for col in const.COST_COLS:
        raw[col] = raw[col] * 1e3

    # raw.dph_base = raw.dph_base * 1e3
    # raw.storage_cost = raw.storage_cost * 1e3
    # raw.inet_up_cost = raw.inet_up_cost * 1e3
    # raw.inet_down_cost = raw.inet_down_cost * 1e3
    # raw.min_bid = raw.min_bid * 1e3
    # raw.credit_discount_max = raw.credit_discount_max * 1e3

    _conv_to_int(raw, const.INT_COLS)
    _conv_to_str(raw, const.STR_COLS)

    # Drop
    to_drop = [c for c in raw if c not in const.KEEP_COLS]
    raw.drop(columns=to_drop, inplace=True, errors='ignore')
    return raw

