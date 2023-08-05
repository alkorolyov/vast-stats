import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_integer_dtype, is_float_dtype, is_string_dtype

from src.tables import INT32_COLS, STR_COLS, DROP_COLS
from src.utils import round_day

VERIFIED_ENUM = {
    'unverified': 0,
    'verified': 1,
    'deverified': 2,
    'de-verified': 3
}

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


def _add_country(raw: pd.DataFrame):
    """ `location` is more accurate than `geolocation`:
        set country based on `location` then for missing add from `geolocation`
    """
    country_geoloc = raw.geolocation.str.split(',').apply(lambda x: x[-1] if x else None).str.strip().replace('Sweden', 'SE')
    raw['country'] = country_geoloc

    # raw.loc[raw.location.isna(), 'location'] = None
    # country_loc = raw.location.apply(lambda x: x['country'] if x else None)
    # raw['country'] = country_loc
    # mask = raw.country.isna()
    # raw.loc[mask, 'country'] = country_geoloc[mask]


def _fillna(raw: pd.DataFrame):
    """ Fill NA's:  0 for numerical types
                    '' for string types
    """

    for col in raw.columns:
        dtype = raw[col].dtype
        if is_integer_dtype(dtype) or is_float_dtype(dtype):
            raw[col].fillna(0, inplace=True)
        elif is_string_dtype(dtype):
            raw[col].fillna('', inplace=True)


def _conv_to_int(raw: pd.DataFrame, cols: list):
    for col in cols:
        if raw.index.name == col:
            raw.index = raw.index.astype(int)
        if col not in raw:
            continue
        if is_float_dtype(raw[col]):
            raw[col] = raw[col].round()
        raw[col] = raw[col].astype(int)


def _conv_to_str(raw: pd.DataFrame, cols: list):
    for col in cols:
        if raw.index.name == col:
            raw.index = raw.index.astype(str)
        if col not in raw:
            continue
        raw[col] = raw[col].astype(str)


def _rename_cols(raw):
    if 'rentable' in raw:
        raw.rentable = ~raw.rentable
        raw.rename(columns={'rentable': 'rented'}, inplace=True)

    if 'reliability2' in raw:
        raw.rename(columns={'reliability2': 'reliability'}, inplace=True)


def preprocess(raw: pd.DataFrame):
    _add_country(raw)
    _fillna(raw)
    _rename_cols(raw)

    # Hardware
    raw.bw_nvlink = raw.bw_nvlink.round(-1)
    raw.cpu_ram = (raw.cpu_ram / 1024).round() # RAM in Gb
    # raw['cpu_ram_rnd'] = _round_ram(raw.cpu_ram)
    # raw['disk_space_rnd'] = raw.disk_space.round(-2).replace(0, 100)

    raw.dlperf = (raw.dlperf * 1e2).round()
    raw.score = (raw.score * 1e2).round()
    raw.pcie_bw = (raw.pcie_bw * 10).round()

    # End Of Day data
    raw.verification = raw.verification.map(VERIFIED_ENUM)
    raw.cuda_max_good = raw.cuda_max_good.astype(str)
    raw.end_date = round_day(raw.end_date)

    # Reliability * 1e4
    raw.reliability = (raw.reliability * 1e4).round()

    # All costs * 1e3 as integer
    raw.dph_base = (raw.dph_base * 1e3).round()
    raw.storage_cost = (raw.storage_cost * 1e3).round()
    raw.inet_up_cost = (raw.inet_up_cost * 1e3).round()
    raw.inet_down_cost = (raw.inet_down_cost * 1e3).round()
    raw.min_bid = (raw.min_bid * 1e3).round()
    raw.credit_discount_max = (raw.credit_discount_max * 1e3).round()

    _conv_to_int(raw, INT32_COLS)
    _conv_to_str(raw, STR_COLS)

    # Drop
    raw.drop(columns=DROP_COLS, inplace=True,  errors='ignore')


