import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_integer_dtype, is_float_dtype, is_string_dtype

from src import tables
# from src.tables import INT32_COLS, STR_COLS, DROP_COLS
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


def _get_slices(raw: pd.DataFrame):
    # raw.sort_values(by=['machine_id', 'num_gpus'], inplace=True)
    # lexidx = np.lexsort((raw.num_gpus.values, raw.machine_id.values))
    # raw = raw.iloc[lexidx]

    machine_ids, num_gpus = raw.machine_id.values, raw.num_gpus.values

    slice_idx = np.diff(machine_ids).nonzero()[0] + 1
    slice_idx = np.hstack([0, slice_idx])    # insert zero at the beginning of slice_idx

    group_count = np.diff(np.concatenate([slice_idx, [machine_ids.size]]))  # groupby('machine_id').count()
    return slice_idx, group_count


def _gpu_min_chunk(raw, slice_idx, group_count):
    machine_ids, num_gpus = raw.machine_id.values, raw.num_gpus.values

    min_chunk_machines = num_gpus[slice_idx]
    min_chunk = np.repeat(min_chunk_machines, group_count)    # expanded min_chunk for each offer
    min_chunk_count = np.add.reduceat((min_chunk == num_gpus), slice_idx)

    # correction for the case where whole machine size is not a multiple of min_chunk
    # in this case, there is always a single remainder chunk which is smaller than actual min_chunk
    # examples: [1 2 2 2 3 4 7], actual min_chunk is 2
    #           [3 4 7], actual min_chunk is 4

    idx = (min_chunk_count == 1) & (group_count >= 3)

    # get second minimum, correct for last index
    second_min_idx = slice_idx + 1
    if second_min_idx[-1] == machine_ids.size:
        second_min_idx[-1] -= 1

    min_chunk_machines[idx] = num_gpus[second_min_idx][idx]

    # update min_chunk
    min_chunk = np.repeat(min_chunk_machines, group_count)
    return min_chunk


def _get_offers(raw, min_chunk):
    """
    filter offers with num_gpus smaller or eq to min_chunk
    """
    return raw[raw.num_gpus <= min_chunk].copy()


def _get_machines(raw, slice_idx):
    """
    machines = offers with max gpu
    """
    machine_ids = raw.machine_id.values
    max_chunk_idx = np.append(slice_idx[1:], machine_ids.size) - 1
    machines = raw.iloc[max_chunk_idx].copy()
    return machines


def preprocess(raw: pd.DataFrame):
    _add_country(raw)
    _fillna(raw)
    _rename_cols(raw)

    # Hardware
    raw.bw_nvlink = raw.bw_nvlink.round(-1)
    raw.cpu_ram = (raw.cpu_ram / 1024).round()  # RAM in Gb
    # raw['cpu_ram_rnd'] = _round_ram(raw.cpu_ram)
    # raw['disk_space_rnd'] = raw.disk_space.round(-2).replace(0, 100)
    raw.pcie_bw = (raw.pcie_bw * 10).round()
    raw.gpu_mem_bw = raw.gpu_mem_bw.round(-1)
    raw.disk_bw = raw.disk_bw.round(-2)

    # scores
    raw.dlperf = raw.dlperf.round()
    raw.score = raw.score.round()

    # inet
    raw.inet_down = raw.inet_down.round(-1)
    raw.inet_up = raw.inet_up.round(-1)

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

    _conv_to_int(raw, tables.INT32_COLS)
    _conv_to_str(raw, tables.STR_COLS)

    # Drop
    raw.drop(columns=tables.DROP_COLS, inplace=True,  errors='ignore')


def split_raw(raw: pd.DataFrame):
    raw.sort_values(by=['machine_id', 'num_gpus'], inplace=True)

    slice_idx, group_count = _get_slices(raw)
    min_chunk = _gpu_min_chunk(raw, slice_idx, group_count)

    offers = _get_offers(raw, min_chunk)
    machines = _get_machines(raw, slice_idx)
    return machines, offers

