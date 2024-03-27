import logging

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_integer_dtype, is_float_dtype, is_string_dtype

from src import const
from src.utils import round_day, round_base, custom_round, np_group_by


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


def _get_isp(raw: pd.DataFrame):
    return raw.location.apply(lambda x: x.get('isp', None) if not pd.isna(x) else None)


def _get_country(raw: pd.DataFrame):
    country_code = raw.geolocation.str.split(',').apply(lambda x: x[-1] if x else None).str.strip().replace('Sweden', 'SE')
    return country_code

    # raw.loc[raw.location.isna(), 'location'] = None
    # country_loc = raw.location.apply(lambda x: x['country'] if x else None)
    # raw['country'] = country_loc
    # mask = raw.country.isna()
    # raw.loc[mask, 'country'] = country_code[mask]


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
            raw.index = np.round(raw.index).astype(int)
        if col not in raw:
            continue
        raw[col] = raw[col].round().astype(int)


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


def _get_slices(raw: pd.DataFrame):
    """
    Given the raw data of offers, sorted by machine_id and num_gpus, get
    the `slice_idx` depicting the start of each machine group and `group_count`
    indicating the total number of offers per each machine.
    @param raw: raw offers data
    @return: slice_idx per machine group, group_count per machine_group
    """
    # raw.sort_values(by=['machine_id', 'num_gpus'], inplace=True)
    # lexidx = np.lexsort((raw.num_gpus.values, raw.machine_id.values))
    # raw = raw.iloc[lexidx]

    machine_ids, num_gpus = raw.machine_id.values, raw.num_gpus.values

    slice_idx = np.diff(machine_ids).nonzero()[0] + 1
    slice_idx = np.hstack([0, slice_idx])    # insert zero at the beginning of slice_idx

    group_count = np.diff(np.concatenate([slice_idx, [machine_ids.size]]))  # groupby('machine_id').count()
    return slice_idx, group_count


def _gpu_min_chunk(raw, slice_idx, group_count):
    """
    @param raw: (pd.DataFrame) input raw of offers
    @param slice_idx: (np.array) array of indices indicating start of each machine subgroup
    @param group_count: (np.array) number of total offers in machine subgroup
    @return: (np.array) array of min_chunk per each offer
    """
    machine_ids, num_gpus = raw.machine_id.values, raw.num_gpus.values

    min_chunk_machines = num_gpus[slice_idx]                  # min_chunk per machine
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
    just the last in the group, as they are sorted
    slice_idx: first idx in the group
    slice_idx - 1: last idx in the previous group
    """
    machine_ids = raw.machine_id.values
    max_chunk_idx = np.append(slice_idx[1:], machine_ids.size) - 1  # skip first zero, add last index
    machines = raw.iloc[max_chunk_idx].copy()
    return machines


def _add_min_chunk(machines, slice_idx, min_chunk):
    if 'min_chunk' not in machines:
        machines['min_chunk'] = min_chunk[slice_idx]


def _add_num_gpus_rented(machines, offers):
    if 'num_gpus_rented' not in machines:
        offers['num_gpus_rented'] = offers.rented * offers.num_gpus
        machines['num_gpus_rented'] = np_group_by(offers, 'num_gpus_rented', np.add)


def _filter_sum_gpus(machines, offers):
    # filter machines where total number of gpu in offers
    # doesn't sum up to num_gpus in machine
    num_gpus = np_group_by(offers, 'num_gpus', np.add)
    mask = machines.num_gpus == num_gpus
    to_keep = machines.machine_id[mask]
    return machines[mask].copy(), offers[offers.machine_id.isin(to_keep)].copy()


def preprocess(raw: pd.DataFrame):
    _rename_cols(raw)

    # raw['isp'] = _get_isp(raw)
    raw['country'] = _get_country(raw)
    raw.verification = raw.verification.map(VERIFIED_ENUM)

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


def split_raw(raw: pd.DataFrame):
    raw.sort_values(by=['machine_id', 'num_gpus'], inplace=True)

    slice_idx, group_count = _get_slices(raw)
    min_chunk = _gpu_min_chunk(raw, slice_idx, group_count)

    offers = _get_offers(raw, min_chunk)
    machines = _get_machines(raw, slice_idx)

    _add_min_chunk(machines, slice_idx, min_chunk)
    _add_num_gpus_rented(machines, offers)

    # check if sum of min_chunk constitute whole machine
    machines, offers = _filter_sum_gpus(machines, offers)

    # drop offer-specific cols from machines
    machines.drop(columns=['id', 'rented'], inplace=True, errors='ignore')

    return machines, offers
