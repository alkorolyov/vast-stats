import numpy as np
import pandas as pd

from src.utils import np_group_by


def _get_slices(raw: pd.DataFrame):
    """
    Given the raw data of offers, sorted by machine_id and num_gpus, pop
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

    # pop second minimum, correct for last index
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
