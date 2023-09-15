import gc
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_integer_dtype, is_float_dtype, is_numeric_dtype


def time_ms(time_sec: float):
    return int(time_sec * 1000)


def time_utc_now() -> pd.Timestamp:
    return pd.Timestamp.utcnow().round(freq='S').tz_convert(None)


def round_day(ts: pd.Series):
    return pd.to_datetime(ts*10**9).dt.round(freq='D').values.astype('int64') // 10**9


def round_base(x: pd.Series, base=5):
    return (base * (x.astype(float)/base).round()).astype(int)


def custom_round(x: pd.Series):
    """
    Custom round function for disk_bw:
    x < 1000: round to 100
    x in [1000, 4000]: round to 500
    x > 4000: round to 1000
    @param x:
    @return: rounded series
    """
    res = x.copy()
    mask = x < 1000
    # res[mask] = res[mask].round(-1)
    res[mask] = round_base(res[mask], base=50)

    mask = x >= 1000
    res[mask] = res[mask].round(-2)

    # mask = x.between(1000, 4000)
    # res[mask] = round_base(res[mask], 500)
    #
    # mask = x > 4000
    # res[mask] = res[mask].round(-3)

    return res

def is_sorted(arr):
    return np.all(arr[:-1] <= arr[1:])


def df_na_vals(df, return_empty=True):
    columns = df.columns
    N = max(len(c) for c in columns) + 5
    empty = []
    for col in columns:
        na_vals = df[col].isna()
        print(col.ljust(N), '->', ' '*(N//3), f'Missing values: {na_vals.sum()} ({na_vals.mean():.2%})')

        if na_vals.mean() > .99:
            empty.append(col)

    if return_empty:
        return empty


def np_argmax_reduceat(vals, slice_idx):
    n = vals.max() + 1
    idx_arr = np.zeros(vals.size, dtype=np.uint8)
    idx_arr[slice_idx] = 1
    offset = n * idx_arr.cumsum()
    argidx = np.argsort(vals + offset)
    last_grp_idx = np.append(slice_idx[1:], vals.size) - 1
    return argidx[last_grp_idx]


def np_group_by(raw: pd.DataFrame, value_col: str, ufunc):
    """
    Group by machine_id using numpy ufunc (np.maximum, np.add etc.)
    :param raw: dataframe containing multiple offers per machine_id
    :param value_col: target column name
    :param ufunc: numpy ufunc to apply after groupby
    :return: arrays of corresponding machine_id, ufunc values
    """

    machine_ids = raw.machine_id.values
    vals = raw[value_col].values
    if not is_sorted(machine_ids):
        idx = np.argsort(machine_ids)
        machine_ids = machine_ids[idx]
        vals = vals[idx]

    # slice_idx = np.r_[0, np.flatnonzero(np.diff(machine_ids)) + 1]
    slice_idx = np.diff(machine_ids).nonzero()[0] + 1
    slice_idx = np.hstack([0, slice_idx])    # insert zero at the beginning of arr
    return machine_ids[slice_idx], ufunc.reduceat(vals, slice_idx)


def np_min_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    lexidx = np.lexsort((raw.num_gpus.values, raw.machine_id.values))

    machine_ids, num_gpus = raw.machine_id.values[lexidx], raw.num_gpus.values[lexidx]
    slice_idx = np.diff(machine_ids).nonzero()[0] + 1
    slice_idx = np.hstack([0, slice_idx])    # insert zero at the beginning of slice_idx

    group_count = np.diff(np.concatenate([slice_idx, [machine_ids.size]]))  # groupby('machine_id').count()

    min_chunk = num_gpus[slice_idx]
    min_chunk_ex = np.repeat(min_chunk, group_count)    # expanded min_chunk for each machine_id
    min_chunk_count = np.add.reduceat((min_chunk_ex == num_gpus), slice_idx)

    # correction for the case where whole machine size is not a multiple of min_chunk
    # in this case, there is always a single remainder chunk which is smaller than actual min_chunk
    # examples: [1 2 2 2 3 4 7], actual min_chunk is 2
    #           [3 4 7], actual min_chunk is 4

    idx = (min_chunk_count == 1) & (group_count >= 3)

    # get second minimum, correct for last index
    second_min_idx = slice_idx + 1
    if second_min_idx[-1] == machine_ids.size:
        second_min_idx[-1] -= 1

    min_chunk[idx] = num_gpus[second_min_idx][idx]

    # update min_chunk_ex
    min_chunk_ex = np.repeat(min_chunk, group_count)

    return raw.loc[lexidx][num_gpus <= min_chunk_ex].copy()


def _pd_min_chunk(df: pd.DataFrame) -> pd.DataFrame:
    gpu_chunks = list(df.num_gpus)
    gpu_chunks.sort()
    min_chunk = min(gpu_chunks)

    # correction for the case where whole machine size is not a multiple of min_chunk
    # in this case, there is always a single remainder chunk which is smaller than actual min_chunk
    # examples: [1 2 2 2 3 4 7] [1 2 3], actual min_chunk is 2
    #           [3 4 7], actual min_chunk is 4
    if gpu_chunks.count(min_chunk) == 1 and len(gpu_chunks) >= 3:
        min_chunk = gpu_chunks[1]

    # filter undivided chunks
    undivided = df[df.num_gpus <= min_chunk]
    return undivided


def _is_float_to_int(arr) -> bool:
    return np.all(np.isclose(arr, np.round(arr)))


def check_if_integer(series) -> bool:
    if is_float_dtype(series.dtype):
        return _is_float_to_int(series.values)
    return False


def reduce_mem_usage(df, int_cast=True, obj_to_category=False, subset=None, verbose=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    :param df: dataframe to reduce (pd.DataFrame)
    :param int_cast: indicate if columns should be tried to be casted to int (bool)
    :param obj_to_category: convert non-datetime related objects to category dtype (bool)
    :param subset: subset of columns to analyse (list)
    :return: dataset with the column dtypes adjusted (pd.DataFrame)
    """
    if verbose:
        start_mem = df.memory_usage().sum() / 1024 ** 2
        gc.collect()
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    cols = subset if subset is not None else df.columns.tolist()

    for col in cols:
        col_type = df[col].dtype

        if is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            # test if column can be converted to an integer
            treat_as_int = is_integer_dtype(col_type)
            if int_cast and not treat_as_int:
                treat_as_int = check_if_integer(df[col])

            if treat_as_int:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # elif 'datetime' not in col_type.name and obj_to_category:
        #     df[col] = df[col].astype('category')

    if verbose:
        gc.collect()
        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.3f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

