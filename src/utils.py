import numpy as np
import pandas as pd


def time_ms(time_sec: float):
    return int(time_sec * 1000)


def time_utc_now() -> pd.Timestamp:
    return pd.Timestamp.utcnow().round(freq='S').tz_convert(None)


def round_day(ts: pd.Series):
    return pd.to_datetime(ts*10**9).dt.round(freq='D').values.astype('int64') // 10**9


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


def get_tables(conn) -> list:
    res = conn.execute(f'''
    SELECT
        name
    FROM
           (SELECT * FROM sqlite_master UNION ALL
            SELECT * FROM sqlite_temp_master)
    WHERE
        type ='table' AND
        name NOT LIKE 'sqlite_%';
    ''').fetchall()
    return [x[0] for x in res]


def get_tbl_info(name, conn) -> pd.DataFrame:
    df = pd.DataFrame(conn.execute(f'PRAGMA table_info({name})').fetchall(),
                      columns=['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk'])
    return df.set_index('cid')


def table_to_df(name, conn) -> pd.DataFrame:
    cols = get_tbl_info(name, conn)['name']
    return pd.DataFrame(conn.execute(f'SELECT * FROM {name}').fetchall(),
                        columns=cols)


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

    slice_idx = np.r_[0, np.flatnonzero(np.diff(machine_ids)) + 1]
    return machine_ids[slice_idx], ufunc.reduceat(vals, slice_idx)


def np_min_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    lexidx = np.lexsort((raw.num_gpus.values, raw.machine_id.values))

    machine_ids, num_gpus = raw.machine_id.values[lexidx], raw.num_gpus.values[lexidx]
    slice_idx = np.r_[0, np.flatnonzero(np.diff(machine_ids)) + 1]

    group_count = np.diff(np.concatenate([slice_idx, [len(machine_ids)]])) # groupby('machine_id').count()

    min_chunk = num_gpus[slice_idx]
    min_chunk_ex = np.repeat(min_chunk, group_count) # expanded min_chunk for each machine_id
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
