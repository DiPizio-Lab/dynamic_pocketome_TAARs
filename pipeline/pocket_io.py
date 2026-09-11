"""Shared CSV/Parquet I/O for the Step 2 pipeline (pocket_dataframes.py,
global_id_and_comparison.py, and the taar_paper_figures/ readers downstream of them).

The big per-pocket tables (all_pockets, pocket_summary, pocket_comparison_table) are written
as BOTH a .csv and a .parquet under the same saving_loc; every reader goes through load_table()
below instead of a hardcoded pd.read_csv.
"""
import os
import pandas as pd


def save_table(df, saving_loc, name, formats=('csv', 'parquet'), index=False):
    """Writes {saving_loc}/{name}.csv and/or {name}.parquet. Returns the list of paths written."""
    os.makedirs(saving_loc, exist_ok=True)
    paths = []
    if 'csv' in formats:
        path = os.path.join(saving_loc, f'{name}.csv')
        df.to_csv(path, index=index)
        paths.append(path)
    if 'parquet' in formats:
        path = os.path.join(saving_loc, f'{name}.parquet')
        df.to_parquet(path, index=index)
        paths.append(path)
    return paths


def load_table(saving_loc, name, prefer='parquet', **read_kwargs):
    """Loads {saving_loc}/{name}.parquet if present (default preference -- smaller, faster,
    keeps dtypes), else falls back to {name}.csv. Pass prefer='csv' to flip the preference,
    e.g. when you need a csv-only read_csv kwarg such as skiprows/usecols/dtype.

    read_kwargs are forwarded to whichever pandas reader ends up being used.
    """
    order = ('parquet', 'csv') if prefer == 'parquet' else ('csv', 'parquet')
    for fmt in order:
        path = os.path.join(saving_loc, f'{name}.{fmt}')
        if os.path.exists(path):
            return pd.read_parquet(path, **read_kwargs) if fmt == 'parquet' else pd.read_csv(path, **read_kwargs)
    raise FileNotFoundError(f"Neither {name}.parquet nor {name}.csv found under {saving_loc}")


def table_path(saving_loc, name, fmt='parquet'):
    """Path a table of this name WOULD have in this format, without checking it exists --
    for callers that need to hand a path to something else (e.g. a chunked csv reader)."""
    return os.path.join(saving_loc, f'{name}.{fmt}')
