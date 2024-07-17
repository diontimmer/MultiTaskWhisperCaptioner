from typing import Callable, Union, Tuple, List, Dict


def set_cols(
    colname: Union[str, Tuple[str, ...], List[str]], func: Callable
) -> Callable:
    """Set column(s) in a dictionary using a provided function."""

    def _func(row: dict) -> dict:
        row = row.copy()
        if isinstance(colname, str):
            row[colname] = func(row)
        else:
            for col, val in zip(colname, func(row)):
                row[col] = val
        return row

    return _func


def del_cols(*args: Union[str, Tuple[str, ...], List[str]]) -> Callable:
    """Delete column(s) from a dictionary."""
    colnames = [
        col
        for arg in args
        for col in (arg if isinstance(arg, (tuple, list)) else [arg])
    ]

    def _func(row: dict) -> dict:
        return {col: val for col, val in row.items() if col not in colnames}

    return _func


def rename_col(mapper: Dict[str, str]) -> Callable:
    """Rename column(s) in a dictionary based on a provided mapping."""

    def _func(row: dict) -> dict:
        row = row.copy()
        for old_colname, new_colname in mapper.items():
            row[new_colname] = row.pop(old_colname)
        return row

    return _func


def explode_col(colnames: List[str], new_name: str, name_keep_in: str) -> Callable:
    """Explode columns into multiple rows."""

    def _func(row: dict) -> List[dict]:
        row = row.copy()
        caption_cols = {colname: row.pop(colname) for colname in colnames}
        return [
            {**row, name_keep_in: colname, new_name: caption}
            for colname, caption in caption_cols.items()
        ]

    return _func
