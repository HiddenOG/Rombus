"""Reshaping an uploaded table before any work starts: the column names are on a lower row.

Spreadsheets often open with a title, a date or a logo row, and the real column names sit a few rows
down. pandas reads the first row as the names, so the table arrives with names like "Unnamed: 1" and
the real names as its first data row. Instead of asking for a new upload, the names are re-taken
from the row the person points at, counted the way Excel shows rows (row 1 = the first line).
"""

from typing import List

import pandas as pd

MAX_HEADER_ROW = 50


def header_from_row(df: pd.DataFrame, row: int) -> pd.DataFrame:
    """The table with its column names taken from spreadsheet row `row` (1 = as uploaded); the rows
    above it are dropped, and columns that now hold only numbers become numbers again."""
    if not 1 <= row <= MAX_HEADER_ROW:
        raise ValueError(f"Choose a row between 1 and {MAX_HEADER_ROW}.")
    if row == 1:
        return df
    position = row - 2  # row 1 became the names; the data's first row is spreadsheet row 2
    if position >= len(df):
        raise ValueError(f"The table only has {len(df) + 1} rows.")
    names = _clean_names(df.iloc[position].tolist())
    out = df.iloc[position + 1:].reset_index(drop=True)
    out.columns = names
    out = out.dropna(axis=1, how="all")  # columns that were only there because of the title rows
    if out.empty:
        raise ValueError(f"There's no data below row {row}.")
    return _numbers_where_possible(out)


def _clean_names(values: List) -> List[str]:
    names, seen = [], {}
    for i, value in enumerate(values, 1):
        name = "" if value is None or (isinstance(value, float) and value != value) else str(value).strip()
        if isinstance(value, float) and value.is_integer():
            name = str(int(value))  # 2024.0 -> "2024"
        name = name or f"column_{i}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}_{count + 1}")
    return names


def _numbers_where_possible(df: pd.DataFrame) -> pd.DataFrame:
    """A column was text only because the title rows were in it: if every value is a number now, make
    it a number column again (whole numbers stay whole)."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        numbers = pd.to_numeric(values.astype(str).str.replace(",", "").str.strip(), errors="coerce")
        if numbers.notna().all():
            converted = pd.to_numeric(df[col].astype(object).where(df[col].notna()).map(
                lambda v: v if v is None or (isinstance(v, float) and v != v) else str(v).replace(",", "").strip()),
                errors="coerce")
            if (converted.dropna() % 1 == 0).all():
                converted = converted.astype("Int64") if converted.isna().any() else converted.astype("int64")
            df[col] = converted
    return df
