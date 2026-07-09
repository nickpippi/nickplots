"""Data loading, type inference and filtering. UI- and backend-agnostic."""
from __future__ import annotations
import pandas as pd


# Type categories the rest of the app understands (used to validate mappings).
NUMBER, CATEGORY, DATETIME = "number", "category", "datetime"


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load a CSV, auto-detecting encoding and separator.
    Tries UTF-8 then latin1; sep=None + python engine auto-detects , ; tab |."""
    sep = kwargs.pop("sep", None)
    engine = kwargs.pop("engine", "python")
    encodings = kwargs.pop("encoding", None)
    if encodings is None:
        encodings = ["utf-8", "latin1", "utf-8-sig", "cp1252"]
    elif isinstance(encodings, str):
        encodings = [encodings]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=sep, engine=engine,
                             encoding=enc, **kwargs)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception:
            raise
    raise ValueError(f"Could not decode the file. Try saving it as UTF-8. "
                     f"Detail: {last_err}")


def column_kinds(df: pd.DataFrame) -> dict[str, str]:
    """Map each column to NUMBER / CATEGORY / DATETIME."""
    kinds: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            kinds[col] = NUMBER
        elif pd.api.types.is_datetime64_any_dtype(s):
            kinds[col] = DATETIME
        else:
            kinds[col] = CATEGORY
    return kinds


def columns_by_kind(df: pd.DataFrame, kind: str) -> list[str]:
    return [c for c, k in column_kinds(df).items() if k == kind]


def to_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a numeric column to text so it counts as CATEGORY (e.g. a numeric
    track_id used as hue becomes discrete colors instead of a continuous gradient)."""
    df = df.copy()
    df[col] = df[col].astype(str)
    return df


def apply_filter(df: pd.DataFrame, expr: str) -> tuple[pd.DataFrame, str | None]:
    """Apply pandas query(). Returns (filtered_df, error_message), never raises."""
    expr = (expr or "").strip()
    if not expr:
        return df, None
    try:
        return df.query(expr), None
    except Exception as e:  # SyntaxError, UndefinedVariableError, etc.
        return df, f"Invalid filter expression: {e}"


def export_filtered(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


# ============================ Excel + reshape ============================ #
def excel_sheets(path: str) -> list[str]:
    return pd.ExcelFile(path).sheet_names


def load_excel(path: str, sheet=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def reshape_melt(df, id_vars, value_vars, var_name="variable", value_name="value"):
    """Wide -> long. Collapse several measure columns into one."""
    return df.melt(id_vars=id_vars or None, value_vars=value_vars or None,
                   var_name=var_name, value_name=value_name)


def reshape_pivot(df, index, columns, values, aggfunc="mean"):
    """Long -> wide. Spread one column's categories into several columns."""
    out = df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
    return out.reset_index()
