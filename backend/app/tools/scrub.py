"""Scrub tools — data cleaning, imputation, encoding, and transformation."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from app.tools import register_tool, validate_column
from app.session import DataSession


@register_tool("remove_columns")
def remove_columns(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Remove specified columns from the dataset."""
    cols = args.get("columns", [])
    if not cols:
        return {"success": False, "message": "No columns specified."}

    session.save_snapshot()
    valid_cols = [c for c in cols if c in session.df.columns]
    invalid_cols = [c for c in cols if c not in session.df.columns]

    if valid_cols:
        session.df.drop(columns=valid_cols, inplace=True)

    msg = f"Removed columns: {valid_cols}."
    if invalid_cols:
        msg += f" Columns not found (skipped): {invalid_cols}."

    return {"success": True, "message": msg}


@register_tool("check_duplicates")
def check_duplicates(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Check for duplicate rows."""
    dup_count = int(session.df.duplicated().sum())
    return {
        "success": True,
        "message": f"Found {dup_count} duplicate rows out of {len(session.df)} total rows.",
        "evidence": {"duplicate_count": dup_count, "total_rows": len(session.df)},
    }


@register_tool("remove_duplicates")
def remove_duplicates(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Remove duplicate rows."""
    session.save_snapshot()
    before = len(session.df)
    session.df.drop_duplicates(inplace=True)
    session.df.reset_index(drop=True, inplace=True)
    after = len(session.df)
    return {
        "success": True,
        "message": f"Removed {before - after} duplicate rows. Row count: {before} → {after}.",
    }


@register_tool("clean_numeric_string")
def clean_numeric_string(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a column by removing non-numeric characters and converting to numeric."""
    col = args.get("column")
    if not col:
        return {"success": False, "message": "Missing 'column' argument."}
    validate_column(session, col)

    session.save_snapshot()
    before_dtype = str(session.df[col].dtype)
    session.df[col] = pd.to_numeric(
        session.df[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True),
        errors='coerce'
    )
    after_nulls = int(session.df[col].isnull().sum())

    return {
        "success": True,
        "message": f"Cleaned '{col}': {before_dtype} → float64. Nulls after conversion: {after_nulls}.",
    }


@register_tool("standardize_text")
def standardize_text(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize text column: lowercase, strip whitespace."""
    col = args.get("column")
    if not col:
        return {"success": False, "message": "Missing 'column' argument."}
    validate_column(session, col)

    session.save_snapshot()
    session.df[col] = session.df[col].astype(str).str.lower().str.strip()
    return {
        "success": True,
        "message": f"Standardized text in '{col}' (lowercased, stripped whitespace).",
    }


@register_tool("impute_missing")
def impute_missing(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Impute missing values in a column."""
    col = args.get("column")
    strategy = args.get("strategy", "mean")

    if not col:
        return {"success": False, "message": "Missing 'column' argument."}
    validate_column(session, col)

    session.save_snapshot()
    missing_before = int(session.df[col].isnull().sum())

    if missing_before == 0:
        return {"success": True, "message": f"Column '{col}' has no missing values. Nothing to impute."}

    if strategy == "mean":
        if not pd.api.types.is_numeric_dtype(session.df[col]):
            return {"success": False, "message": f"Cannot use 'mean' strategy on non-numeric column '{col}'. Use 'mode' instead."}
        val = session.df[col].mean()
    elif strategy == "median":
        if not pd.api.types.is_numeric_dtype(session.df[col]):
            return {"success": False, "message": f"Cannot use 'median' strategy on non-numeric column '{col}'. Use 'mode' instead."}
        val = session.df[col].median()
    elif strategy == "mode":
        mode_res = session.df[col].mode()
        val = mode_res[0] if not mode_res.empty else 0
    else:
        return {"success": False, "message": f"Unknown strategy '{strategy}'. Use 'mean', 'median', or 'mode'."}

    session.df[col] = session.df[col].fillna(val)

    val_str = f"{val:.4f}" if isinstance(val, float) else str(val)

    return {
        "success": True,
        "message": f"Imputed {missing_before} missing values in '{col}' using {strategy} (value: {val_str}).",
    }


@register_tool("encode_categorical")
def encode_categorical(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Encode a categorical column using label or one-hot encoding."""
    col = args.get("column")
    method = args.get("method", "label")

    if not col:
        return {"success": False, "message": "Missing 'column' argument."}
    validate_column(session, col)

    session.save_snapshot()

    if method == "label":
        le = LabelEncoder()
        session.df[col] = le.fit_transform(session.df[col].astype(str))
        session.label_encoders[col] = le
        msg = f"Label-encoded '{col}'. Classes: {list(le.classes_[:10])}{'...' if len(le.classes_) > 10 else ''}."
    elif method == "onehot":
        dummies = pd.get_dummies(session.df[col], prefix=col)
        session.df = pd.concat([session.df, dummies], axis=1).drop(columns=[col])
        msg = f"One-hot encoded '{col}' into {len(dummies.columns)} new columns."
    else:
        return {"success": False, "message": f"Unknown encoding method '{method}'. Use 'label' or 'onehot'."}

    return {"success": True, "message": msg}


@register_tool("scale_features")
def scale_features(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Scale numeric features using StandardScaler."""
    cols = args.get("columns", [])

    session.save_snapshot()

    if not cols:
        cols = session.df.select_dtypes(include=[np.number]).columns.tolist()
        if session.target_column and session.target_column in cols:
            cols.remove(session.target_column)

    if not cols:
        return {"success": False, "message": "No numeric columns available to scale."}

    # Validate all columns exist
    for c in cols:
        validate_column(session, c)

    scaler = StandardScaler()
    session.df[cols] = scaler.fit_transform(session.df[cols])
    session.scaler = scaler

    return {
        "success": True,
        "message": f"Scaled {len(cols)} features using StandardScaler: {cols}.",
    }


@register_tool("log_transform")
def log_transform(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply log1p transformation to reduce skewness."""
    col = args.get("column")
    if not col:
        return {"success": False, "message": "Missing 'column' argument."}
    validate_column(session, col)

    session.save_snapshot()

    if not pd.api.types.is_numeric_dtype(session.df[col]):
        return {"success": False, "message": f"Column '{col}' is not numeric. Cannot apply log transform."}

    skew_before = float(session.df[col].skew())
    session.df[col] = np.log1p(session.df[col])
    skew_after = float(session.df[col].skew())

    return {
        "success": True,
        "message": f"Log-transformed '{col}'. Skewness: {skew_before:.3f} → {skew_after:.3f}.",
    }


# ── Batch tools (apply one change to ALL columns at once) ────────────────────


@register_tool("impute_all_missing")
def impute_all_missing(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Impute missing values across ALL columns in one shot.
    
    Uses mean for numeric columns and mode for categorical columns.
    Optionally accepts 'strategy' to override (mean, median, mode).
    """
    strategy = args.get("strategy")  # None = auto-select per column
    session.save_snapshot()

    cols_with_missing = [c for c in session.df.columns if session.df[c].isnull().any()]
    if not cols_with_missing:
        return {"success": True, "message": "No missing values found in any column. Nothing to impute."}

    results = []
    for col in cols_with_missing:
        missing_count = int(session.df[col].isnull().sum())
        is_numeric = pd.api.types.is_numeric_dtype(session.df[col])

        # Determine strategy for this column
        col_strategy = strategy
        if col_strategy is None:
            col_strategy = "mean" if is_numeric else "mode"
        
        # Guard: can't use mean/median on non-numeric
        if col_strategy in ("mean", "median") and not is_numeric:
            col_strategy = "mode"

        if col_strategy == "mean":
            val = session.df[col].mean()
        elif col_strategy == "median":
            val = session.df[col].median()
        else:  # mode
            mode_res = session.df[col].mode()
            val = mode_res[0] if not mode_res.empty else 0

        session.df[col] = session.df[col].fillna(val)
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        results.append(f"  • {col}: {missing_count} values → {col_strategy} ({val_str})")

    return {
        "success": True,
        "message": f"Imputed missing values in {len(results)} columns:\n" + "\n".join(results),
    }


@register_tool("encode_all_categorical")
def encode_all_categorical(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Encode ALL remaining categorical (object/category) columns at once.
    
    Default method is 'label'. Accepts 'method' arg: 'label' or 'onehot'.
    Skips the target column to avoid data leakage.
    """
    method = args.get("method", "label")

    cat_cols = session.df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Skip target column
    if session.target_column and session.target_column in cat_cols:
        cat_cols.remove(session.target_column)

    if not cat_cols:
        return {"success": True, "message": "No categorical columns to encode (all columns are already numeric)."}

    session.save_snapshot()
    results = []

    for col in cat_cols:
        if method == "label":
            le = LabelEncoder()
            session.df[col] = le.fit_transform(session.df[col].astype(str))
            session.label_encoders[col] = le
            results.append(f"  • {col}: label-encoded ({len(le.classes_)} classes)")
        elif method == "onehot":
            dummies = pd.get_dummies(session.df[col], prefix=col)
            session.df = pd.concat([session.df, dummies], axis=1).drop(columns=[col])
            results.append(f"  • {col}: one-hot → {len(dummies.columns)} new columns")
        else:
            return {"success": False, "message": f"Unknown encoding method '{method}'. Use 'label' or 'onehot'."}

    return {
        "success": True,
        "message": f"Encoded {len(results)} categorical columns ({method}):\n" + "\n".join(results),
    }


@register_tool("scale_all_features")
def scale_all_features(session: DataSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Scale ALL numeric features at once using StandardScaler.
    
    Automatically excludes the target column.
    """
    session.save_snapshot()

    cols = session.df.select_dtypes(include=[np.number]).columns.tolist()
    if session.target_column and session.target_column in cols:
        cols.remove(session.target_column)

    if not cols:
        return {"success": False, "message": "No numeric columns available to scale."}

    scaler = StandardScaler()
    session.df[cols] = scaler.fit_transform(session.df[cols])
    session.scaler = scaler

    return {
        "success": True,
        "message": f"Scaled {len(cols)} numeric features using StandardScaler:\n  " + ", ".join(cols),
    }

