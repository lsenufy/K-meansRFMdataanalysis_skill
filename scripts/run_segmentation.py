"""Configurable, read-only RFM/K-means runner. See references/data-and-runner.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from threadpoolctl import threadpool_limits


DEFAULTS = {
    "invalid_rows": "error", "drop_exact_duplicates": False,
    "monetary": "net", "negative_monetary": "error",
    "monetary_zero_tolerance": 1e-9,
    "order_date_policy": "error",
    "returns": {"amount_sign": "signed"},
    "purchases": {"allow_zero_amount": False},
    "rfm": {"score_levels": 5, "fm_rounding": "none",
            "scoring": {"r": {"method": "percentile"}, "f": {"method": "percentile"},
                        "m": {"method": "percentile"}},
            "segmentation": {"mode": "generic_matrix",
                             "cutoffs": {"low_max": 1 / 3, "mid_max": 2 / 3}}},
    "kmeans": {"candidates": list(range(2, 10)), "random_state": 42,
               "n_init": 20, "seeds": [0, 7, 21, 101], "scaler": "standard",
               "transforms": {"R_days": "auto", "Frequency": "auto", "Monetary": "auto"},
               "silhouette_sample_size": 3000},
    "baseline": {"budgets": [0.1, 0.2, 0.3], "budget_is_business": False,
                 "random_seed": 42},
    "cross": {"min_customers": 20},
}


def merged(base, extra):
    result = dict(base)
    for key, value in extra.items():
        result[key] = merged(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def day(value):
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        raise ValueError("Normalize timestamps to one business timezone before running.")
    if pd.isna(result):
        raise ValueError("Invalid date boundary.")
    return result.normalize()


def numeric(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def clean_money(series, config, audit, purpose):
    """Remove only documented floating point residuals, never material refunds."""
    tiny = series.abs().le(config["monetary_zero_tolerance"]) & series.ne(0)
    audit.setdefault("floating_zero_adjustments", {})[purpose] = {
        "count": int(tiny.sum()), "signed_amount": float(series.loc[tiny].sum())}
    return series.mask(tiny, 0.0)


def dates(series):
    result = pd.to_datetime(series, errors="coerce", format="mixed")
    if not pd.api.types.is_datetime64_dtype(result.dtype):
        raise ValueError("Dates must use a consistent, normalized business timezone.")
    return result


def read_input(spec, columns, base, audit, purpose, config):
    path = Path(spec["path"]).expanduser()
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, dtype=str, keep_default_na=False,
                          encoding=spec.get("encoding", "utf-8-sig"))
    elif path.suffix.lower() in {".xlsx", ".xlsm"}:
        raw = pd.read_excel(path, sheet_name=spec.get("sheet", 0), dtype=str,
                            keep_default_na=False)
    else:
        raise ValueError("Supported files: CSV, XLSX, XLSM with column headers.")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    info = {"purpose": purpose, "path": str(path), "rows": len(raw), "sha256": digest}
    audit["inputs"].append(info)
    missing = [source for source in columns.values() if source not in raw.columns]
    if missing:
        raise ValueError(f"Missing mapped columns: {missing}")
    if len(set(columns.values())) != len(columns):
        raise ValueError("A source column cannot map to multiple canonical fields.")
    if config["drop_exact_duplicates"]:
        original = len(raw)
        raw = raw.drop_duplicates()
        info["exact_duplicates_removed"] = original - len(raw)
    result = raw[list(columns.values())].rename(columns={v: k for k, v in columns.items()}).copy()
    for key in ["customer_id", "order_id"]:
        if key in result:
            result[key] = result[key].astype("string").str.strip()
    return result.reset_index(drop=True)


def filter_invalid(frame, bad, config, audit, purpose):
    count = int(bad.sum())
    audit.setdefault("invalid_rows", {})[purpose] = count
    if count and config["invalid_rows"] == "error":
        raise ValueError(f"{purpose}: {count} invalid required rows; fix source or explicitly select invalid_rows=exclude.")
    return frame.loc[~bad].copy()


def prepare_transactions(frame, config, audit, purpose):
    for key in ["customer_id", "order_id", "date"]:
        if key not in frame:
            raise ValueError(f"Transaction input requires {key}.")
    frame["date"] = dates(frame["date"])
    if "quantity" in frame:
        frame["quantity"] = numeric(frame["quantity"])
    if "amount" in frame:
        frame["amount"] = numeric(frame["amount"])
    elif {"quantity", "unit_price"}.issubset(frame.columns):
        frame["unit_price"] = numeric(frame["unit_price"])
        frame["amount"] = frame["quantity"] * frame["unit_price"]
    else:
        raise ValueError("Provide amount or both quantity and unit_price.")
    bad = (frame["customer_id"].fillna("").eq("") | frame["order_id"].fillna("").eq("") |
           frame["date"].isna() | ~np.isfinite(frame["amount"]))
    if "quantity" in frame:
        bad |= ~np.isfinite(frame["quantity"])
    frame = filter_invalid(frame, bad, config, audit, purpose)
    returns = config["returns"]
    mask = pd.Series(False, index=frame.index)
    if returns.get("prefix"):
        prefixes = returns["prefix"]
        prefixes = [prefixes] if isinstance(prefixes, str) else prefixes
        mask |= frame["order_id"].str.upper().str.startswith(tuple(str(x).upper() for x in prefixes))
    if "return_values" in returns:
        if "is_return" not in frame:
            raise ValueError("returns.return_values requires columns.is_return.")
        values = [str(x).strip().casefold() for x in returns["return_values"]]
        mask |= frame["is_return"].astype(str).str.strip().str.casefold().isin(values)
    if returns.get("signed_amount"):
        mask |= frame["amount"].lt(0)
    if returns.get("negative_quantity"):
        if "quantity" not in frame:
            raise ValueError("negative_quantity requires columns.quantity.")
        mask |= frame["quantity"].lt(0)
    frame["IsReturn"] = mask
    if returns["amount_sign"] == "unsigned":
        frame.loc[mask, "amount"] = -frame.loc[mask, "amount"].abs()
    elif (mask & frame["amount"].gt(0)).any():
        raise ValueError("Return rows contain positive signed amounts. Check returns.amount_sign.")
    positive_amount = frame["amount"].ge(0) if config["purchases"]["allow_zero_amount"] else frame["amount"].gt(0)
    frame["Purchase"] = ~mask & positive_amount
    if "quantity" in frame:
        frame["Purchase"] &= frame["quantity"].gt(0)
    frame["Value"] = frame["amount"] if config["monetary"] == "net" else frame["amount"].where(frame["Purchase"], 0)
    frame["BusinessDate"] = frame["date"].dt.normalize()
    order_dates = frame.loc[frame["Purchase"]].groupby(["customer_id", "order_id"])["BusinessDate"].nunique()
    cross_day = int(order_dates.gt(1).sum())
    policy = config["order_date_policy"]
    if cross_day and policy == "error":
        raise ValueError(f"{cross_day} purchase orders span multiple dates. Choose order_date_policy=first or last after reviewing the order lifecycle; monetary remains on transaction dates.")
    effective = frame.loc[frame["Purchase"]].groupby(["customer_id", "order_id"])["BusinessDate"].transform(
        "max" if policy == "last" else "min")
    frame["OrderDate"] = pd.NaT
    frame.loc[frame["Purchase"], "OrderDate"] = effective
    audit.setdefault("transactions", {})[purpose] = {
        "rows": len(frame), "return_rows": int(mask.sum()),
        "purchase_rows": int(frame["Purchase"].sum()),
        "cross_day_purchase_orders": cross_day, "order_date_policy": policy,
        "frequency_recency_date": "OrderDate", "monetary_date": "transaction date",
        "date_min": str(frame["date"].min()), "date_max": str(frame["date"].max())}
    return frame


def window(frame, start=None, end=None):
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= frame["date"].ge(day(start))
    if end:
        mask &= frame["date"].lt(day(end) + pd.Timedelta(days=1))
    if start and end and day(start) > day(end):
        raise ValueError("Window start must not follow window end.")
    return frame.loc[mask].copy()


def customer_rfm(frame, config, audit):
    if "customer_id" not in frame or "frequency" not in frame or "monetary" not in frame:
        raise ValueError("Customer input requires customer_id, frequency and monetary.")
    if config.get("analysis_start") or config.get("analysis_end"):
        raise ValueError("Customer aggregates must already reflect their observation window; do not filter them by last date.")
    frame["frequency"] = numeric(frame["frequency"])
    frame["monetary"] = numeric(frame["monetary"])
    recent_source = frame["recency_days"].copy() if "recency_days" in frame else frame.get("date")
    if "recency_days" in frame:
        frame["recency_days"] = numeric(frame["recency_days"])
    elif "date" in frame:
        frame["date"] = dates(frame["date"])
        if not config.get("reference_date") and frame["date"].notna().sum() == 0:
            raise ValueError("No valid purchase dates from which to infer reference_date.")
        ref = day(config["reference_date"]) if config.get("reference_date") else frame["date"].max().normalize() + pd.Timedelta(days=1)
        frame["recency_days"] = (ref - frame["date"].dt.normalize()).dt.days
        audit["reference_date"] = str(ref.date())
    else:
        raise ValueError("Customer input requires recency_days or date.")
    recent_nonblank = recent_source.astype("string").fillna("").str.strip().ne("")
    invalid_recency = ((frame["recency_days"].isna() & (frame["frequency"].gt(0) | recent_nonblank)) |
                       (frame["recency_days"].notna() & ~np.isfinite(frame["recency_days"])) |
                       frame["recency_days"].lt(0))
    bad = (frame["customer_id"].fillna("").eq("") | invalid_recency | ~np.isfinite(frame["frequency"]) |
           frame["frequency"].lt(0) | frame["frequency"].mod(1).ne(0) |
           ~np.isfinite(frame["monetary"]))
    frame = filter_invalid(frame, bad, config, audit, "customers")
    if frame["customer_id"].duplicated().any():
        raise ValueError("Customer input must contain exactly one row per customer_id.")
    return frame.rename(columns={"customer_id": "CustomerID", "recency_days": "R_days",
                                 "frequency": "Frequency", "monetary": "Monetary"})[
        ["CustomerID", "R_days", "Frequency", "Monetary"]].copy()


def transaction_rfm(frame, config, audit):
    if frame.empty:
        raise ValueError("No transactions in observation window.")
    ref = day(config["reference_date"]) if config.get("reference_date") else frame["BusinessDate"].max() + pd.Timedelta(days=1)
    if ref < frame["BusinessDate"].max():
        raise ValueError("reference_date is earlier than observation transactions.")
    audit["reference_date"] = str(ref.date())
    all_customers = frame.groupby("customer_id", sort=False)["Value"].sum().rename("Monetary").to_frame()
    orders = purchase_orders(frame, config.get("analysis_start"), config.get("analysis_end"))
    purchases = orders.groupby("customer_id", sort=False).agg(
        LastPurchaseDate=("OrderDate", "max"), Frequency=("order_id", "nunique"))
    result = all_customers.join(purchases).reset_index().rename(columns={"customer_id": "CustomerID"})
    result["Frequency"] = result["Frequency"].fillna(0).astype(int)
    result["R_days"] = (ref - result["LastPurchaseDate"].dt.normalize()).dt.days
    return result


def purchase_orders(frame, start=None, end=None):
    """One effective order date drives F, R and monthly F; money uses occurrence dates."""
    orders = frame.loc[frame["Purchase"], ["customer_id", "order_id", "OrderDate"]].drop_duplicates(
        ["customer_id", "order_id"])
    if start:
        orders = orders.loc[orders["OrderDate"].ge(day(start))]
    if end:
        orders = orders.loc[orders["OrderDate"].lt(day(end) + pd.Timedelta(days=1))]
    return orders


def percent_score(series, levels, ascending=True):
    percentile = (series.rank(method="min", ascending=ascending) - 1) / max(len(series) - 1, 1)
    return np.minimum(np.floor(percentile * levels).astype(int) + 1, levels)


def shopify_groups(r, fm):
    conditions = [(r <= 2) & (fm <= 2), (r <= 2) & (fm >= 3) & (fm <= 4),
                  (r <= 2) & (fm == 5), (r == 3) & (fm <= 2), (r == 3) & (fm == 3),
                  (r >= 3) & (r <= 4) & (fm >= 4), (r == 4) & (fm == 1),
                  (r >= 4) & (fm >= 2) & (fm <= 3), (r == 5) & (fm == 1),
                  (r == 5) & (fm >= 4)]
    labels = ["Dormant", "At risk", "Previously loyal", "Almost lost", "Needs attention",
              "Loyal", "Promising", "Active", "New", "Champions"]
    return np.select(conditions, labels, default="Unassigned")


def fm_score(f, m, rounding):
    value = (f + m) / 2
    return (np.floor(value) if rounding == "floor" else np.ceil(value) if rounding == "ceil"
            else np.floor(value + .5) if rounding == "round" else value)


def custom_rules(settings, levels):
    rules = settings.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("Custom segmentation requires a nonempty rules list.")
    allowed = {"label"} | {f"{key}_{bound}" for key in ["r", "f", "m", "fm"] for bound in ["min", "max"]}
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) - allowed or not isinstance(rule.get("label"), str) or not rule["label"].strip():
            raise ValueError("Custom rules require label and optional r/f/m/fm_min/max score bounds.")
        for key in ["r", "f", "m", "fm"]:
            lo, hi = rule.get(f"{key}_min", 1), rule.get(f"{key}_max", levels)
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or not (1 <= lo <= hi <= levels):
                raise ValueError(f"Invalid custom {key} bounds: expected 1 <= min <= max <= score_levels.")
    # Check the complete possible score domain, not just customers observed in this run.
    for f in range(1, levels + 1):
        for m in range(1, levels + 1):
            fm = float(fm_score(f, m, settings["fm_rounding"]))
            intervals = []
            for rule in rules:
                if all(rule.get(f"{key}_min", 1) <= value <= rule.get(f"{key}_max", levels)
                       for key, value in [("f", f), ("m", m), ("fm", fm)]):
                    lo = int(np.ceil(rule.get("r_min", 1)))
                    hi = int(np.floor(rule.get("r_max", levels)))
                    if lo <= hi:
                        intervals.append((lo, hi))
            next_r = 1
            for lo, hi in sorted(intervals):
                if lo != next_r:
                    raise ValueError(f"Custom rules overlap or leave a gap at F={f}, M={m}; cover every possible R/F/M score exactly once.")
                next_r = hi + 1
            if next_r != levels + 1:
                raise ValueError(f"Custom rules leave uncovered scores at F={f}, M={m}.")
    return rules


def score_rfm(customers, config, audit):
    result = customers.copy()
    settings, levels = config["rfm"], config["rfm"]["score_levels"]
    if not isinstance(levels, int) or isinstance(levels, bool) or levels < 2:
        raise ValueError("rfm.score_levels must be an integer >=2.")
    resolved = {}
    for key, column in [("r", "R_days"), ("f", "Frequency"), ("m", "Monetary")]:
        scoring = dict(settings["scoring"][key])
        if key == "f" and "frequency_thresholds" in settings:
            if scoring.get("method", "percentile") != "percentile" or "thresholds" in scoring:
                raise ValueError("Use either frequency_thresholds or scoring.f, not both threshold definitions.")
            scoring = {"method": "thresholds", "thresholds": settings["frequency_thresholds"]}
        method = scoring.get("method", "percentile")
        if method == "percentile":
            values = percent_score(result[column], levels, ascending=key != "r")
        elif method == "thresholds":
            thresholds = scoring.get("thresholds")
            if not isinstance(thresholds, list) or len(thresholds) != levels - 1 or not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in thresholds) or any(
                    a >= b for a, b in zip(thresholds, thresholds[1:])):
                raise ValueError(f"scoring.{key}.thresholds requires score_levels-1 finite, strictly increasing upper bounds.")
            values = np.searchsorted(thresholds, result[column], side="left") + 1
            if key == "r":
                values = levels + 1 - values
        else:
            raise ValueError(f"Unknown scoring.{key}.method: {method}.")
        result[f"{key.upper()}_score"] = values
        resolved[key] = scoring
    result["FM_score"] = fm_score(result["F_score"], result["M_score"], settings["fm_rounding"])
    result["RFM_code"] = result[["R_score", "F_score", "M_score"]].astype(str).agg("-".join, axis=1)
    segmentation = settings["segmentation"]
    mode = segmentation["mode"]
    if mode == "generic_matrix":
        cutoffs = segmentation["cutoffs"]
        low, mid = cutoffs["low_max"], cutoffs["mid_max"]
        if not isinstance(low, (int, float)) or not isinstance(mid, (int, float)) or not (0 <= low < mid < 1):
            raise ValueError("Matrix cutoffs require 0 <= low_max < mid_max < 1 on normalized scores.")
        for column, label in [("R_score", "Current_band"), ("FM_score", "Historical_band")]:
            normalized = (result[column] - 1) / (levels - 1)
            result[label] = np.select([normalized.le(low), normalized.le(mid)], ["low", "mid"], default="high")
        result["Customer_group"] = "Current_" + result["Current_band"] + "__Historical_" + result["Historical_band"]
    elif mode == "shopify":
        if levels != 5 or settings["fm_rounding"] != "floor":
            raise ValueError("Shopify segmentation requires score_levels=5 and fm_rounding=floor.")
        result["Customer_group"] = shopify_groups(result["R_score"], result["FM_score"])
    elif mode == "custom":
        rules = custom_rules({**segmentation, "fm_rounding": settings["fm_rounding"]}, levels)
        conditions = []
        for rule in rules:
            mask = pd.Series(True, index=result.index)
            for key in ["r", "f", "m", "fm"]:
                mask &= result[f"{key.upper()}_score"].between(rule.get(f"{key}_min", 1), rule.get(f"{key}_max", levels))
            conditions.append(mask)
        result["Customer_group"] = np.select(conditions, [rule["label"] for rule in rules], default="Unassigned")
    else:
        raise ValueError("rfm.segmentation.mode must be generic_matrix, shopify or custom.")
    if result["Customer_group"].eq("Unassigned").any():
        raise ValueError("RFM groups failed coverage check.")
    audit["rfm_scoring"] = {"score_levels": levels, "scoring": resolved, "percentile_ties": "min rank; identical values receive identical scores",
                            "r_direction": "fewer days => higher score", "threshold_boundaries": "inclusive upper bounds; R bins are reversed",
                            "fm_rounding": settings["fm_rounding"], "segmentation": segmentation,
                            "matrix_normalization": "(score - 1) / (score_levels - 1); <= low_max is low, <= mid_max is mid",
                            "current_axis": "recency score only", "historical_axis": "mean F/M score with configured rounding"}
    return result


def divide(series, total):
    return series / total if total > 0 else series * np.nan


def profile(frame, key):
    result = frame.groupby(key, observed=True).agg(
        Customers=("CustomerID", "size"), R_mean=("R_days", "mean"),
        F_mean=("Frequency", "mean"), M_mean=("Monetary", "mean"),
        Orders=("Frequency", "sum"), M_total=("Monetary", "sum"))
    for source, short in [("R_days", "R"), ("Frequency", "F"), ("Monetary", "M")]:
        for q, name in [(.25, "q25"), (.5, "median"), (.75, "q75")]:
            result[f"{short}_{name}"] = frame.groupby(key, observed=True)[source].quantile(q)
    result["Customer_share"] = result["Customers"] / len(frame)
    result["M_share"] = divide(result["M_total"], frame["Monetary"].sum())
    return result.reset_index()


def cross_profile(frame, minimum):
    result = profile(frame, ["Cluster", "Customer_group"])
    result["RowCustomerShare"] = result["Customers"] / result.groupby("Cluster")["Customers"].transform("sum")
    result["ColumnCustomerShare"] = result["Customers"] / result.groupby("Customer_group")["Customers"].transform("sum")
    result["MinCustomersGuidance"] = minimum
    result["BelowMinCustomers"] = result["Customers"].lt(minimum)
    return result


def future_metrics(frame, start, end, by_month=False):
    keys = ["customer_id"]
    orders = purchase_orders(frame, start, end)
    if by_month:
        frame = frame.copy()
        frame["Month"] = frame["date"].dt.to_period("M").astype(str)
        orders = orders.copy()
        orders["Month"] = orders["OrderDate"].dt.to_period("M").astype(str)
        keys.append("Month")
    money = frame.groupby(keys)["Value"].sum().rename("Monetary").to_frame()
    frequency = orders.groupby(keys)["order_id"].nunique().rename("Frequency")
    result = money.join(frequency, how="outer").reset_index().rename(columns={"customer_id": "CustomerID"})
    result["Monetary"] = result["Monetary"].fillna(0)
    result["Frequency"] = result["Frequency"].fillna(0).astype(int)
    return result


def validation_summary(frame, group_cols, monthly=False):
    prefix = "Monthly" if monthly else "Future"
    result = frame.groupby(group_cols, observed=True).agg(
        Customers=("CustomerID", "size"), PurchasingCustomers=("Purchased", "sum"),
        PurchaseRate=("Purchased", "mean"), TotalOrders=("Frequency", "sum"),
        AvgOrders=("Frequency", "mean"), TotalMonetary=("Monetary", "sum"),
        AvgMonetary=("Monetary", "mean")).reset_index()
    result = result.rename(columns={"PurchaseRate": f"{prefix}PurchaseRate", "TotalOrders": f"Total{prefix}Orders",
        "AvgOrders": f"Avg{prefix}Orders", "TotalMonetary": f"Total{prefix}Monetary", "AvgMonetary": f"Avg{prefix}Monetary"})
    if monthly:
        denominator = result.groupby("Month")[["Customers", f"Total{prefix}Orders", f"Total{prefix}Monetary"]].transform("sum")
        result["CustomerShare"] = result["Customers"] / denominator["Customers"]
        result["OrderShare"] = result[f"Total{prefix}Orders"] / denominator[f"Total{prefix}Orders"].replace(0, np.nan)
        result["MonetaryShare"] = result[f"Total{prefix}Monetary"] / denominator[f"Total{prefix}Monetary"].where(denominator[f"Total{prefix}Monetary"].gt(0))
    else:
        result["CustomerShare"] = result["Customers"] / len(frame)
        result["OrderShare"] = divide(result[f"Total{prefix}Orders"], frame["Frequency"].sum())
        result["MonetaryShare"] = divide(result[f"Total{prefix}Monetary"], frame["Monetary"].sum())
    return result


def transformed_features(customers, config, supplied, audit):
    settings = config["kmeans"]
    transformed = customers[["R_days", "Frequency", "Monetary"]].astype(float).copy()
    requested = settings["transforms"]
    if requested == "auto":
        requested = {column: "auto" for column in transformed}
    if not isinstance(requested, dict) or set(requested) - set(transformed.columns):
        raise ValueError("kmeans.transforms must map R_days/Frequency/Monetary to supported transforms, or be auto.")
    explicit = supplied.get("kmeans", {}).get("transforms", {})
    resolved = {}
    for column in transformed:
        mode = requested.get(column, "auto")
        legacy_key = {"Frequency": "log_frequency", "Monetary": "log_monetary"}.get(column)
        if legacy_key in settings and not (isinstance(explicit, dict) and column in explicit):
            mode = ("signed_log" if column == "Monetary" and transformed[column].lt(0).any() else "log1p") if settings[legacy_key] else "identity"
        allowed = ["auto", "identity", "log1p"] + (["signed_log"] if column == "Monetary" else [])
        if mode not in allowed:
            raise ValueError(f"Unsupported transform for {column}: {mode}; expected {allowed}.")
        skew = float(transformed[column].skew())
        actual = mode
        if mode == "auto":
            actual = "signed_log" if transformed[column].lt(0).any() else "log1p" if np.isfinite(skew) and skew > 1 else "identity"
        if actual == "log1p":
            if transformed[column].lt(0).any():
                raise ValueError(f"{column} contains negatives; log1p requires nonnegative inputs. Monetary supports signed_log.")
            transformed[column] = np.log1p(transformed[column])
        elif actual == "signed_log":
            transformed[column] = np.sign(transformed[column]) * np.log1p(transformed[column].abs())
        resolved[column] = {"requested": mode, "actual": actual, "historical_skew": skew if np.isfinite(skew) else None}
    scaler_name = settings["scaler"]
    if scaler_name not in ["standard", "robust"]:
        raise ValueError("kmeans.scaler must be standard or robust.")
    scaler = StandardScaler() if scaler_name == "standard" else RobustScaler()
    x = scaler.fit_transform(transformed)
    center = scaler.mean_ if scaler_name == "standard" else scaler.center_
    audit["feature_transforms"] = resolved
    audit["scaler"] = {"method": scaler_name, "columns": list(transformed.columns),
                       "center": center.tolist(), "scale": scaler.scale_.tolist(), "fit_scope": "historical eligible cohort only"}
    return transformed, x


def kmeans_seed_protocol(settings):
    """Use distinct non-reference seeds so self-comparisons cannot inflate ARI."""
    def valid_seed(value):
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2 ** 32

    reference = settings["random_state"]
    if not valid_seed(reference):
        raise ValueError("kmeans.random_state must be an integer in [0, 2**32-1].")
    requested = settings["seeds"]
    if not isinstance(requested, list) or any(not valid_seed(seed) for seed in requested):
        raise ValueError("kmeans.seeds must be a list of integers in [0, 2**32-1].")
    unique = list(dict.fromkeys(requested))
    seeds = [seed for seed in unique if seed != reference]
    if len(seeds) < 2:
        raise ValueError("K diagnostics require at least 2 distinct non-reference kmeans.seeds after deduplication and excluding random_state; 3 or more are recommended. Omit seeds to use [0, 7, 21, 101]. An empty seeds list does not provide K-selection stability evidence.")
    if not isinstance(settings["n_init"], int) or isinstance(settings["n_init"], bool) or settings["n_init"] < 1:
        raise ValueError("kmeans.n_init must be a positive integer, shared by every reference and comparison fit.")
    return seeds, {"reference_seed": reference, "requested_seeds": requested,
                   "comparison_seeds": seeds, "comparison_count_per_k": len(seeds),
                   "duplicate_entries_removed": len(requested) - len(unique),
                   "reference_seed_excluded": reference in unique,
                   "minimum_non_reference_seeds": 2, "recommended_non_reference_seeds": "3 or more",
                   "n_init": settings["n_init"], "comparison": "each non-reference fit versus the reference fit at the same K",
                   "feature_scope": "same historical eligible customers, transforms and scaling for all fits",
                   "interpretation": "initialization stability only; not temporal stability or external validation"}


def plot_k_selection(diagnostics, path):
    """Show all evidence without automatically declaring an optimal K or elbow."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ordered = diagnostics.sort_values("K")
    if not ordered.empty:
        k_values = ordered["K"].to_numpy()
        axes[0].plot(k_values, ordered["Inertia"], marker="o")
        axes[1].plot(k_values, ordered["Silhouette"], marker="o")
        axes[2].plot(k_values, ordered["ARI_mean"], marker="o", label="mean")
        axes[2].plot(k_values, ordered["ARI_min"], marker="v", linestyle="--", label="min")
        axes[2].plot(k_values, ordered["ARI_max"], marker="^", linestyle="--", label="max")
        axes[2].legend()
        for axis in axes:
            axis.set_xticks(k_values)
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No valid candidate K", transform=axis.transAxes, ha="center")
    if len(ordered) < 2:
        axes[0].text(0.5, 0.95, "Insufficient K values for elbow evidence",
                     transform=axes[0].transAxes, ha="center", va="top", fontsize=9)
    axes[0].set(title="Elbow diagnostic", ylabel="Inertia")
    axes[1].set(title="Silhouette diagnostic", ylabel="Silhouette")
    axes[2].set(title="Seed stability diagnostic", ylabel="ARI versus reference seed")
    for axis in axes:
        axis.set_xlabel("Candidate K")
        axis.grid(alpha=0.25)
    figure.suptitle("K selection evidence: combine all three metrics with cluster profiles; no automatic optimum")
    try:
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def future_oracle_reference(evaluation):
    """Post-hoc same-N monetary upper reference; never a selectable strategy."""
    ordered = evaluation.sort_values(["Monetary", "CustomerID"], ascending=[False, True], kind="mergesort")
    ids = ordered["CustomerID"].tolist()
    totals = ordered["Monetary"].cumsum().to_numpy()
    monetary = evaluation.set_index("CustomerID")["Monetary"]
    top_ids_by_count = {}

    def reference(selected_ids):
        count = len(selected_ids)
        denominator = float(totals[count - 1])
        selected_m = float(monetary.loc[selected_ids].sum())
        if count not in top_ids_by_count:
            top_ids_by_count[count] = set(ids[:count])
        return {"OracleSameNMonetary": denominator,
                "OracleMonetaryCapture": selected_m / denominator if denominator > 0 else np.nan,
                "OracleCustomerOverlapShare": len(set(selected_ids) & top_ids_by_count[count]) / count}

    return reference


def budget_plan(settings, size):
    budgets = settings["budgets"]
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("baseline.budgets must be a nonempty list of fractions (0,1] or {customers: integer}.")
    plans = []
    for index, budget in enumerate(budgets, 1):
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and 0 < budget <= 1:
            kind, value = "fraction", float(budget)
            count = max(1, int(np.floor(size * value)))
        elif isinstance(budget, dict) and set(budget) == {"customers"} and isinstance(budget["customers"], int) and not isinstance(budget["customers"], bool) and 1 <= budget["customers"] <= size:
            kind, value, count = "customers", budget["customers"], budget["customers"]
        else:
            raise ValueError("Each baseline budget must be a fraction (0,1] or {customers: integer between 1 and cohort size}.")
        plans.append({"BudgetID": f"b{index:02d}", "BudgetType": kind, "RequestedBudget": value,
                      "SelectedCustomers": count, "ActualCustomerShare": count / size})
    return plans


def action_order(customers, column, settings, kind):
    priorities = settings.get(f"{kind}_priority")
    rule = settings.get("historical_priority_rule")
    rule = rule.get(kind) if isinstance(rule, dict) else rule
    labels = customers[column].astype(str)
    if priorities is not None:
        if not isinstance(priorities, list) or not priorities or len(set(map(str, priorities))) != len(priorities):
            raise ValueError(f"baseline.{kind}_priority must be a nonempty ordered list without duplicates.")
        priorities = list(map(str, priorities))
        missing = set(labels) - set(priorities)
        if missing:
            raise ValueError(f"baseline.{kind}_priority omits observed groups: {sorted(missing)}.")
        basis = "configured group priorities; CustomerID breaks within-group ties"
    elif rule is not None:
        choices = {"monetary_desc": ("Monetary", False), "frequency_desc": ("Frequency", False), "recency_asc": ("R_days", True)}
        if rule not in choices:
            raise ValueError("historical_priority_rule must be monetary_desc, frequency_desc or recency_asc, optionally mapped separately to rfm and cluster.")
        feature, ascending = choices[rule]
        averages = customers.assign(GroupLabel=labels).groupby("GroupLabel")[feature].mean().reset_index()
        priorities = averages.sort_values([feature, "GroupLabel"], ascending=[ascending, True], kind="mergesort")["GroupLabel"].tolist()
        basis = f"historical group mean {rule}; CustomerID breaks within-group ties"
    else:
        return None, None, None
    rank = labels.map({label: index for index, label in enumerate(priorities)})
    ordered = customers.assign(Priority=rank).sort_values(["Priority", "CustomerID"], kind="mergesort")["CustomerID"].tolist()
    return ordered, basis, priorities


def baseline_outputs(customers, future, config, audit, tables):
    """Freeze every ranking from history before evaluating an equal-sized selection."""
    settings = config["baseline"]
    if not isinstance(settings["budget_is_business"], bool):
        raise ValueError("baseline.budget_is_business must be true or false.")
    if not isinstance(settings["random_seed"], int) or isinstance(settings["random_seed"], bool) or settings["random_seed"] < 0:
        raise ValueError("baseline.random_seed must be a nonnegative integer for reproducible random comparisons.")
    plans = budget_plan(settings, len(customers))
    stable = customers.sort_values("CustomerID", kind="mergesort")
    strategies = []
    for name, column, ascending in [("baseline_monetary", "Monetary", False), ("baseline_frequency", "Frequency", False), ("baseline_recency", "R_days", True)]:
        ordered = stable.sort_values(column, ascending=ascending, kind="mergesort")["CustomerID"].tolist()
        strategies.append((name, ordered, f"historical {column}; CustomerID breaks ties"))
    strategies.append(("baseline_random", np.random.default_rng(settings["random_seed"]).permutation(stable["CustomerID"].to_numpy()).tolist(),
                       f"seeded random permutation, seed={settings['random_seed']}"))
    priority_audit, missing = {}, []
    for kind, column in [("rfm", "Customer_group"), ("cluster", "Cluster")]:
        if column not in customers:
            continue
        ordered, basis, priorities = action_order(customers, column, settings, kind)
        if ordered is None:
            missing.append(kind)
        else:
            strategies.append((f"{kind}_priority", ordered, basis))
            priority_audit[kind] = {"priority": priorities, "basis": basis, "frozen_from": "history only"}
    use_future = future is not None
    evaluation = future if use_future else customers.assign(Purchased=customers["Frequency"].gt(0).astype(int))
    oracle_reference = future_oracle_reference(evaluation) if use_future else None
    scope = "holdout_same_cohort_descriptive" if use_future else "in_sample_descriptive"
    eval_index = evaluation.set_index("CustomerID")
    historical = customers.set_index("CustomerID")
    total_m, total_f = float(evaluation["Monetary"].sum()), float(evaluation["Frequency"].sum())
    total_buyers = int(evaluation["Purchased"].sum())
    rankings, metrics, structures = [], [], []
    for name, ordered, basis in strategies:
        rankings.append(pd.DataFrame({"Strategy": name, "CustomerID": ordered, "Rank": np.arange(1, len(ordered) + 1), "RankingBasis": basis}))
        for plan in plans:
            selected_ids = ordered[:plan["SelectedCustomers"]]
            selected = eval_index.loc[selected_ids]
            current = historical.loc[selected_ids]
            money, frequency, buyers = float(selected["Monetary"].sum()), int(selected["Frequency"].sum()), int(selected["Purchased"].sum())
            row = {**plan, "Strategy": name, "EvaluationScope": scope, "BusinessBudgetDefined": settings["budget_is_business"],
                   "BudgetBasis": "business" if settings["budget_is_business"] else "scenario",
                   "CohortCustomers": len(customers), "TotalMonetary": money, "AvgMonetary": money / len(selected),
                   "MonetaryCoverage": money / total_m if total_m > 0 else np.nan,
                   "TotalOrders": frequency, "OrderCoverage": frequency / total_f if total_f else np.nan,
                   "PurchaseRate": buyers / len(selected), "PurchasingCustomers": buyers,
                   "PurchasingCustomerCoverage": buyers / total_buyers if total_buyers else np.nan,
                   "CurrentMonetary": float(current["Monetary"].sum()), "CurrentOrders": int(current["Frequency"].sum()),
                   "MechanicalCurrentMonetaryUpperBound": not use_future and name == "baseline_monetary"}
            if oracle_reference is not None:
                row.update(oracle_reference(selected_ids))
            metrics.append(row)
            for column in ["Customer_group"] + (["Cluster"] if "Cluster" in current else []):
                grouped = current.reset_index().groupby(column, observed=True).agg(Customers=("CustomerID", "size"), CurrentMonetary=("Monetary", "sum"), CurrentOrders=("Frequency", "sum")).reset_index()
                cohort_counts = customers.groupby(column, observed=True).size()
                for record in grouped.to_dict("records"):
                    structures.append({**plan, "Strategy": name, "Grouping": column, "Group": record[column],
                                       "Customers": record["Customers"], "SelectedCustomerShare": record["Customers"] / len(current),
                                       "GroupCustomerCoverage": record["Customers"] / int(cohort_counts.loc[record[column]]),
                                       "CurrentMonetary": record["CurrentMonetary"], "CurrentOrders": record["CurrentOrders"]})
    metrics = pd.DataFrame(metrics)
    comparisons = []
    for plan in plans:
        rows = metrics.loc[metrics["BudgetID"].eq(plan["BudgetID"])].set_index("Strategy")
        for name in [name for name in rows.index if name.endswith("_priority")]:
            for baseline in [name for name in rows.index if name.startswith("baseline_")]:
                comparison = {**plan, "Strategy": name, "Baseline": baseline, "EvaluationScope": scope,
                              "BusinessBudgetDefined": settings["budget_is_business"],
                              "BudgetBasis": "business" if settings["budget_is_business"] else "scenario",
                              "Interpretation": "descriptive holdout comparison; no causal or significance claim" if use_future else "current-window coverage only; not predictive lift"}
                for metric in ["MonetaryCoverage", "AvgMonetary", "PurchaseRate", "OrderCoverage", "PurchasingCustomerCoverage"]:
                    comparison[f"Delta{metric}"] = rows.loc[name, metric] - rows.loc[baseline, metric]
                comparisons.append(comparison)
    tables["baseline_rankings.csv"] = pd.concat(rankings, ignore_index=True)
    tables["baseline_metrics.csv"] = metrics
    tables["baseline_structure.csv"] = pd.DataFrame(structures)
    tables["baseline_comparisons.csv"] = pd.DataFrame(comparisons, columns=list(comparisons[0]) if comparisons else ["BudgetID", "Strategy", "Baseline", "EvaluationScope"])
    audit["baseline"] = {"status": "missing_action_priorities" if missing else "awaiting_k_selection" if "Cluster" not in customers else "complete",
                         "evaluation_scope": scope, "business_budget_defined": settings["budget_is_business"],
                         "budget_basis": "business" if settings["budget_is_business"] else "scenario",
                         "budget_rounding": "floor(N*fraction), at least one customer; explicit counts unchanged", "budgets": plans,
                         "priorities": priority_audit, "missing_action_priorities": missing,
                         "cluster_strategy_status": "available" if "Cluster" in customers else "not_compared_K_not_selected",
                         "compared_segmentation_strategies": [name for name, _, _ in strategies if name.endswith("_priority")],
                         "ranking_freeze": "all methods use historical eligible cohort; never rank by future monetary",
                         "random_comparison": "one seeded random realization; no uncertainty estimate",
                         "monetary_baseline_in_sample": "mechanical maximum of current total monetary for a fixed customer count; not predictive evidence"}
    audit["baseline"]["future_oracle_reference"] = {
        "status": "post_hoc_diagnostic_only" if use_future else "not_computed_without_holdout",
        "cohort": "same historical eligible cohort including zero-future outcomes",
        "ranking": "future monetary descending; CustomerID ascending breaks ties",
        "same_n_monetary": "sum of the top N signed future monetary values",
        "monetary_capture": "selected signed future monetary / OracleSameNMonetary; NaN when denominator <= 0; no clipping",
        "customer_overlap_share": "size of selected-ID intersection with oracle top-N IDs / N; distinct from monetary capture",
        "restriction": "post-hoc upper reference only; never added to strategy rankings, priorities or baseline comparisons"}
    if total_m <= 0:
        audit["warnings"].append("Baseline evaluation cohort monetary denominator is nonpositive: monetary coverage is undefined (NaN); signed totals remain unchanged.")
    if not settings["budget_is_business"]:
        audit["warnings"].append("Baseline budgets are sensitivity scenarios, not deployed business budgets: equal-count descriptive comparisons are allowed; no deployment advantage or ROI conclusion.")
    if missing:
        audit["warnings"].append(f"No actionable priority configured for {', '.join(missing)}; supply complete group priorities or an explicit historical_priority_rule to compare those strategies.")


def run(config_path, output):
    base = config_path.parent
    supplied = json.loads(config_path.read_text(encoding="utf-8-sig"))
    supported = set(DEFAULTS) | {"input", "columns", "reference_date", "analysis_start", "analysis_end", "currency", "holdout"}
    if set(supplied) - supported:
        raise ValueError(f"Unsupported top-level config keys: {sorted(set(supplied) - supported)}")
    config = merged(DEFAULTS, supplied)
    for field in ["drop_exact_duplicates", "returns.signed_amount", "returns.negative_quantity",
                  "purchases.allow_zero_amount", "kmeans.log_frequency", "kmeans.log_monetary",
                  "holdout.monthly"]:
        value = config
        for part in field.split("."):
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            if not isinstance(value, bool):
                raise ValueError(f"{field} must be a boolean (true or false).")
    if not isinstance(config["monetary_zero_tolerance"], (int, float)) or not np.isfinite(config["monetary_zero_tolerance"]) or config["monetary_zero_tolerance"] < 0:
        raise ValueError("monetary_zero_tolerance must be finite and nonnegative.")
    # Windows CPU probing can depend on unavailable system commands; keep this process bounded.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    for key, allowed in {"invalid_rows": ["error", "exclude"], "monetary": ["net", "gross"],
                         "negative_monetary": ["error", "exclude", "keep", "signed_log"]}.items():
        if config[key] not in allowed:
            raise ValueError(f"Invalid {key}: expected {allowed}.")
    if config["returns"]["amount_sign"] not in ["signed", "unsigned"] or config["rfm"]["fm_rounding"] not in ["none", "floor", "round", "ceil"]:
        raise ValueError("Invalid return sign or FM rounding mode.")
    if config["order_date_policy"] not in ["error", "first", "last"]:
        raise ValueError("order_date_policy must be error, first or last.")
    minimum = config["cross"]["min_customers"]
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("cross.min_customers must be a positive integer; it is descriptive guidance, not a significance guarantee.")
    if output.exists():
        raise ValueError("Output directory already exists; select a new run directory.")
    audit = {"inputs": [], "config": config, "warnings": [], "selected": False,
             "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                          "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                          "matplotlib": matplotlib.__version__}}
    with Path(__file__).open("rb") as script_source:
        audit["runner_sha256"] = hashlib.file_digest(script_source, "sha256").hexdigest()
    raw = read_input(config["input"], config["columns"], base, audit, "analysis", config)
    level = config["input"]["level"]
    transactions = None
    lifecycle_qualification = False
    if level == "transactions":
        transactions = prepare_transactions(raw, config, audit, "analysis_source")
        history = window(transactions, config.get("analysis_start"), config.get("analysis_end"))
        customers = transaction_rfm(history, config, audit)
        end_of_history = day(config["analysis_end"]) if config.get("analysis_end") else history["BusinessDate"].max()
        late_order_keys = history.loc[history["Purchase"] & history["OrderDate"].gt(end_of_history), ["customer_id", "order_id"]].drop_duplicates()
        audit["historical_orders_with_later_effective_date"] = len(late_order_keys)
        lifecycle_qualification = not late_order_keys.empty
        if lifecycle_qualification:
            audit["warnings"].append("Some last-policy OrderDates depend on lifecycle rows after the observation window. Freeze a historical order snapshot upstream before treating holdout results as clean out-of-time validation.")
    elif level == "customers":
        customers = customer_rfm(raw, config, audit)
        if config.get("reference_date"):
            audit["reference_date"] = str(day(config["reference_date"]).date())
    else:
        raise ValueError("input.level must be transactions or customers.")
    customers["Monetary"] = clean_money(customers["Monetary"], config, audit, "historical_customers")
    # F can count a shorter window than R: zero recent orders does not erase a known past purchase.
    no_purchase = customers["R_days"].isna()
    excluded = customers.loc[no_purchase].assign(Reason="no_known_valid_recency")
    customers = customers.loc[~no_purchase].copy()
    negative = customers["Monetary"].lt(0)
    audit["negative_monetary_customers"] = int(negative.sum())
    if negative.any() and config["negative_monetary"] == "error":
        raise ValueError("Negative customer net amounts exist. Explicitly choose keep (retain legitimate net negatives) or exclude; signed_log is a legacy alias for keep.")
    if config["negative_monetary"] == "exclude":
        excluded = pd.concat([excluded, customers.loc[negative].assign(Reason="negative_monetary")], ignore_index=True)
        customers = customers.loc[~negative].copy()
    if customers.empty:
        raise ValueError("No eligible customers after configured exclusions.")
    customers = score_rfm(customers.reset_index(drop=True), config, audit)
    audit["eligible_customers"] = len(customers)
    audit["eligible_zero_window_frequency"] = int(customers["Frequency"].eq(0).sum())
    audit["excluded_customers"] = len(excluded)
    audit["historical_orders"] = int(customers["Frequency"].sum())
    audit["historical_monetary"] = float(customers["Monetary"].sum())
    if customers["Monetary"].sum() <= 0:
        audit["warnings"].append("Nonpositive net monetary denominator; interpret monetary shares with care.")
    settings = config["kmeans"]
    transformed, x = transformed_features(customers, config, supplied, audit)
    distinct = len(np.unique(x, axis=0))
    ks = list(dict.fromkeys(settings["candidates"] + ([settings["k"]] if settings.get("k") is not None else [])))
    valid = sorted(k for k in ks if isinstance(k, int) and not isinstance(k, bool) and 2 <= k < len(x) and k <= distinct)
    audit["skipped_k"] = [k for k in ks if k not in valid]
    selected = settings.get("k")
    if selected is not None and selected not in valid:
        raise ValueError("Selected K is invalid for sample size or distinct feature vectors.")
    if not valid:
        audit["warnings"].append("Insufficient distinct customers for meaningful K-means; only RFM available.")
    seeds, seed_protocol = kmeans_seed_protocol(settings)
    audit["kmeans_seed_protocol"] = seed_protocol
    if len(seeds) < 3:
        audit["warnings"].append("Only 2 distinct non-reference seeds configured: minimum ARI evidence is available; 3 or more are recommended for K selection.")
    sample_size = settings["silhouette_sample_size"]
    if sample_size is not None and (not isinstance(sample_size, int) or sample_size < 2):
        raise ValueError("silhouette_sample_size must be null or an integer >=2.")
    tables = {"rfm_summary.csv": profile(customers, "Customer_group"), "excluded_customers.csv": excluded}
    candidates, profiles, candidate_crosses, models, candidate_stability = [], [], [], {}, []
    with threadpool_limits(limits=4):
        for k in valid:
            model = KMeans(n_clusters=k, random_state=settings["random_state"], n_init=settings["n_init"])
            labels = model.fit_predict(x)
            models[k] = (model, labels)
            metric, error = np.nan, ""
            try:
                metric = silhouette_score(x, labels, sample_size=sample_size, random_state=settings["random_state"])
            except ValueError as exc:
                error = str(exc)
            k_stability = []
            for seed in seeds:
                comparison = KMeans(n_clusters=k, random_state=seed, n_init=settings["n_init"]).fit_predict(x)
                ari = float(adjusted_rand_score(labels, comparison))
                k_stability.append(ari)
                candidate_stability.append({"K": k, "ReferenceSeed": settings["random_state"], "Seed": seed, "ARI": ari})
            cluster_sizes = pd.Series(labels).value_counts()
            candidates.append({"K": k, "Inertia": model.inertia_, "Silhouette": metric,
                               "ARI_mean": float(np.mean(k_stability)), "ARI_min": float(np.min(k_stability)),
                               "ARI_max": float(np.max(k_stability)), "ARI_std": float(np.std(k_stability, ddof=0)),
                               "ARI_comparisons": len(k_stability),
                               "SmallestCluster": int(cluster_sizes.min()), "SmallestClusterShare": float(cluster_sizes.min() / len(x)),
                               "LargestClusterShare": float(cluster_sizes.max() / len(x)),
                               "ClustersBelowMinCustomers": int(cluster_sizes.lt(minimum).sum()),
                               "SilhouetteSampled": sample_size is not None and sample_size < len(x), "MetricNote": error})
            candidate_profile = profile(customers.assign(Cluster=labels), "Cluster")
            candidate_profile.insert(0, "K", k)
            profiles.append(candidate_profile)
            candidate_cross = cross_profile(customers.assign(Cluster=labels), minimum)
            candidate_cross.insert(0, "K", k)
            candidate_crosses.append(candidate_cross)
        if selected is not None:
            model, labels = models[selected]
            customers["Cluster"] = labels
            tables["cluster_summary.csv"] = profile(customers, "Cluster")
            cross = pd.crosstab(customers["Cluster"], customers["Customer_group"])
            tables["cross_counts.csv"] = cross.reset_index()
            tables["cross_row_share.csv"] = cross.div(cross.sum(axis=1), axis=0).reset_index()
            tables["cross_column_share.csv"] = cross.div(cross.sum(axis=0), axis=1).reset_index()
            tables["cross_history_summary.csv"] = cross_profile(customers, minimum)
            tables["seed_stability.csv"] = pd.DataFrame(
                [{key: record[key] for key in ["ReferenceSeed", "Seed", "ARI"]}
                 for record in candidate_stability if record["K"] == selected])
            audit["selected"] = True
    diagnostics = pd.DataFrame(candidates, columns=["K", "Inertia", "Silhouette", "ARI_mean", "ARI_min", "ARI_max", "ARI_std", "ARI_comparisons", "SmallestCluster", "SmallestClusterShare", "LargestClusterShare", "ClustersBelowMinCustomers", "SilhouetteSampled", "MetricNote"])
    previous_inertia = diagnostics["Inertia"].shift(1)
    diagnostics["PreviousK"] = diagnostics["K"].shift(1)
    diagnostics["InertiaDrop"] = previous_inertia - diagnostics["Inertia"]
    diagnostics["InertiaRelativeDrop"] = diagnostics["InertiaDrop"] / previous_inertia.where(previous_inertia.gt(0))
    tables["k_diagnostics.csv"] = diagnostics
    tables["candidate_seed_stability.csv"] = pd.DataFrame(candidate_stability, columns=["K", "ReferenceSeed", "Seed", "ARI"])
    audit["k_selection_evidence"] = {
        "status": "candidate_evidence_ready_for_review" if len(valid) >= 2 else "single_k_reproduction" if len(valid) == 1 and selected is not None else "single_k_diagnostics" if valid else "unavailable_insufficient_vectors",
        "valid_candidate_k": valid, "configured_selected_k": selected,
        "selection_source": "explicit config k; not an automatic optimum" if selected is not None else "awaiting explicit selection after evidence review",
        "required_review": ["inertia curve / elbow and relative drops", "silhouette", "ARI mean/min/max/std across seeds", "candidate profiles, small clusters and business interpretability"],
        "elbow_evidence": "available for visual review; no unique elbow or optimum is asserted" if len(valid) >= 2 else "insufficient: fewer than 2 valid candidate K values",
        "inertia_drop_basis": "previous lower valid K in ascending order; first K and nonpositive previous inertia have no relative drop",
        "ari_std_definition": "population standard deviation (ddof=0) over non-reference comparisons",
        "decision_rule": "assistant and user combine all three metrics with cluster profiles; no single metric or elbow heuristic automatically chooses K",
        "selection_justification_complete": False,
        "files": ["k_diagnostics.csv", "candidate_seed_stability.csv", "candidate_profiles.csv", "candidate_cross_summary.csv", "k_selection_diagnostics.png"]}
    if len(valid) == 1:
        audit["warnings"].append("Only one valid K was run: this is reproduction/diagnostics, not a completed K-selection argument; elbow evidence is insufficient.")
    tables["candidate_profiles.csv"] = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame(columns=["K", "Cluster", "Customers"])
    tables["candidate_cross_summary.csv"] = pd.concat(candidate_crosses, ignore_index=True) if candidate_crosses else pd.DataFrame(columns=["K", "Cluster", "Customer_group", "Customers", "RowCustomerShare", "ColumnCustomerShare"])
    audit["cross"] = {"status": "selected" if selected is not None else "candidate_only" if valid else "unavailable_insufficient_vectors",
                      "min_customers_guidance": minimum, "interpretation": "descriptive overlap of two historical segmentations; agreement is not independent validation"}
    tables["customer_labels.csv"] = customers
    tables["kmeans_transformed_features.csv"] = pd.concat([customers[["CustomerID"]], transformed.add_prefix("Transformed_")], axis=1)
    holdout = config.get("holdout")
    joined = None
    if holdout:
        start, end = day(holdout["start"]), day(holdout["end"])
        if start > end:
            raise ValueError("holdout.start must not follow holdout.end.")
        if not audit.get("reference_date"):
            raise ValueError("Future validation requires a documented reference_date.")
        if day(audit["reference_date"]) > start:
            raise ValueError("Reference date must not follow holdout start.")
        if level == "transactions" and (history["BusinessDate"].max() >= start or
              (config.get("analysis_end") and day(config["analysis_end"]) >= start)):
            raise ValueError("History and holdout windows overlap.")
        if "input" in holdout:
            if holdout["input"].get("level", "transactions") != "transactions":
                raise ValueError("Holdout must contain transaction rows.")
            future = read_input(holdout["input"], holdout.get("columns", config["columns"]), base, audit, "holdout", config)
            future = prepare_transactions(future, config, audit, "holdout_source")
            if transactions is not None:
                past_keys = history.loc[history["Purchase"], ["customer_id", "order_id"]].drop_duplicates()
                future_keys = window(future, holdout["start"], holdout["end"])
                future_keys = future_keys.loc[future_keys["Purchase"], ["customer_id", "order_id"]].drop_duplicates()
                reused_orders = len(past_keys.merge(future_keys, on=["customer_id", "order_id"]))
                audit["orders_reused_across_separate_history_holdout_inputs"] = reused_orders
                if reused_orders:
                    lifecycle_qualification = True
                    audit["warnings"].append("Separate history/holdout sources reuse customer/order keys. Canonical lifecycle dates are source-local; unify order history upstream to prevent double-counted frequency before claiming clean future validation.")
        elif transactions is not None and config.get("analysis_end"):
            future = transactions
        else:
            raise ValueError("Holdout needs a separate transaction input or an explicit same-source analysis_end.")
        future = window(future, holdout["start"], holdout["end"])
        if future.empty:
            audit["warnings"].append("No transactions in the holdout window. Verify data coverage before interpreting zero outcomes.")
        keys = ["CustomerID", "Customer_group"] + (["Cluster"] if selected is not None else [])
        frozen = customers[keys]
        joined = frozen.merge(future_metrics(future, holdout["start"], holdout["end"]), on="CustomerID", how="left", validate="one_to_one")
        joined[["Frequency", "Monetary"]] = joined[["Frequency", "Monetary"]].fillna(0)
        joined["Monetary"] = clean_money(joined["Monetary"], config, audit, "future_customers")
        joined["Frequency"] = joined["Frequency"].astype(int)
        joined["Purchased"] = joined["Frequency"].gt(0).astype(int)
        tables["customer_future_metrics.csv"] = joined.rename(columns={"Frequency": "FutureFrequency", "Monetary": "FutureMonetary", "Purchased": "Repurchased"})
        tables["rfm_future_summary.csv"] = validation_summary(joined, ["Customer_group"])
        if selected is not None:
            tables["cluster_future_summary.csv"] = validation_summary(joined, ["Cluster"])
            joint = validation_summary(joined, ["Cluster", "Customer_group"])
            joint["MinCustomersGuidance"] = minimum
            joint["BelowMinCustomers"] = joint["Customers"].lt(minimum)
            cross_oracle = future_oracle_reference(joined)
            cross_oracle_rows = []
            for record in joint.to_dict("records"):
                ids = joined.loc[joined["Cluster"].eq(record["Cluster"]) & joined["Customer_group"].eq(record["Customer_group"]), "CustomerID"].tolist()
                cross_oracle_rows.append(cross_oracle(ids))
            joint = pd.concat([joint, pd.DataFrame(cross_oracle_rows)], axis=1)
            tables["cross_future_summary.csv"] = joint
            audit["cross"]["future_oracle_reference"] = "each frozen cross group is compared with post-hoc top-N signed future monetary in the whole eligible cohort at the same group N; amount capture and ID overlap are separate diagnostics; no clipping or strategy ranking"
        audit["future_cohort"] = {"customers": len(joined), "orders": int(joined["Frequency"].sum()),
            "monetary": float(joined["Monetary"].sum()), "repurchased": int(joined["Purchased"].sum())}
        audit["holdout_assessment"] = {"status": "qualified_order_lifecycle_crosses_cutoff" if lifecycle_qualification else "separate_observation_window",
                                       "cohort": "historical eligible customers, including zero-future outcomes",
                                       "new_future_customers": "excluded from same-cohort denominators"}
        if holdout.get("monthly", True):
            months = pd.DataFrame({"Month": pd.period_range(start, end, freq="M").astype(str)})
            monthly = frozen.merge(months, how="cross").merge(future_metrics(future, holdout["start"], holdout["end"], True),
                on=["CustomerID", "Month"], how="left", validate="one_to_one")
            monthly[["Frequency", "Monetary"]] = monthly[["Frequency", "Monetary"]].fillna(0)
            monthly["Monetary"] = clean_money(monthly["Monetary"], config, audit, "future_customer_months")
            monthly["Frequency"] = monthly["Frequency"].astype(int)
            monthly["Purchased"] = monthly["Frequency"].gt(0).astype(int)
            tables["customer_monthly_metrics.csv"] = monthly.rename(columns={"Frequency": "MonthlyOrders", "Monetary": "MonthlyMonetary", "Purchased": "PurchasedThisMonth"})
            tables["rfm_monthly_summary.csv"] = validation_summary(monthly, ["Month", "Customer_group"], True)
            if selected is not None:
                tables["cluster_monthly_summary.csv"] = validation_summary(monthly, ["Month", "Cluster"], True)
                joint_monthly = validation_summary(monthly, ["Month", "Cluster", "Customer_group"], True)
                joint_monthly["MinCustomersGuidance"] = minimum
                joint_monthly["BelowMinCustomers"] = joint_monthly["Customers"].lt(minimum)
                tables["cross_monthly_summary.csv"] = joint_monthly
            if start.day != 1 or end != end + pd.offsets.MonthEnd(0):
                audit["warnings"].append("Monthly validation includes partial months; compare coverage explicitly.")
    baseline_outputs(customers, joined, config, audit, tables)
    audit["stage_status"] = {"governance": "complete_with_audit", "rfm": "complete",
                             "kmeans": "selected" if selected is not None else "diagnostics_only" if valid else "unavailable_insufficient_vectors",
                             "baseline": audit["baseline"]["status"], "cross": audit["cross"]["status"]}
    for info in audit["inputs"]:
        if Path(info["path"]).is_relative_to(output):
            raise ValueError("Output directory cannot contain an input file.")
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / name, index=False, encoding="utf-8-sig")
    plot_k_selection(diagnostics, output / "k_selection_diagnostics.png")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "eligible_customers": len(customers), "selected": audit["selected"],
                      "baseline_status": audit["baseline"]["status"], "cross_status": audit["cross"]["status"],
                      "selected_k": selected, "files": sorted(tables) + ["k_selection_diagnostics.png", "audit.json"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.config.resolve(), args.output.resolve())
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(2, f"Segmentation stopped: {error}\n")
