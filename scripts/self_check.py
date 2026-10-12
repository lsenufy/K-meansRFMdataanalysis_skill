"""Audit bundled runner artifacts in three rounds; pass does not authenticate sources or prove causality."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
TOLERANCE = 1e-8
RELATIVE_TOLERANCE = 1e-12
SHOPIFY_GRID = [
    ["Dormant", "Dormant", "At risk", "At risk", "Previously loyal"],
    ["Dormant", "Dormant", "At risk", "At risk", "Previously loyal"],
    ["Almost lost", "Almost lost", "Needs attention", "Loyal", "Loyal"],
    ["Promising", "Active", "Active", "Loyal", "Loyal"],
    ["New", "Active", "Active", "Champions", "Champions"],
]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def save_json(path, content):
    path.write_text(json.dumps(json_value(content), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_artifact_csv(path):
    # Converters preserve literal identifiers while numeric columns retain normal NA parsing.
    return pd.read_csv(path, converters={"CustomerID": str, "Customer_group": str, "Group": str})


def overall(items):
    states = {item["status"] for item in items}
    return "fail" if "fail" in states else "review" if "review" in states else "pass"


def normalize_day(value):
    return pd.Timestamp(value).normalize()


def date_window(frame, start=None, end=None, column="date"):
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= frame[column].ge(normalize_day(start))
    if end:
        mask &= frame[column].lt(normalize_day(end) + pd.Timedelta(days=1))
    return frame.loc[mask].copy()


def independent_orders(frame, start=None, end=None):
    rows = frame.loc[frame["Purchase"], ["customer_id", "order_id", "OrderDate"]].copy()
    rows = rows.loc[~rows.duplicated(["customer_id", "order_id"], keep="first")]
    return date_window(rows, start, end, "OrderDate")


def independent_profile(frame, keys):
    keys = [keys] if isinstance(keys, str) else keys
    rows = []
    for labels, group in frame.groupby(keys, observed=True, sort=True):
        labels = labels if isinstance(labels, tuple) else (labels,)
        row = dict(zip(keys, labels))
        row.update(Customers=len(group), R_mean=group["R_days"].mean(), F_mean=group["Frequency"].mean(),
                   M_mean=group["Monetary"].mean(), Orders=group["Frequency"].sum(), M_total=group["Monetary"].sum(),
                   Customer_share=len(group) / len(frame))
        total = frame["Monetary"].sum()
        row["M_share"] = row["M_total"] / total if total > 0 else np.nan
        for source, short in [("R_days", "R"), ("Frequency", "F"), ("Monetary", "M")]:
            for quantile, suffix in [(.25, "q25"), (.5, "median"), (.75, "q75")]:
                row[f"{short}_{suffix}"] = group[source].quantile(quantile)
        rows.append(row)
    return pd.DataFrame(rows)


def independent_future_summary(frame, keys, monthly=False):
    prefix = "Monthly" if monthly else "Future"
    rows = []
    for labels, group in frame.groupby(keys, observed=True, sort=True):
        labels = labels if isinstance(labels, tuple) else (labels,)
        row = dict(zip(keys, labels))
        denominator = frame.loc[frame["Month"].eq(row["Month"])] if monthly else frame
        total_m, total_f = denominator["Monetary"].sum(), denominator["Frequency"].sum()
        row.update(Customers=len(group), PurchasingCustomers=int(group["Purchased"].sum()),
                   CustomerShare=len(group) / len(denominator),
                   OrderShare=group["Frequency"].sum() / total_f if total_f else np.nan,
                   MonetaryShare=group["Monetary"].sum() / total_m if total_m > 0 else np.nan)
        row.update({f"{prefix}PurchaseRate": group["Purchased"].mean(), f"Total{prefix}Orders": group["Frequency"].sum(),
                    f"Avg{prefix}Orders": group["Frequency"].mean(), f"Total{prefix}Monetary": group["Monetary"].sum(),
                    f"Avg{prefix}Monetary": group["Monetary"].mean()})
        rows.append(row)
    return pd.DataFrame(rows)


class Checks:
    def __init__(self, run, output, claims):
        self.run, self.output, self.claims = run, output, claims
        self.audit = json.loads((run / "audit.json").read_text(encoding="utf-8-sig"))
        self.config = self.audit["config"]
        self.items, self.state = [], {}
        self.artifacts = {str(path.relative_to(run)): digest(path) for path in sorted(run.rglob("*")) if path.is_file()}
        runner_path = Path(__file__).with_name("run_segmentation.py")
        spec = importlib.util.spec_from_file_location("bundled_segmentation_runner", runner_path)
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)
        self.runner_path = runner_path

    def add(self, name, status, evidence, **values):
        self.items.append({"check": name, "status": status, "evidence": evidence, **values})

    def condition(self, name, condition, evidence, **values):
        self.add(name, "pass" if condition else "fail", evidence, **values)

    def numbers(self, name, expected, actual, evidence, tolerance=TOLERANCE, relative_tolerance=RELATIVE_TOLERANCE):
        expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
        matching = expected.shape == actual.shape and np.allclose(expected, actual, atol=tolerance, rtol=relative_tolerance, equal_nan=True)
        delta = np.abs(expected - actual) if expected.shape == actual.shape else np.array([np.nan])
        finite = delta[np.isfinite(delta)]
        self.condition(name, matching, evidence, expected=expected.tolist(), actual=actual.tolist(),
                       difference={"max_absolute": float(finite.max()) if finite.size else None}, absolute_tolerance=tolerance, relative_tolerance=relative_tolerance)

    def csv(self, name):
        return read_artifact_csv(self.run / name)

    def table(self, name, expected, keys, columns=None):
        actual = self.csv(name)
        columns = columns or [column for column in expected if column not in keys]
        if any(column not in actual for column in keys + columns):
            self.add(name, "fail", "Required output columns are missing.", expected=keys + columns, actual=actual.columns.tolist())
            return
        if actual.duplicated(keys).any() or expected.duplicated(keys).any():
            self.add(name, "fail", "Duplicate aggregation keys prevent a one-to-one comparison.")
            return
        left, right = expected.copy(), actual.copy()
        for key in keys:
            left[key], right[key] = left[key].astype(str), right[key].astype(str)
        joined = left[keys + columns].merge(right[keys + columns], on=keys, how="outer", suffixes=("_expected", "_actual"), indicator=True)
        issues, maximum = [], 0.0
        if not joined["_merge"].eq("both").all():
            issues.append({"keys": "missing_or_extra", "rows": joined.loc[joined["_merge"].ne("both"), keys + ["_merge"]].head(5).to_dict("records")})
        for column in columns:
            e, a = joined[f"{column}_expected"], joined[f"{column}_actual"]
            if pd.api.types.is_bool_dtype(expected[column]):
                good = a.map(lambda value: isinstance(value, (bool, np.bool_))) & e.eq(a)
            elif pd.api.types.is_numeric_dtype(expected[column]):
                e, a = pd.to_numeric(e, errors="coerce").astype(float), pd.to_numeric(a, errors="coerce").astype(float)
                good = np.isclose(e, a, atol=TOLERANCE, rtol=RELATIVE_TOLERANCE, equal_nan=True)
                delta = (e - a).abs().dropna()
                maximum = max(maximum, float(delta.max()) if len(delta) else 0.0)
            else:
                good = e.fillna("<missing>").astype(str).eq(a.fillna("<missing>").astype(str))
            if not np.all(good):
                issues.append({"column": column, "rows": joined.loc[~good, keys + [f"{column}_expected", f"{column}_actual"]].head(5).to_dict("records")})
        self.condition(name, not issues, "Independently recomputed rows aligned by aggregation/customer keys.",
                       expected={"rows": len(expected), "columns": columns}, actual={"rows": len(actual)},
                       difference={"max_absolute_numeric": maximum, "issues": issues}, absolute_tolerance=TOLERANCE, relative_tolerance=RELATIVE_TOLERANCE)

    def source(self, purpose, spec, columns):
        matches = [item for item in self.audit["inputs"] if item["purpose"] == purpose]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one input audit entry for {purpose}.")
        info = matches[0]
        path = Path(info["path"])
        if path.suffix.lower() == ".csv":
            raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=spec.get("encoding", "utf-8-sig"))
        else:
            raw = pd.read_excel(path, sheet_name=spec.get("sheet", 0), dtype=str, keep_default_na=False)
        self.condition(f"{purpose}_source_rows", len(raw) == info["rows"], str(path), expected=info["rows"], actual=len(raw))
        duplicates = int(raw.duplicated().sum())
        if self.config["drop_exact_duplicates"]:
            raw = raw.loc[~raw.duplicated()].copy()
            self.condition(f"{purpose}_exact_duplicate_removal", duplicates == info.get("exact_duplicates_removed"),
                           "Independent full-row duplicate count before canonical column mapping.", expected=duplicates, actual=info.get("exact_duplicates_removed"))
        canonical = raw[list(columns.values())].rename(columns={v: k for k, v in columns.items()}).copy()
        for key in ["customer_id", "order_id"]:
            if key in canonical:
                canonical[key] = canonical[key].astype("string").str.strip()
        return canonical

    def clean_money(self, values):
        tolerance = self.config["monetary_zero_tolerance"]
        return values.mask(values.abs().le(tolerance) & values.ne(0), 0.0)

    def round_data(self):
        self.condition("runner_fingerprint", digest(self.runner_path) == self.audit.get("runner_sha256"),
                       "Bundled runner must match the implementation that produced this run; rerun after repairs.",
                       expected=self.audit.get("runner_sha256"), actual=digest(self.runner_path))
        for info in self.audit["inputs"]:
            source = Path(info["path"])
            actual = digest(source) if source.is_file() else None
            self.condition(f"input_fingerprint_{info['purpose']}", actual == info["sha256"], str(source), expected=info["sha256"], actual=actual)
        customers = self.csv("customer_labels.csv")
        self.condition("customer_keys_unique", customers["CustomerID"].notna().all() and customers["CustomerID"].str.strip().ne("").all() and not customers["CustomerID"].duplicated().any(),
                       "customer_labels.csv customer key coverage and uniqueness.", actual={"rows": len(customers), "unique": customers["CustomerID"].nunique()})
        frame = self.source("analysis", self.config["input"], self.config["columns"])
        if self.config["input"]["level"] == "customers":
            f, m = pd.to_numeric(frame["frequency"], errors="coerce"), pd.to_numeric(frame["monetary"], errors="coerce")
            if "recency_days" in frame:
                original = frame["recency_days"]
                recency = pd.to_numeric(original, errors="coerce")
            else:
                original = frame["date"]
                parsed = pd.to_datetime(original, errors="coerce", format="mixed")
                reference = normalize_day(self.config.get("reference_date") or (parsed.max().normalize() + pd.Timedelta(days=1)))
                recency = (reference - parsed.dt.normalize()).dt.days
            nonblank = original.astype("string").fillna("").str.strip().ne("")
            bad = (frame["customer_id"].fillna("").eq("") | ~np.isfinite(f) | f.lt(0) | f.mod(1).ne(0) | ~np.isfinite(m) |
                   (recency.isna() & (f.gt(0) | nonblank)) | (recency.notna() & ~np.isfinite(recency)) | recency.lt(0))
            self.condition("customer_input_invalid_rows", int(bad.sum()) == self.audit["invalid_rows"]["customers"],
                           "Independent required-field, finite-value, integer-frequency and recency validation.", expected=int(bad.sum()), actual=self.audit["invalid_rows"]["customers"])
            self.condition("customer_source_keys_unique", not frame.loc[~bad, "customer_id"].duplicated().any(), "Customer aggregate input requires exactly one row per customer.")
            expected = pd.DataFrame({"CustomerID": frame["customer_id"], "R_days": recency, "Frequency": f, "Monetary": m}).loc[~bad].copy()
        else:
            parsing_audit = {"warnings": []}
            transactions = self.runner.prepare_transactions(frame, self.config, parsing_audit, "analysis_source")
            self.state["transactions"] = transactions
            history = date_window(transactions, self.config.get("analysis_start"), self.config.get("analysis_end"))
            reference = normalize_day(self.config.get("reference_date") or (history["BusinessDate"].max() + pd.Timedelta(days=1)))
            money = history.groupby("customer_id")["Value"].sum()
            orders = independent_orders(history, self.config.get("analysis_start"), self.config.get("analysis_end"))
            count = orders.groupby("customer_id")["order_id"].size()
            last = orders.groupby("customer_id")["OrderDate"].max()
            expected = pd.DataFrame({"Monetary": money, "Frequency": count, "LastDate": last}).reindex(money.index)
            expected["Frequency"] = expected["Frequency"].fillna(0)
            expected["R_days"] = (reference - expected["LastDate"]).dt.days
            expected = expected.reset_index().rename(columns={"customer_id": "CustomerID"})
            self.numbers("transaction_order_deduplication", len(orders), expected["Frequency"].sum(),
                         "Purchase orders independently deduplicated by customer/order, then filtered by effective OrderDate.")
            self.add("transaction_parser_independence", "review", "Date/return/invalid-row governance reuses bundled prepare_transactions; window selection, order deduplication and customer F/M/R aggregation are independent. Shared parser defects require manual review.",
                     actual=parsing_audit.get("transactions", {}).get("analysis_source"))
        expected["Monetary"] = self.clean_money(expected["Monetary"])
        no_recency = expected["R_days"].isna()
        negative = expected["Monetary"].lt(0)
        self.numbers("negative_monetary_audit_count", int((negative & ~no_recency).sum()), self.audit["negative_monetary_customers"], "Negative net monetary count after no-recency exclusion, before configured negative handling.")
        excluded = no_recency | (negative & (self.config["negative_monetary"] == "exclude"))
        excluded_ids = set(expected.loc[excluded, "CustomerID"])
        expected = expected.loc[~excluded, ["CustomerID", "R_days", "Frequency", "Monetary"]].reset_index(drop=True)
        self.table("customer_labels.csv", expected, ["CustomerID"])
        actual_excluded = self.csv("excluded_customers.csv")
        self.condition("excluded_customer_ids", set(actual_excluded["CustomerID"]) == excluded_ids,
                       "Independent no-recency and configured negative-monetary exclusions.", expected=sorted(excluded_ids), actual=sorted(actual_excluded["CustomerID"]))
        self.numbers("excluded_customer_count", [len(excluded_ids), len(excluded_ids)], [len(actual_excluded), self.audit["excluded_customers"]], "Excluded customer table and audit count agree with independently reconstructed eligibility.")
        self.numbers("eligible_customer_audit", len(expected), self.audit["eligible_customers"], "Independent post-governance eligible cohort size.")
        self.numbers("historical_totals", [len(expected), expected["Frequency"].sum(), expected["Monetary"].sum()],
                     [len(customers), self.audit["historical_orders"], self.audit["historical_monetary"]], "Independent customer count, deduplicated order count and signed monetary total.")
        self.condition("historical_negative_values_preserved", self.config["negative_monetary"] == "exclude" or
                       set(expected.loc[expected["Monetary"].lt(0), "CustomerID"]) == set(customers.loc[customers["Monetary"].lt(0), "CustomerID"]),
                       "Legitimate negative customer net amounts are retained under keep/signed_log; only documented tiny residuals are zeroed.")
        self.state.update(customers=customers, independent_customers=expected)

    def round_methods(self):
        customers, config = self.state["customers"], self.config
        rfm, levels = config["rfm"], config["rfm"]["score_levels"]
        scores = {}
        for key, column in [("r", "R_days"), ("f", "Frequency"), ("m", "Monetary")]:
            spec = rfm["scoring"][key]
            spec = {"method": "thresholds", "thresholds": rfm["frequency_thresholds"]} if key == "f" and "frequency_thresholds" in rfm else spec
            values = customers[column]
            if spec.get("method", "percentile") == "percentile":
                ranks = values.rank(method="min", ascending=key != "r")
                expected = ((ranks - 1) / max(len(values) - 1, 1) * levels).apply(np.floor).add(1).clip(upper=levels)
            else:
                expected = values.apply(lambda value: 1 + sum(value > boundary for boundary in spec["thresholds"]))
                if key == "r":
                    expected = levels + 1 - expected
            scores[key] = expected
            self.numbers(f"{key.upper()}_scores_and_direction", expected, customers[f"{key.upper()}_score"],
                         "Independent min-rank percentile / inclusive upper thresholds; fewer R days and larger F/M receive larger scores.")
        fm = (scores["f"] + scores["m"]) / 2
        fm = np.floor(fm) if rfm["fm_rounding"] == "floor" else np.ceil(fm) if rfm["fm_rounding"] == "ceil" else np.floor(fm + .5) if rfm["fm_rounding"] == "round" else fm
        self.numbers("FM_rounding", fm, customers["FM_score"], f"Independent configured FM rounding: {rfm['fm_rounding']}.")
        mode = rfm["segmentation"]["mode"]
        if mode == "shopify":
            self.condition("shopify_five_levels_floor", levels == 5 and rfm["fm_rounding"] == "floor", "Shopify requires five scores and floor((F+M)/2).")
            groups = [SHOPIFY_GRID[int(r) - 1][int(value) - 1] for r, value in zip(scores["r"], fm)]
        elif mode == "generic_matrix":
            cuts = rfm["segmentation"]["cutoffs"]
            band = lambda value: "low" if value <= cuts["low_max"] else "mid" if value <= cuts["mid_max"] else "high"
            groups = [f"Current_{band((r-1)/(levels-1))}__Historical_{band((value-1)/(levels-1))}" for r, value in zip(scores["r"], fm)]
        else:
            rules = rfm["segmentation"]["rules"]
            groups = []
            for r, f, m, value in zip(scores["r"], scores["f"], scores["m"], fm):
                matches = [rule["label"] for rule in rules if all(rule.get(f"{key}_min", 1) <= item <= rule.get(f"{key}_max", levels) for key, item in [("r", r), ("f", f), ("m", m), ("fm", value)])]
                groups.append(matches[0] if len(matches) == 1 else "<invalid-rule-coverage>")
            self.add("custom_full_domain_rule_review", "review", "Observed rows are checked independently; full custom-rule domain and business meaning require review.")
        self.condition("RFM_group_coverage", groups == customers["Customer_group"].tolist() and all(value not in ["Unassigned", "<invalid-rule-coverage>"] for value in groups),
                       "Independent configured group assignment for every eligible customer.", expected=groups, actual=customers["Customer_group"].tolist())
        grid_r = np.repeat(np.arange(1, 6), 5)
        grid_fm = np.tile(np.arange(1, 6), 5)
        actual_grid = self.runner.shopify_groups(grid_r, grid_fm).tolist()
        self.condition("shopify_all_25_cells", actual_grid == sum(SHOPIFY_GRID, []), "Literal independent 5x5 mapping versus bundled Shopify mapping, including cells absent from this cohort.", expected=SHOPIFY_GRID, actual=np.array(actual_grid).reshape(5, 5).tolist())
        transformed = customers[["R_days", "Frequency", "Monetary"]].astype(float).copy()
        for column, info in self.audit["feature_transforms"].items():
            requested = config["kmeans"]["transforms"]
            requested = "auto" if requested == "auto" else requested.get(column, "auto")
            legacy = {"Frequency": "log_frequency", "Monetary": "log_monetary"}.get(column)
            if legacy in config["kmeans"]:
                self.add(f"legacy_transform_precedence_{column}", "review", "Legacy transform precedence depends on the original unmerged configuration; audit.json preserves only the merged configuration. Review original config.")
                requested = info["requested"]
            skew = float(transformed[column].skew())
            mode = ("signed_log" if transformed[column].lt(0).any() else "log1p" if np.isfinite(skew) and skew > 1 else "identity") if requested == "auto" else requested
            self.condition(f"transform_choice_{column}", mode == info["actual"], "Transform recomputed from configuration and historical eligible feature distribution.", expected=mode, actual=info["actual"])
            if mode == "log1p":
                transformed[column] = np.log1p(transformed[column])
            elif mode == "signed_log":
                transformed[column] = np.sign(transformed[column]) * np.log1p(transformed[column].abs())
            elif mode != "identity":
                raise ValueError(f"Unsupported audited transform: {mode}")
        expected_features = transformed.add_prefix("Transformed_")
        expected_features.insert(0, "CustomerID", customers["CustomerID"])
        self.table("kmeans_transformed_features.csv", expected_features, ["CustomerID"])
        scaler = StandardScaler() if config["kmeans"]["scaler"] == "standard" else RobustScaler()
        x = scaler.fit_transform(transformed)
        center = scaler.mean_ if hasattr(scaler, "mean_") else scaler.center_
        self.numbers("historical_scaler_center", center, self.audit["scaler"]["center"], "Recomputed scaler fit uses historical eligible customers only.")
        self.numbers("historical_scaler_scale", scaler.scale_, self.audit["scaler"]["scale"], "Recomputed historical standard/robust scaling.")
        self.condition("historical_fit_scope_record", self.audit["scaler"].get("fit_scope") == "historical eligible cohort only", "Audit scope agrees with independently reproduced historical parameters; chronology also requires source snapshot review.")
        settings = config["kmeans"]
        seeds = list(dict.fromkeys(settings["seeds"]))
        seeds = [seed for seed in seeds if seed != settings["random_state"]]
        self.condition("ARI_seed_protocol", len(seeds) >= 2 and all(isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**32 for seed in seeds),
                       "Deduplicated non-reference seeds; no self comparison.", expected={"minimum": 2}, actual={"seeds": seeds, "count": len(seeds)})
        diagnostics, stability = self.csv("k_diagnostics.csv"), self.csv("candidate_seed_stability.csv")
        requested_k = list(dict.fromkeys(settings["candidates"] + ([settings["k"]] if settings.get("k") is not None else [])))
        valid_k = sorted(k for k in requested_k if isinstance(k, int) and not isinstance(k, bool) and 2 <= k < len(x) and k <= len(np.unique(x, axis=0)))
        self.condition("configured_candidate_K_coverage", diagnostics["K"].tolist() == valid_k,
                       "All and only valid configured candidate/final K values must appear once in ascending order.", expected=valid_k, actual=diagnostics["K"].tolist())
        protocol = self.audit.get("kmeans_seed_protocol", {})
        self.condition("audited_seed_protocol_matches_config", protocol.get("comparison_seeds") == seeds and protocol.get("reference_seed") == settings["random_state"] and protocol.get("comparison_count_per_k") == len(seeds) and protocol.get("n_init") == settings["n_init"],
                       "Audit reference seed, non-reference comparison seeds, comparison count and n_init match the effective configuration.", expected={"reference": settings["random_state"], "seeds": seeds, "n_init": settings["n_init"]}, actual=protocol)
        self.condition("candidate_stability_keys", not stability.duplicated(["K", "Seed"]).any() and not stability["Seed"].eq(stability["ReferenceSeed"]).any(), "Candidate ARI detail has unique K/seed keys and excludes the reference seed.")
        reference_labels, expected_rows, expected_stability = {}, [], []
        with threadpool_limits(limits=4):
            for k in diagnostics["K"]:
                model = KMeans(n_clusters=int(k), random_state=settings["random_state"], n_init=settings["n_init"])
                labels = model.fit_predict(x)
                reference_labels[int(k)] = labels
                try:
                    silhouette = silhouette_score(x, labels, sample_size=settings["silhouette_sample_size"], random_state=settings["random_state"])
                except ValueError:
                    silhouette = np.nan
                ari = []
                for seed in seeds:
                    comparison = KMeans(n_clusters=int(k), random_state=seed, n_init=settings["n_init"]).fit_predict(x)
                    value = adjusted_rand_score(labels, comparison)
                    ari.append(value)
                    expected_stability.append({"K": k, "ReferenceSeed": settings["random_state"], "Seed": seed, "ARI": value})
                sizes = pd.Series(labels).value_counts()
                expected_rows.append({"K": k, "Inertia": model.inertia_, "Silhouette": silhouette, "ARI_mean": np.mean(ari), "ARI_min": np.min(ari), "ARI_max": np.max(ari), "ARI_std": np.std(ari, ddof=0), "ARI_comparisons": len(ari),
                                      "SmallestCluster": int(sizes.min()), "SmallestClusterShare": sizes.min()/len(x), "LargestClusterShare": sizes.max()/len(x), "ClustersBelowMinCustomers": int(sizes.lt(config["cross"]["min_customers"]).sum()), "SilhouetteSampled": settings["silhouette_sample_size"] is not None and settings["silhouette_sample_size"] < len(x)})
        expected_diagnostics = pd.DataFrame(expected_rows, columns=["K", "Inertia", "Silhouette", "ARI_mean", "ARI_min", "ARI_max", "ARI_std", "ARI_comparisons", "SmallestCluster", "SmallestClusterShare", "LargestClusterShare", "ClustersBelowMinCustomers", "SilhouetteSampled"])
        previous = expected_diagnostics["Inertia"].shift(1)
        expected_diagnostics["PreviousK"] = expected_diagnostics["K"].shift(1)
        expected_diagnostics["InertiaDrop"] = previous - expected_diagnostics["Inertia"]
        expected_diagnostics["InertiaRelativeDrop"] = expected_diagnostics["InertiaDrop"] / previous.where(previous.gt(0))
        self.table("k_diagnostics.csv", expected_diagnostics, ["K"])
        self.table("candidate_seed_stability.csv", pd.DataFrame(expected_stability, columns=["K", "ReferenceSeed", "Seed", "ARI"]), ["K", "Seed"])
        selected = settings.get("k")
        if selected is None:
            self.add("final_K", "review", "candidate_only: valid exploration, but no final selected K and no final delivery claim.")
        else:
            self.condition("final_K_valid", selected in reference_labels and 2 <= selected < len(x) and selected <= len(np.unique(x, axis=0)), "Configured final K must be represented and valid for cohort and distinct vectors.", actual=selected)
            self.numbers("final_cluster_assignments", reference_labels[selected], customers["Cluster"], "Independent reference-seed fit at selected K.")
            selected_details = pd.DataFrame(expected_stability).loc[lambda values: values["K"].eq(selected)].drop(columns="K")
            self.table("seed_stability.csv", selected_details, ["Seed"])
        if len(diagnostics) < 2:
            self.add("elbow_evidence", "review", "Fewer than two valid K values: reproduction is allowed; elbow evidence cannot justify final selection.", actual=diagnostics["K"].tolist())
        self.add("K_selection_reasoning", "review", "Metric arithmetic is checked; final K requires a recorded human/assistant review combining elbow, silhouette, ARI and customer profiles. No metric automatically proves optimal K.", actual=self.audit.get("k_selection_evidence"))
        holdout = config.get("holdout")
        if holdout:
            valid = normalize_day(holdout["start"]) <= normalize_day(holdout["end"]) and normalize_day(self.audit["reference_date"]) <= normalize_day(holdout["start"])
            if config.get("analysis_end"):
                valid &= normalize_day(config["analysis_end"]) < normalize_day(holdout["start"])
            self.condition("holdout_window_order", valid, "History/reference date precedes the holdout; start does not follow end.")
            if str(self.audit.get("holdout_assessment", {}).get("status", "")).startswith("qualified"):
                self.add("holdout_lifecycle_qualification", "review", "Runner qualified this holdout because order lifecycle crosses boundaries/reuses keys; do not claim clean future validation without snapshot review.", actual=self.audit["holdout_assessment"])
            if config["input"]["level"] == "customers":
                self.add("aggregate_snapshot_provenance", "review", "Aggregate input contains no full order chronology; historical snapshot timing cannot be established from recency alone.")
        self.state.update(reference_labels=reference_labels, transformed=transformed)

    def independent_future(self, monthly=False):
        holdout = self.config["holdout"]
        if "future_transactions" not in self.state:
            if "input" in holdout:
                raw = self.source("holdout", holdout["input"], holdout.get("columns", self.config["columns"]))
                transactions = self.runner.prepare_transactions(raw, self.config, {"warnings": []}, "holdout_source")
            else:
                transactions = self.state["transactions"]
            self.state["future_transactions"] = date_window(transactions, holdout["start"], holdout["end"])
            self.add("future_parser_independence", "review", "Return/date governance shares the bundled parser; holdout window, order deduplication, monetary aggregation, zero-fill and cohort joins below are independent.")
        future = self.state["future_transactions"].copy()
        orders = independent_orders(future, holdout["start"], holdout["end"])
        keys = ["customer_id"]
        if monthly:
            future["Month"] = future["date"].dt.strftime("%Y-%m")
            orders["Month"] = orders["OrderDate"].dt.strftime("%Y-%m")
            keys.append("Month")
        money = future.groupby(keys)["Value"].sum().rename("Monetary")
        counts = orders.groupby(keys)["order_id"].size().rename("Frequency")
        values = pd.concat([money, counts], axis=1).reset_index().rename(columns={"customer_id": "CustomerID"})
        customers = self.state["customers"]
        label_columns = ["CustomerID", "Customer_group"] + (["Cluster"] if "Cluster" in customers else [])
        frozen = customers[label_columns].copy()
        join_keys = ["CustomerID"]
        if monthly:
            months = pd.DataFrame({"Month": pd.period_range(holdout["start"], holdout["end"], freq="M").astype(str)})
            frozen = frozen.merge(months, how="cross")
            join_keys.append("Month")
        joined = frozen.merge(values, on=join_keys, how="left", validate="one_to_one")
        joined[["Frequency", "Monetary"]] = joined[["Frequency", "Monetary"]].fillna(0)
        joined["Monetary"] = self.clean_money(joined["Monetary"])
        joined["Frequency"] = joined["Frequency"].astype(int)
        joined["Purchased"] = joined["Frequency"].gt(0).astype(int)
        return joined

    @staticmethod
    def oracle(evaluation, ids):
        top = evaluation.sort_values(["Monetary", "CustomerID"], ascending=[False, True], kind="mergesort").head(len(ids))
        denominator = float(top["Monetary"].sum())
        numerator = float(evaluation.set_index("CustomerID").loc[ids, "Monetary"].sum())
        return {"OracleSameNMonetary": denominator, "OracleMonetaryCapture": numerator / denominator if denominator > 0 else np.nan,
                "OracleCustomerOverlapShare": len(set(ids) & set(top["CustomerID"])) / len(ids)}

    def check_baselines(self, future):
        labels = self.state["customers"]
        group_columns = ["Customer_group"] + (["Cluster"] if "Cluster" in labels else [])
        customers = self.state["independent_customers"].merge(labels[["CustomerID"] + group_columns], on="CustomerID", validate="one_to_one")
        settings = self.config["baseline"]
        self.condition("baseline_business_budget_flag", isinstance(settings["budget_is_business"], bool),
                       "Business-budget evidence must use a JSON boolean, not string or numeric truthiness.")
        plans = []
        for index, budget in enumerate(settings["budgets"], 1):
            if isinstance(budget, (int, float)) and not isinstance(budget, bool) and 0 < budget <= 1:
                kind, value = "fraction", float(budget)
                count = max(1, int(np.floor(len(customers) * value)))
            elif isinstance(budget, dict) and set(budget) == {"customers"} and isinstance(budget["customers"], int) and not isinstance(budget["customers"], bool) and 1 <= budget["customers"] <= len(customers):
                kind, value, count = "customers", budget["customers"], budget["customers"]
            else:
                raise ValueError("Invalid baseline budget in audited configuration.")
            plans.append({"BudgetID": f"b{index:02d}", "BudgetType": kind, "RequestedBudget": value,
                          "SelectedCustomers": count, "ActualCustomerShare": count / len(customers)})
        if not plans:
            raise ValueError("Baseline budgets must not be empty.")
        stable = customers.sort_values("CustomerID", kind="mergesort")
        strategies, bases = {}, {}
        for name, feature, ascending in [("baseline_monetary", "Monetary", False), ("baseline_frequency", "Frequency", False), ("baseline_recency", "R_days", True)]:
            strategies[name] = stable.sort_values(feature, ascending=ascending, kind="mergesort")["CustomerID"].tolist()
            bases[name] = f"historical {feature}; CustomerID breaks ties"
        strategies["baseline_random"] = np.random.default_rng(settings["random_seed"]).permutation(stable["CustomerID"].to_numpy()).tolist()
        bases["baseline_random"] = f"seeded random permutation, seed={settings['random_seed']}"
        for kind, column in [("rfm", "Customer_group"), ("cluster", "Cluster")]:
            if column not in customers:
                continue
            priorities, rule = settings.get(f"{kind}_priority"), settings.get("historical_priority_rule")
            rule = rule.get(kind) if isinstance(rule, dict) else rule
            basis = "configured group priorities; CustomerID breaks within-group ties"
            if priorities is None and rule:
                feature, ascending = {"monetary_desc": ("Monetary", False), "frequency_desc": ("Frequency", False), "recency_asc": ("R_days", True)}[rule]
                means = customers.assign(Label=customers[column].astype(str)).groupby("Label")[feature].mean().reset_index()
                priorities = means.sort_values([feature, "Label"], ascending=[ascending, True], kind="mergesort")["Label"].tolist()
                basis = f"historical group mean {rule}; CustomerID breaks within-group ties"
            if priorities is not None:
                mapping = {str(label): rank for rank, label in enumerate(priorities)}
                ordered = customers.assign(Priority=customers[column].astype(str).map(mapping)).sort_values(["Priority", "CustomerID"], kind="mergesort")
                strategies[f"{kind}_priority"] = ordered["CustomerID"].tolist()
                bases[f"{kind}_priority"] = basis
        ranking_rows = [{"Strategy": strategy, "CustomerID": customer, "Rank": rank, "RankingBasis": bases[strategy]}
                        for strategy, ids in strategies.items() for rank, customer in enumerate(ids, 1)]
        self.table("baseline_rankings.csv", pd.DataFrame(ranking_rows), ["Strategy", "CustomerID"])
        rankings = self.csv("baseline_rankings.csv")
        self.condition("oracle_never_strategy", not rankings["Strategy"].str.contains("oracle", case=False).any(), "Oracle is a post-hoc monetary/ID-overlap diagnostic, absent from executable strategy rankings.")
        evaluation = future if future is not None else customers.assign(Purchased=customers["Frequency"].gt(0).astype(int))
        scope = "holdout_same_cohort_descriptive" if future is not None else "in_sample_descriptive"
        budget_basis = "business" if settings["budget_is_business"] else "scenario"
        expected_audit = {"evaluation_scope": scope, "business_budget_defined": settings["budget_is_business"],
                          "budget_basis": budget_basis, "budgets": plans}
        actual_audit = {key: self.audit.get("baseline", {}).get(key) for key in expected_audit}
        self.condition("baseline_audit_scope_and_budgets", expected_audit == actual_audit,
                       "Evaluation scope, business/scenario budget and independently reconstructed budget plans agree with the audit.",
                       expected=expected_audit, actual=actual_audit)
        indexed = evaluation.set_index("CustomerID")
        historical = customers.set_index("CustomerID")
        total_m, total_f, total_buyers = evaluation["Monetary"].sum(), evaluation["Frequency"].sum(), evaluation["Purchased"].sum()
        rows, structures = [], []
        for plan in plans:
            n = plan["SelectedCustomers"]
            for strategy, ids in strategies.items():
                selected_ids = ids[:n]
                selected, current = indexed.loc[selected_ids], historical.loc[selected_ids]
                money, orders, buyers = selected["Monetary"].sum(), selected["Frequency"].sum(), selected["Purchased"].sum()
                row = {**plan, "Strategy": strategy, "EvaluationScope": scope, "BusinessBudgetDefined": settings["budget_is_business"],
                       "BudgetBasis": budget_basis,
                       "CohortCustomers": len(customers), "TotalMonetary": money, "AvgMonetary": money / n,
                       "MonetaryCoverage": money / total_m if total_m > 0 else np.nan, "TotalOrders": orders,
                       "OrderCoverage": orders / total_f if total_f else np.nan, "PurchaseRate": buyers / n, "PurchasingCustomers": buyers,
                       "PurchasingCustomerCoverage": buyers / total_buyers if total_buyers else np.nan,
                       "CurrentMonetary": current["Monetary"].sum(), "CurrentOrders": current["Frequency"].sum(),
                       "MechanicalCurrentMonetaryUpperBound": future is None and strategy == "baseline_monetary"}
                if future is not None:
                    row.update(self.oracle(evaluation, selected_ids))
                rows.append(row)
                for grouping in group_columns:
                    for label, members in current.groupby(grouping, observed=True, sort=True):
                        group_size = int(customers[grouping].eq(label).sum())
                        structures.append({**plan, "Strategy": strategy, "Grouping": grouping, "Group": label,
                                           "Customers": len(members), "SelectedCustomerShare": len(members) / n,
                                           "GroupCustomerCoverage": len(members) / group_size,
                                           "CurrentMonetary": members["Monetary"].sum(), "CurrentOrders": members["Frequency"].sum()})
        metrics = pd.DataFrame(rows)
        self.table("baseline_metrics.csv", metrics, ["BudgetID", "Strategy"])
        self.table("baseline_structure.csv", pd.DataFrame(structures), ["BudgetID", "Strategy", "Grouping", "Group"])
        if future is None:
            self.condition("no_oracle_without_future", not any(column.startswith("Oracle") for column in self.csv("baseline_metrics.csv")), "No post-hoc future monetary oracle is computed without holdout data.")
        comparisons = []
        for plan in plans:
            group = metrics.loc[metrics["BudgetID"].eq(plan["BudgetID"])]
            records = group.set_index("Strategy")
            for strategy in [name for name in records.index if name.endswith("_priority")]:
                for baseline in [name for name in records.index if name.startswith("baseline_")]:
                    row = {**plan, "Strategy": strategy, "Baseline": baseline, "EvaluationScope": scope,
                           "BusinessBudgetDefined": settings["budget_is_business"], "BudgetBasis": budget_basis,
                           "Interpretation": "descriptive holdout comparison; no causal or significance claim" if future is not None else "current-window coverage only; not predictive lift"}
                    for metric in ["MonetaryCoverage", "AvgMonetary", "PurchaseRate", "OrderCoverage", "PurchasingCustomerCoverage"]:
                        row[f"Delta{metric}"] = records.loc[strategy, metric] - records.loc[baseline, metric]
                    comparisons.append(row)
        expected_comparisons = pd.DataFrame(comparisons) if comparisons else pd.DataFrame(columns=["BudgetID", "Strategy", "Baseline", "EvaluationScope"])
        self.table("baseline_comparisons.csv", expected_comparisons, ["BudgetID", "Strategy", "Baseline"])
        if not self.config["baseline"]["budget_is_business"]:
            self.add("business_budget_evidence", "review", "Configured budgets are scenario sensitivity checks; deployment advantage/ROI needs a real resource constraint and intervention evidence.")

    def round_results(self):
        customers = self.state["customers"]
        self.table("rfm_summary.csv", independent_profile(customers, "Customer_group"), ["Customer_group"])
        if "Cluster" in customers:
            self.table("cluster_summary.csv", independent_profile(customers, "Cluster"), ["Cluster"])
            joint = independent_profile(customers, ["Cluster", "Customer_group"])
            joint["RowCustomerShare"] = joint["Customers"] / joint.groupby("Cluster")["Customers"].transform("sum")
            joint["ColumnCustomerShare"] = joint["Customers"] / joint.groupby("Customer_group")["Customers"].transform("sum")
            self.table("cross_history_summary.csv", joint, ["Cluster", "Customer_group"])
            counts = customers.groupby(["Cluster", "Customer_group"], observed=True).size().unstack(fill_value=0)
            for name, expected in [("cross_counts.csv", counts), ("cross_row_share.csv", counts.div(counts.sum(axis=1), axis=0)), ("cross_column_share.csv", counts.div(counts.sum(axis=0), axis=1))]:
                self.table(name, expected.reset_index(), ["Cluster"])
        candidate_profiles, candidate_crosses = [], []
        for k, labels in self.state["reference_labels"].items():
            frame = customers.assign(Cluster=labels)
            profile = independent_profile(frame, "Cluster")
            profile.insert(0, "K", k)
            candidate_profiles.append(profile)
            cross = independent_profile(frame, ["Cluster", "Customer_group"])
            cross.insert(0, "K", k)
            cross["RowCustomerShare"] = cross["Customers"] / cross.groupby("Cluster")["Customers"].transform("sum")
            cross["ColumnCustomerShare"] = cross["Customers"] / cross.groupby("Customer_group")["Customers"].transform("sum")
            candidate_crosses.append(cross)
        if candidate_profiles:
            self.table("candidate_profiles.csv", pd.concat(candidate_profiles, ignore_index=True), ["K", "Cluster"])
            self.table("candidate_cross_summary.csv", pd.concat(candidate_crosses, ignore_index=True), ["K", "Cluster", "Customer_group"])
        future = None
        if self.config.get("holdout"):
            future = self.independent_future()
            self.table("customer_future_metrics.csv", future.rename(columns={"Frequency": "FutureFrequency", "Monetary": "FutureMonetary", "Purchased": "Repurchased"}), ["CustomerID"])
            self.table("rfm_future_summary.csv", independent_future_summary(future, ["Customer_group"]), ["Customer_group"])
            self.condition("future_same_cohort_zero_fill", len(future) == len(customers) and set(future["CustomerID"]) == set(customers["CustomerID"]), "Independent eligible-cohort left join preserves zero-future customers and excludes new future customers.")
            self.numbers("future_audit_totals", [len(future), future["Frequency"].sum(), future["Monetary"].sum(), future["Purchased"].sum()],
                         [self.audit["future_cohort"][key] for key in ["customers", "orders", "monetary", "repurchased"]], "Independent same-cohort future totals, retaining signed net amounts.")
            if "Cluster" in customers:
                self.table("cluster_future_summary.csv", independent_future_summary(future, ["Cluster"]), ["Cluster"])
                joint = independent_future_summary(future, ["Cluster", "Customer_group"])
                oracle_rows = []
                for row in joint.to_dict("records"):
                    ids = future.loc[future["Cluster"].eq(row["Cluster"]) & future["Customer_group"].eq(row["Customer_group"]), "CustomerID"].tolist()
                    oracle_rows.append(self.oracle(future, ids))
                joint = pd.concat([joint, pd.DataFrame(oracle_rows)], axis=1)
                self.table("cross_future_summary.csv", joint, ["Cluster", "Customer_group"])
            if self.config["holdout"].get("monthly", True):
                monthly = self.independent_future(monthly=True)
                expected = monthly.rename(columns={"Frequency": "MonthlyOrders", "Monetary": "MonthlyMonetary", "Purchased": "PurchasedThisMonth"})
                self.table("customer_monthly_metrics.csv", expected, ["CustomerID", "Month"])
                self.table("rfm_monthly_summary.csv", independent_future_summary(monthly, ["Month", "Customer_group"], True), ["Month", "Customer_group"])
                if "Cluster" in customers:
                    self.table("cluster_monthly_summary.csv", independent_future_summary(monthly, ["Month", "Cluster"], True), ["Month", "Cluster"])
                    self.table("cross_monthly_summary.csv", independent_future_summary(monthly, ["Month", "Cluster", "Customer_group"], True), ["Month", "Cluster", "Customer_group"])
                self.numbers("monthly_full_window_totals", [future["Frequency"].sum(), future["Monetary"].sum()], [monthly["Frequency"].sum(), monthly["Monetary"].sum()], "Sum of monthly order and signed monetary totals equals the whole holdout same-cohort totals.")
        self.check_baselines(future)
        png = self.run / "k_selection_diagnostics.png"
        self.condition("K_diagnostics_image", png.is_file() and png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "Expected three-panel K diagnostics image exists and has a PNG signature.", actual={"file": str(png), "bytes": png.stat().st_size if png.exists() else 0})
        self.check_claims()
        current = {str(path.relative_to(self.run)): digest(path) for path in sorted(self.run.rglob("*")) if path.is_file()}
        self.condition("run_artifacts_unchanged_during_checks", current == self.artifacts, "Run fingerprints before and after the three rounds must agree; checking never edits run artifacts.")
        for info in self.audit["inputs"]:
            source = Path(info["path"])
            self.condition(f"input_fingerprint_after_checks_{info['purpose']}", source.is_file() and digest(source) == info["sha256"], "Recheck input fingerprint after all rounds to detect input mutation during validation.", expected=info["sha256"], actual=digest(source) if source.is_file() else None)

    def check_claims(self):
        if self.claims is None:
            self.add("reported_numeric_claims", "review", "No --claims file supplied: narrative/report numbers and conclusions have not been automatically cleared.")
            return
        document = json.loads(self.claims.read_text(encoding="utf-8-sig"))
        claims = document.get("claims", []) if isinstance(document, dict) else document
        if not isinstance(claims, list) or not claims:
            self.add("reported_numeric_claims", "fail", "Claims must be a nonempty list or {claims: [...]}, not an empty template.")
            return
        for claim in claims:
            try:
                path = (self.run / claim["file"]).resolve()
                if not path.is_relative_to(self.run) or path.suffix.lower() != ".csv":
                    raise ValueError("Claim file must be a CSV inside --run.")
                frame = read_artifact_csv(path)
                for column, value in claim.get("filter", {}).items():
                    frame = frame.loc[frame[column].eq(value)]
                column, aggregation = claim["column"], claim["aggregation"]
                if aggregation == "count":
                    actual = len(frame) if column == "*" else int(frame[column].count())
                elif aggregation == "value":
                    if len(frame) != 1:
                        raise ValueError("value requires exactly one filtered row.")
                    actual = float(frame[column].iloc[0])
                elif aggregation in ["sum", "mean"]:
                    values = pd.to_numeric(frame[column], errors="raise")
                    actual = float(values.sum(min_count=1) if aggregation == "sum" else values.mean())
                else:
                    raise ValueError("Supported aggregations: sum, mean, count, value.")
                tolerance = claim.get("tolerance", TOLERANCE)
                if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
                    raise ValueError("Claim tolerance must be finite and nonnegative.")
                relative_tolerance = claim.get("relative_tolerance", 0)
                if not isinstance(relative_tolerance, (int, float)) or not np.isfinite(relative_tolerance) or relative_tolerance < 0:
                    raise ValueError("Claim relative_tolerance must be finite and nonnegative.")
                self.numbers(f"claim_{claim['id']}", claim["expected"], actual, {"file": claim["file"], "sha256": digest(path), "filter": claim.get("filter", {}), "column": column, "aggregation": aggregation, "matched_rows": len(frame)}, tolerance=tolerance, relative_tolerance=relative_tolerance)
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.add(f"claim_{claim.get('id', '<missing>')}", "fail", str(error))
        self.add("narrative_causality_and_source_truth", "review", "Numeric claims are checked against artifacts. Source authenticity, business interpretation and causal claims still require human evidence; fingerprints prove file consistency only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Existing bundled runner output directory.")
    parser.add_argument("--output", required=True, type=Path, help="New independent check directory; overwrite is refused.")
    parser.add_argument("--claims", type=Path, help="Optional numeric claims JSON.")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(run):
        parser.error("--output must be a new directory outside --run; previous evidence cannot be overwritten.")
    if not (run / "audit.json").is_file():
        parser.error("--run must contain bundled runner audit.json.")
    try:
        checks = Checks(run, output, args.claims.resolve() if args.claims else None)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    rounds = []
    for number, label, method in [(1, "data", checks.round_data), (2, "methods", checks.round_methods), (3, "results", checks.round_results)]:
        checks.items = []
        try:
            method()
        except Exception as error:
            checks.add(f"round_{number}_execution", "fail", f"{type(error).__name__}: {error}; this round did not complete.")
        result = {"round": number, "focus": label, "status": overall(checks.items), "checks": checks.items}
        save_json(output / f"round_{number}_{label}.json", result)
        rounds.append(result)
    flattened = [item for result in rounds for item in result["checks"]]
    summary = {"status": overall(flattened), "run": str(run), "check_directory": str(output),
               "rounds": [{"round": item["round"], "focus": item["focus"], "status": item["status"], "checks": len(item["checks"])} for item in rounds],
               "counts": {status: sum(item["status"] == status for item in flattened) for status in ["pass", "review", "fail"]},
               "input_fingerprints": [{"purpose": info["purpose"], "path": info["path"], "recorded_sha256": info["sha256"], "current_sha256": digest(Path(info["path"])) if Path(info["path"]).is_file() else None} for info in checks.audit["inputs"]],
               "run_artifact_fingerprints": checks.artifacts, "self_check_sha256": digest(Path(__file__)),
               "claims_fingerprint": {"path": str(checks.claims), "sha256": digest(checks.claims)} if checks.claims and checks.claims.is_file() else None,
               "machine_checks_passed": overall(flattened) == "pass",
               "interpretation": "Fail takes priority over review, then pass. Only all-pass permits saying machine checks passed. This does not authenticate sources or establish business causality. Candidate-only exploration remains allowed; final delivery requires explicit K and review evidence. After any repair, run all three rounds into a new check directory."}
    save_json(output / "summary.json", summary)
    lines = ["# Three-round self-check", "", f"Overall: **{summary['status']}**", "", f"Run: `{run}`", "", "| Round | Focus | Status | Checks |", "| --- | --- | --- | --- |"]
    lines += [f"| {item['round']} | {item['focus']} | {item['status']} | {item['checks']} |" for item in summary["rounds"]]
    lines += ["", "Items requiring attention:", ""]
    lines += [f"- [{item['status']}] {item['check']}: {json.dumps(json_value(item['evidence']), ensure_ascii=False)}" for item in flattened if item["status"] != "pass"]
    lines += ["", summary["interpretation"], ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "counts": summary["counts"], "output": str(output)}, ensure_ascii=False))
    return 1 if summary["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
