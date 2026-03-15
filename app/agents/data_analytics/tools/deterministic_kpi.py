import math
import os
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np


NUMERIC_COLS = ("int64", "float64", "int32", "float32", "int16", "float16")


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
        if candidate in df.columns:
            return candidate
    # Heuristic: first datetime-like column
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def _guess_group_columns(df: pd.DataFrame) -> List[str]:
    # Prefer common categorical columns for grouping
    candidates = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            # Skip columns that are too high-cardinality (e.g., IDs)
            unique_ratio = df[col].nunique() / max(1, len(df))
            if unique_ratio <= 0.5:
                candidates.append(col)
    return candidates[:2]


def _guess_metric_columns(df: pd.DataFrame) -> List[str]:
    metrics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Drop obvious identifiers if any numeric column has mostly unique values
    filtered = []
    for col in metrics:
        unique_ratio = df[col].nunique() / max(1, len(df))
        if unique_ratio < 0.95:
            filtered.append(col)
    return filtered[:3] if filtered else metrics[:3]


def detect_pii_columns(df: pd.DataFrame) -> List[str]:
    pii_keywords = ["email", "phone", "ssn", "aadhaar", "pan", "credit", "card", "password", "token"]
    cols = []
    for col in df.columns:
        if any(k in col.lower() for k in pii_keywords):
            cols.append(col)
    return cols


def redact_pii(df: pd.DataFrame) -> pd.DataFrame:
    df_local = df.copy()
    pii_cols = detect_pii_columns(df_local)
    for col in pii_cols:
        try:
            df_local[col] = "***REDACTED***"
        except Exception:
            pass
    return df_local


def auto_date_cutoff(df: pd.DataFrame, max_future_days: int = 30) -> pd.DataFrame:
    date_col = _find_date_column(df)
    if not date_col:
        return df
    try:
        df_local = df.copy()
        df_local[date_col] = pd.to_datetime(df_local[date_col], errors="coerce")
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        cutoff = today + pd.Timedelta(days=max_future_days)
        df_local = df_local[df_local[date_col].isna() | (df_local[date_col] <= cutoff)]
        return df_local
    except Exception:
        return df


def compute_deterministic_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available for summary."
    group_cols = _guess_group_columns(df)
    metric_cols = _guess_metric_columns(df)
    if not group_cols or not metric_cols:
        return "Summary generated, but dataset lacks clear group or metric columns."

    summary_lines = []
    try:
        grouped = df.groupby(group_cols, as_index=False)[metric_cols].sum()
        # pick top group by first metric
        metric = metric_cols[0]
        top = grouped.sort_values(metric, ascending=False).head(3)
        for _, row in top.iterrows():
            group_name = " | ".join([str(row[c]) for c in group_cols])
            summary_lines.append(f"Top group {group_name} with {metric}={row[metric]:.2f}.")
    except Exception:
        return "Summary generated, but grouping failed due to schema complexity."

    return " ".join(summary_lines) if summary_lines else "Summary generated, but no dominant groups found."


def _parse_unique_id(value: str) -> Tuple[str, str]:
    if not value:
        return ("Unknown", "Unknown")
    if "_" in value:
        head, tail = value.split("_", 1)
        return head.strip() or "Unknown", tail.strip() or "Unknown"
    return ("All", value)


def load_forecast_dataframe(forecast_csv_url: str) -> pd.DataFrame:
    if not forecast_csv_url:
        return pd.DataFrame()
    try:
        rel = forecast_csv_url.replace("/api/v1/exports/", "")
        export_dir = os.path.join(os.getcwd(), "data", "exports")
        path = os.path.join(export_dir, rel)
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_forecast_table(
    forecast_df: pd.DataFrame,
    metric_label: str = "units_sold",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if forecast_df.empty:
        return rows
    if "unique_id" in forecast_df.columns:
        for _, r in forecast_df.iterrows():
            region, product = _parse_unique_id(str(r.get("unique_id", "")))
            rows.append({
                "region": region,
                "product": product,
                "date": str(r.get("ds", ""))[:10],
                "metric": metric_label,
                "forecast_value": float(r.get("y_forecast", 0.0)),
                "lower_ci": None,
                "upper_ci": None
            })
        return rows
    if "ds" in forecast_df.columns and "yhat" in forecast_df.columns:
        for _, r in forecast_df.iterrows():
            rows.append({
                "region": "All",
                "product": "All",
                "date": str(r.get("ds", ""))[:10],
                "metric": metric_label,
                "forecast_value": float(r.get("yhat", 0.0)),
                "lower_ci": None,
                "upper_ci": None
            })
    return rows


def compute_per_region_summaries(
    forecast_table: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not forecast_table:
        return []

    df = pd.DataFrame(forecast_table)
    if df.empty:
        return []
    try:
        df["forecast_value"] = pd.to_numeric(df["forecast_value"], errors="coerce")
        df = df.dropna(subset=["forecast_value"])
        if df.empty:
            return []
        summaries: List[Dict[str, Any]] = []
        for region, region_df in df.groupby("region"):
            top_products = []
            for product, prod_df in region_df.groupby("product"):
                prod_df = prod_df.sort_values("date")
                start = prod_df["forecast_value"].iloc[0]
                end = prod_df["forecast_value"].iloc[-1]
                if start == 0:
                    delta_pct = 0.0
                else:
                    delta_pct = ((end - start) / abs(start)) * 100.0
                direction = "up" if delta_pct > 0 else "down" if delta_pct < 0 else "flat"
                top_products.append({
                    "product": str(product),
                    "forecast_delta_percent": float(delta_pct),
                    "direction": direction,
                    "reason": f"Forecast moved from {start:.2f} to {end:.2f}."
                })
            # Sort by absolute delta and keep top 3
            top_products = sorted(top_products, key=lambda x: abs(x.get("forecast_delta_percent", 0)), reverse=True)[:3]
            overall_direction = "flat"
            if top_products:
                avg_delta = sum(p["forecast_delta_percent"] for p in top_products) / len(top_products)
                overall_direction = "up" if avg_delta > 0 else "down" if avg_delta < 0 else "flat"
            summaries.append({
                "region": str(region),
                "top_products": top_products,
                "overall_region_trend": overall_direction,
                "notes": f"{len(top_products)} products evaluated."
            })
        return summaries
    except Exception:
        return []


def _safe_sum(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    try:
        return float(pd.to_numeric(df[col], errors="coerce").sum())
    except Exception:
        return None


def _safe_mean(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    try:
        return float(pd.to_numeric(df[col], errors="coerce").mean())
    except Exception:
        return None


def _format_value(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value:,.0f}"
    if abs(value) >= 1_000:
        return f"{value:,.2f}"
    return f"{value:.2f}"


def _compute_growth(df: pd.DataFrame, date_col: str, value_col: str) -> Tuple[Optional[float], str]:
    try:
        df_local = df[[date_col, value_col]].copy()
        df_local[date_col] = pd.to_datetime(df_local[date_col], errors="coerce")
        df_local[value_col] = pd.to_numeric(df_local[value_col], errors="coerce")
        df_local = df_local.dropna(subset=[date_col, value_col])
        if df_local.empty:
            return None, ""
        df_local = df_local.sort_values(date_col)
        # Aggregate by date (daily)
        df_daily = df_local.groupby(date_col, as_index=False)[value_col].sum()
        if len(df_daily) < 2:
            return None, ""
        last_val = df_daily[value_col].iloc[-1]
        prev_val = df_daily[value_col].iloc[-2]
        if prev_val == 0:
            return None, ""
        delta = ((last_val - prev_val) / abs(prev_val)) * 100.0
        return float(delta), "day_over_day"
    except Exception:
        return None, ""


def compute_basic_kpis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    kpis: List[Dict[str, Any]] = []
    if df.empty:
        return kpis

    # Common business columns (if present)
    totals = {
        "Units_Sold": _safe_sum(df, "Units_Sold"),
        "Revenue": _safe_sum(df, "Revenue"),
        "Marketing_Spend": _safe_sum(df, "Marketing_Spend"),
    }
    for title, val in totals.items():
        if val is not None:
            kpis.append({
                "title": f"Total {title.replace('_', ' ')}",
                "value": _format_value(val),
                "trend": "",
                "direction": "flat",
                "delta_percent": 0.0,
                "period": "total"
            })

    # Average revenue per unit
    if totals.get("Revenue") is not None and totals.get("Units_Sold"):
        avg = totals["Revenue"] / totals["Units_Sold"]
        kpis.append({
            "title": "Avg Revenue per Unit",
            "value": _format_value(avg),
            "trend": "",
            "direction": "flat",
            "delta_percent": 0.0,
            "period": "total"
        })

    # Top product / region by revenue if columns exist
    if "Product_Category" in df.columns and "Revenue" in df.columns:
        try:
            top_prod = df.groupby("Product_Category", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False).head(1)
            if not top_prod.empty:
                kpis.append({
                    "title": "Top Product by Revenue",
                    "value": str(top_prod.iloc[0]["Product_Category"]),
                    "trend": "",
                    "direction": "flat",
                    "delta_percent": 0.0,
                    "period": "total"
                })
        except Exception:
            pass

    if "Region" in df.columns and "Revenue" in df.columns:
        try:
            top_region = df.groupby("Region", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False).head(1)
            if not top_region.empty:
                kpis.append({
                    "title": "Top Region by Revenue",
                    "value": str(top_region.iloc[0]["Region"]),
                    "trend": "",
                    "direction": "flat",
                    "delta_percent": 0.0,
                    "period": "total"
                })
        except Exception:
            pass

    # Growth KPI (if date column and Revenue exists)
    date_col = _find_date_column(df)
    if date_col and "Revenue" in df.columns:
        delta, period = _compute_growth(df, date_col, "Revenue")
        if delta is not None:
            direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
            kpis.append({
                "title": "Revenue Growth",
                "value": f"{delta:.2f}%",
                "trend": "",
                "direction": direction,
                "delta_percent": float(delta),
                "period": period
            })

    return kpis


def compute_kpi_explanations(kpis: List[Dict[str, Any]]) -> Dict[str, str]:
    explanations = {}
    for k in kpis:
        title = k.get("title", "")
        if title:
            explanations[title] = f"{title} computed deterministically from dataset aggregates."
    return explanations


def compute_metadata(df: pd.DataFrame, merge_note: str = "") -> Dict[str, Any]:
    if df.empty:
        return {
            "rows_processed": 0,
            "columns_used": [],
            "date_range": "",
            "missingness_percent": 0.0,
            "merge_note": merge_note
        }

    missingness = float(df.isna().mean().mean() * 100.0)
    date_col = _find_date_column(df)
    date_range = ""
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            if dates.notna().any():
                date_range = f"{dates.min().date()} to {dates.max().date()}"
        except Exception:
            date_range = ""

    return {
        "rows_processed": int(len(df)),
        "columns_used": [str(c) for c in df.columns],
        "date_range": date_range,
        "missingness_percent": round(missingness, 2),
        "merge_note": merge_note
    }


def compute_segments(df: pd.DataFrame) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if df.empty:
        return segments
    if "Region" in df.columns and "Product_Category" in df.columns:
        try:
            grouped = df.groupby(["Region", "Product_Category"], as_index=False)["Revenue"].sum()
            top = grouped.sort_values("Revenue", ascending=False).head(5)
            for _, row in top.iterrows():
                segments.append({
                    "segment": f"{row['Region']} | {row['Product_Category']}",
                    "revenue": float(row["Revenue"])
                })
        except Exception:
            pass
    return segments


def compute_time_windows(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    return {}


def compute_governance_checks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    if df.empty:
        return checks

    # Missingness
    missingness = float(df.isna().mean().mean() * 100.0)
    if missingness > 10:
        checks.append({
            "check": "missingness",
            "status": "warn",
            "details": f"Missing values ~{missingness:.1f}%"
        })

    # Duplicate rows
    try:
        dup = df.duplicated().sum()
        if dup > 0:
            checks.append({
                "check": "duplicates",
                "status": "warn",
                "details": f"{int(dup)} duplicate rows detected"
            })
    except Exception:
        pass

    # Date continuity (if date column exists)
    date_col = _find_date_column(df)
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
            if len(dates) > 2:
                gaps = (dates.diff().dt.days > 7).sum()
                if gaps > 0:
                    checks.append({
                        "check": "date_gaps",
                        "status": "warn",
                        "details": f"{int(gaps)} gaps >7 days detected"
                    })
        except Exception:
            pass

    # PII detection
    pii_cols = detect_pii_columns(df)
    if pii_cols:
        checks.append({
            "check": "pii_detected",
            "status": "warn",
            "details": f"PII-like columns detected: {', '.join(pii_cols)}"
        })

    if not checks:
        checks.append({
            "check": "baseline",
            "status": "pass",
            "details": "No obvious governance issues detected"
        })

    return checks


def compute_causal_proxy(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Lightweight causal proxy: difference-in-differences if treatment + date + outcome exist.
    """
    if df.empty:
        return {}
    treatment_cols = [c for c in df.columns if c.lower() in ["treatment", "is_treatment", "exposed", "variant"]]
    outcome_cols = [c for c in df.columns if c in ["Revenue", "Units_Sold", "Marketing_Spend", "y"]]
    date_col = _find_date_column(df)
    if not treatment_cols or not outcome_cols or not date_col:
        return {}

    treatment_col = treatment_cols[0]
    outcome_col = outcome_cols[0]

    try:
        # Attempt DoWhy if available
        try:
            import dowhy  # noqa: F401
            from dowhy import CausalModel
            df_local = df[[date_col, treatment_col, outcome_col]].copy()
            df_local = df_local.dropna()
            if len(df_local) > 50:
                model = CausalModel(
                    data=df_local,
                    treatment=treatment_col,
                    outcome=outcome_col
                )
                estimand = model.identify_effect()
                estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
                return {
                    "method": "dowhy_backdoor_linear_regression",
                    "outcome": outcome_col,
                    "estimated_effect": float(estimate.value)
                }
        except Exception:
            pass

        df_local = df[[date_col, treatment_col, outcome_col]].copy()
        df_local[date_col] = pd.to_datetime(df_local[date_col], errors="coerce")
        df_local[outcome_col] = pd.to_numeric(df_local[outcome_col], errors="coerce")
        df_local = df_local.dropna(subset=[date_col, outcome_col, treatment_col])
        if df_local.empty:
            return {}
        df_local = df_local.sort_values(date_col)
        split_date = df_local[date_col].median()
        pre = df_local[df_local[date_col] < split_date]
        post = df_local[df_local[date_col] >= split_date]

        def _avg(group_df, treat_val):
            subset = group_df[group_df[treatment_col] == treat_val]
            return float(subset[outcome_col].mean()) if not subset.empty else None

        pre_t = _avg(pre, 1)
        pre_c = _avg(pre, 0)
        post_t = _avg(post, 1)
        post_c = _avg(post, 0)

        if None in (pre_t, pre_c, post_t, post_c):
            return {}

        did = (post_t - pre_t) - (post_c - pre_c)
        return {
            "method": "difference_in_differences",
            "outcome": outcome_col,
            "estimated_effect": float(did)
        }
    except Exception:
        return {}


def compute_scenario_simulation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple what-if simulator using linear regression (if possible).
    """
    if df.empty:
        return {}
    try:
        from sklearn.linear_model import LinearRegression
    except Exception:
        return {}

    # Select numeric features and a target
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return {}

    target = "Revenue" if "Revenue" in numeric_cols else numeric_cols[0]
    features = [c for c in numeric_cols if c != target]
    if not features:
        return {}

    try:
        df_local = df[features + [target]].dropna()
        if len(df_local) < 10:
            return {}
        X = df_local[features]
        y = df_local[target]

        model = LinearRegression()
        model.fit(X, y)

        # Simulate +10% for the first feature
        scenario = X.copy()
        scenario[features[0]] = scenario[features[0]] * 1.10
        predicted = model.predict(scenario).mean()
        baseline = model.predict(X).mean()
        uplift = predicted - baseline

        return {
            "target": target,
            "feature_adjusted": features[0],
            "adjustment": "+10%",
            "baseline_predicted": float(baseline),
            "scenario_predicted": float(predicted),
            "uplift": float(uplift)
        }
    except Exception:
        return {}
    date_col = _find_date_column(df)
    if not date_col:
        return {}
    try:
        df_local = df.copy()
        df_local[date_col] = pd.to_datetime(df_local[date_col], errors="coerce")
        df_local = df_local.dropna(subset=[date_col])
        if df_local.empty:
            return {}
        max_date = df_local[date_col].max()
        window_7 = df_local[df_local[date_col] >= (max_date - pd.Timedelta(days=7))]
        window_30 = df_local[df_local[date_col] >= (max_date - pd.Timedelta(days=30))]
        return {
            "last_7_days": {
                "revenue": _safe_sum(window_7, "Revenue"),
                "units_sold": _safe_sum(window_7, "Units_Sold")
            },
            "last_30_days": {
                "revenue": _safe_sum(window_30, "Revenue"),
                "units_sold": _safe_sum(window_30, "Units_Sold")
            }
        }
    except Exception:
        return {}


def compute_risk_alerts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if df.empty:
        return alerts

    # Missingness
    missingness = float(df.isna().mean().mean() * 100.0)
    if missingness > 10:
        alerts.append({
            "type": "missing_data",
            "message": f"Missing data detected across dataset (~{missingness:.1f}%).",
            "severity": "medium"
        })

    # Negative values in common numeric columns
    for col in ["Units_Sold", "Revenue", "Marketing_Spend"]:
        if col in df.columns:
            try:
                neg_count = (pd.to_numeric(df[col], errors="coerce") < 0).sum()
                if neg_count > 0:
                    alerts.append({
                        "type": "anomaly",
                        "message": f"Column {col} has {int(neg_count)} negative values.",
                        "severity": "high"
                    })
            except Exception:
                pass

    return alerts


def compute_statistical_tests(df: pd.DataFrame) -> List[Dict[str, Any]]:
    tests: List[Dict[str, Any]] = []
    if df.empty:
        return tests

    # Correlation (Pearson) between two most numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        try:
            col_a, col_b = numeric_cols[0], numeric_cols[1]
            series_a = pd.to_numeric(df[col_a], errors="coerce").fillna(0)
            series_b = pd.to_numeric(df[col_b], errors="coerce").fillna(0)
            if len(series_a) > 2:
                corr = float(np.corrcoef(series_a, series_b)[0, 1])
                tests.append({
                    "test": "pearson_correlation",
                    "result": f"Correlation between {col_a} and {col_b} = {corr:.4f}",
                    "p_value": None,
                    "significant": None
                })
        except Exception:
            pass

    return tests


def compute_auto_stat_tests(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Automatically select statistical tests based on detected column types.
    """
    tests: List[Dict[str, Any]] = []
    if df.empty:
        return tests

    try:
        from scipy import stats
    except Exception:
        return tests

    date_col = _find_date_column(df)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])]

    # Pearson correlation between first two numeric cols
    if len(numeric_cols) >= 2:
        a, b = numeric_cols[0], numeric_cols[1]
        series_a = pd.to_numeric(df[a], errors="coerce").dropna()
        series_b = pd.to_numeric(df[b], errors="coerce").dropna()
        if len(series_a) > 2 and len(series_b) > 2:
            corr, p = stats.pearsonr(series_a[:len(series_b)], series_b[:len(series_a)])
            tests.append({
                "test": "pearson_correlation",
                "result": f"Correlation between {a} and {b} = {corr:.4f}",
                "p_value": float(p),
                "significant": bool(p < 0.05)
            })

    # T-test for binary categorical vs numeric
    for cat in categorical_cols:
        if df[cat].nunique() == 2 and numeric_cols:
            num = numeric_cols[0]
            groups = df.groupby(cat)[num]
            try:
                g1, g2 = [pd.to_numeric(g, errors="coerce").dropna() for _, g in groups]
                if len(g1) > 2 and len(g2) > 2:
                    t_stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                    tests.append({
                        "test": "t_test",
                        "result": f"T-test on {num} by {cat} = {t_stat:.4f}",
                        "p_value": float(p),
                        "significant": bool(p < 0.05)
                    })
            except Exception:
                pass

    # ANOVA for multi-category vs numeric
    for cat in categorical_cols:
        if df[cat].nunique() > 2 and numeric_cols:
            num = numeric_cols[0]
            try:
                groups = [pd.to_numeric(df[df[cat] == v][num], errors="coerce").dropna() for v in df[cat].unique()]
                if all(len(g) > 2 for g in groups):
                    f_stat, p = stats.f_oneway(*groups)
                    tests.append({
                        "test": "anova",
                        "result": f"ANOVA on {num} by {cat} = {f_stat:.4f}",
                        "p_value": float(p),
                        "significant": bool(p < 0.05)
                    })
            except Exception:
                pass

    return tests


def compute_drift_alerts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if df.empty:
        return alerts
    try:
        from scipy import stats
    except Exception:
        return alerts

    date_col = _find_date_column(df)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not date_col or not numeric_cols:
        return alerts

    try:
        df_local = df.copy()
        df_local[date_col] = pd.to_datetime(df_local[date_col], errors="coerce")
        df_local = df_local.dropna(subset=[date_col])
        if df_local.empty:
            return alerts
        split = df_local[date_col].median()
        pre = df_local[df_local[date_col] < split]
        post = df_local[df_local[date_col] >= split]
        for col in numeric_cols[:2]:
            if len(pre[col]) > 10 and len(post[col]) > 10:
                ks, p = stats.ks_2samp(pre[col].dropna(), post[col].dropna())
                if p < 0.05:
                    alerts.append({
                        "metric": col,
                        "test": "ks_drift",
                        "p_value": float(p),
                        "status": "drift_detected"
                    })
    except Exception:
        pass

    return alerts
