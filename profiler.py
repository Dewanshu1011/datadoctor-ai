"""Deterministic, local data-quality profiling rules."""
from __future__ import annotations

import io
import re
from collections import Counter

import numpy as np
import pandas as pd

EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def load_dataset(raw: bytes, filename: str) -> pd.DataFrame:
    """Read supported files without writing uploads to disk."""
    if filename.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def _finding(severity: str, title: str, description: str, columns: list[str] | None = None) -> dict:
    return {"severity": severity, "title": title, "description": description, "columns": columns or []}


def _looks_like_date(name: str) -> bool:
    return any(token in name.lower() for token in ("date", "time", "_at", "dob", "birth"))


def _likely_identifier(name: str) -> bool:
    return name.lower() == "id" or name.lower().endswith(("_id", "id"))


def build_report(df: pd.DataFrame, source_name: str = "dataset") -> dict:
    rows, cols = df.shape
    findings: list[dict] = []
    column_profiles: list[dict] = []
    penalties: list[float] = []
    invalid_email_columns, invalid_date_columns, invalid_numeric_columns, outlier_columns, constants = [], [], [], [], []

    if rows == 0:
        findings.append(_finding("critical", "Dataset is empty", "No rows were found; quality checks cannot be meaningfully evaluated."))
        penalties.append(50)
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        pct = duplicate_rows / max(rows, 1) * 100
        severity = "critical" if pct >= 5 else "warning"
        findings.append(_finding(severity, "Duplicate rows detected", f"{duplicate_rows:,} fully duplicate rows ({pct:.1f}%). Remove or deduplicate records before analysis."))
        penalties.append(min(20, pct))
    else:
        findings.append(_finding("passed", "No duplicate rows", "No fully duplicate records were found."))

    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        missing_pct = round(missing / max(rows, 1) * 100, 2)
        unique = int(series.nunique(dropna=True))
        profile = {"column": str(col), "dtype": str(series.dtype), "non_null": non_null, "missing": missing,
                   "missing_pct": missing_pct, "unique": unique, "unique_pct": round(unique / max(non_null, 1) * 100, 2)}
        if pd.api.types.is_numeric_dtype(series):
            profile.update({"min": _safe_num(series.min()), "max": _safe_num(series.max()), "mean": _safe_num(series.mean())})
            cleaned = series.dropna()
            if len(cleaned) >= 8:
                q1, q3 = cleaned.quantile([.25, .75])
                iqr = q3 - q1
                outliers = int(((cleaned < q1 - 1.5 * iqr) | (cleaned > q3 + 1.5 * iqr)).sum()) if iqr else 0
                profile["outliers"] = outliers
                if outliers:
                    outlier_columns.append(f"{col} ({outliers:,})")
        else:
            profile["sample_values"] = [str(v)[:80] for v in series.dropna().astype(str).head(3)]

        if missing:
            severity = "critical" if missing_pct >= 25 else "warning"
            findings.append(_finding(severity, f"Missing values in '{col}'", f"{missing:,} values are missing ({missing_pct:.1f}%).", [str(col)]))
            penalties.append(min(12, missing_pct * .18))
        if non_null > 1 and unique <= 1:
            constants.append(str(col))
        if _likely_identifier(str(col)) and non_null and unique < non_null:
            duplicates = non_null - unique
            findings.append(_finding("critical", f"Possible duplicate key: '{col}'", f"{duplicates:,} duplicate non-null values found in a likely identifier column.", [str(col)]))
            penalties.append(12)
        if "email" in str(col).lower() and non_null:
            invalid = int((~series.dropna().astype(str).str.match(EMAIL)).sum())
            if invalid:
                invalid_email_columns.append(f"{col} ({invalid:,})")
        if _looks_like_date(str(col)) and non_null and not pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series.dropna(), errors="coerce")
            invalid = int(parsed.isna().sum())
            if invalid:
                invalid_date_columns.append(f"{col} ({invalid:,})")
        # A numeric-sounding column represented as text often indicates a mixed
        # type import (for example, an "N/A" embedded in an amount column).
        numeric_hint = any(token in str(col).lower() for token in ("amount", "total", "price", "count", "age", "quantity", "score"))
        if numeric_hint and not pd.api.types.is_numeric_dtype(series) and non_null:
            coerced = pd.to_numeric(series.dropna(), errors="coerce")
            invalid = int(coerced.isna().sum())
            if invalid:
                invalid_numeric_columns.append(f"{col} ({invalid:,})")
        column_profiles.append(profile)

    if constants:
        findings.append(_finding("warning", "Constant columns", f"These columns contain only one non-null value: {', '.join(constants)}.", constants))
        penalties.append(min(8, len(constants) * 2))
    if invalid_email_columns:
        findings.append(_finding("warning", "Invalid email formats", "Malformed email values: " + ", ".join(invalid_email_columns) + ".", [x.split(" (")[0] for x in invalid_email_columns]))
        penalties.append(6)
    if invalid_date_columns:
        findings.append(_finding("warning", "Invalid date formats", "Values could not be parsed as dates: " + ", ".join(invalid_date_columns) + ".", [x.split(" (")[0] for x in invalid_date_columns]))
        penalties.append(6)
    if invalid_numeric_columns:
        findings.append(_finding("warning", "Invalid numeric values", "Text or invalid values found in numeric-looking columns: " + ", ".join(invalid_numeric_columns) + ".", [x.split(" (")[0] for x in invalid_numeric_columns]))
        penalties.append(6)
    if outlier_columns:
        findings.append(_finding("warning", "Potential numeric outliers", "IQR-based outliers found in: " + ", ".join(outlier_columns) + ". Review these values in context.", [x.split(" (")[0] for x in outlier_columns]))
        penalties.append(min(8, len(outlier_columns) * 2))
    if not any(f["severity"] in ("critical", "warning") for f in findings):
        findings.append(_finding("passed", "All core checks passed", "No data-quality issues were detected by the configured rules."))

    score = max(0, round(100 - sum(penalties)))
    return {"source": source_name, "quality_score": score, "score_label": "Excellent" if score >= 85 else "Needs attention" if score >= 60 else "At risk",
            "overview": {"rows": int(rows), "columns": int(cols), "estimated_memory": _format_bytes(int(df.memory_usage(deep=True).sum())), "duplicate_rows": duplicate_rows},
            "columns": column_profiles, "findings": findings}


def _safe_num(value: object) -> float | None:
    return None if pd.isna(value) else round(float(value), 4)


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return str(size)


def report_to_markdown(report: dict) -> str:
    lines = [f"# DataDoctor AI report: {report['source']}", f"\n## Quality score\n{report['quality_score']}/100 — {report['score_label']}", "\n## Findings"]
    for item in report["findings"]:
        lines.append(f"- **{item['severity'].title()} — {item['title']}**: {item['description']}")
    lines.append("\n## Column profile")
    lines.extend([f"- `{c['column']}`: {c['dtype']}; {c['missing_pct']}% missing; {c['unique']} unique" for c in report["columns"]])
    return "\n".join(lines)
