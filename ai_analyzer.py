"""Optional OpenAI-powered interpretation. All deterministic checks live in profiler.py."""
from __future__ import annotations

import json
import os


def _safe_context(report: dict) -> str:
    """Return metadata only; omits sample cell values and raw data."""
    safe = {"quality_score": report["quality_score"], "overview": report["overview"], "findings": report["findings"],
            "columns": [{k: v for k, v in col.items() if k != "sample_values"} for col in report["columns"]]}
    return json.dumps(safe, default=str)


def _client_response(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), input=prompt)
        return response.output_text
    except Exception as exc:
        return f"AI service is unavailable: {exc}"


def _fallback_explanation(report: dict) -> str:
    issues = [f for f in report["findings"] if f["severity"] != "passed"]
    if not issues:
        return "### Dataset looks healthy\nThe deterministic checks found no material quality issues. Keep monitoring quality as new data arrives."
    bullets = "\n".join(f"- **{i['title']}**: {i['description']}" for i in issues[:6])
    return f"### What needs attention\nYour score is **{report['quality_score']}/100**. Prioritize critical issues first, then validate warning-level patterns with the data owner.\n\n{bullets}\n\n*Add `OPENAI_API_KEY` to receive a tailored AI explanation and remediation plan.*"


def generate_ai_explanation(report: dict) -> str:
    prompt = """You are a senior data quality engineer. Explain the supplied DATASET METADATA in concise Markdown. Prioritize risks, likely downstream impact, and a practical ordered remediation plan. Do not claim to inspect raw data.\n\nDATASET METADATA:\n""" + _safe_context(report)
    return _client_response(prompt) or _fallback_explanation(report)


def generate_fix(report: dict, target: str) -> str:
    prompt = f"""You are a senior analytics engineer. Based only on this anonymized data quality report, produce a concise, safe {target} remediation template. Use clearly marked placeholder table/dataframe names, comments for assumptions, and do not invent column-specific values. Return code only.\n\n{_safe_context(report)}"""
    result = _client_response(prompt)
    if result:
        return result
    if target == "SQL":
        return """-- Replace source_table and key_column after confirming business rules.
WITH deduplicated AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY key_column ORDER BY updated_at DESC) AS rn
  FROM source_table
)
SELECT * EXCEPT (rn)
FROM deduplicated
WHERE rn = 1
  AND email IS NOT NULL
  AND REGEXP_LIKE(email, '^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$');"""
    return """# Replace source_df and key_column after confirming business rules.
from pyspark.sql import functions as F
from pyspark.sql.window import Window

clean_df = (source_df
    .filter(F.col("email").rlike(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$"))
    .withColumn("_rn", F.row_number().over(Window.partitionBy("key_column").orderBy(F.col("updated_at").desc_nulls_last())))
    .filter(F.col("_rn") == 1)
    .drop("_rn"))"""
