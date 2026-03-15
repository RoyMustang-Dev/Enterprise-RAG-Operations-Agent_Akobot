import json
import os
import time
from pathlib import Path
import requests


def write_section(lines, title, body_lines):
    lines.append(f"## {title}")
    lines.extend(body_lines)
    lines.append("")


def main():
    base = "http://localhost:8000/api/v1"
    headers = {"x-tenant-id": "aditya-ds"}
    report_lines = ["# Verification Report", ""]

    # 1) Langfuse import check
    try:
        import langfuse  # noqa: F401
        write_section(report_lines, "Langfuse Import", ["- status: ok"])
    except Exception as e:
        write_section(report_lines, "Langfuse Import", [f"- status: failed ({e})"])

    # 2) RAG query (simple) with file attachment for guaranteed context
    rag_file = Path("test-files") / "new-flow-test" / "Automaatte.txt"
    rag_query = "Summarize the attached document in 5 bullet points."
    rag_resp = None
    rag_err = None
    try:
        if rag_file.exists():
            with rag_file.open("rb") as f:
                files = {"files": (rag_file.name, f, "text/plain")}
                data = {"query": rag_query}
                t0 = time.time()
                rag_resp = requests.post(f"{base}/chat", headers=headers, files=files, data=data, timeout=120)
                rag_time = round(time.time() - t0, 2)
        else:
            t0 = time.time()
            rag_resp = requests.post(f"{base}/chat", headers=headers, json={"query": rag_query}, timeout=120)
            rag_time = round(time.time() - t0, 2)
    except Exception as e:
        rag_time = round(time.time() - t0, 2)
        rag_err = str(e)
    write_section(
        report_lines,
        "RAG Query",
        [
            f"- file: {rag_file if rag_file.exists() else 'none'}",
            f"- query: {rag_query}",
            f"- status: {rag_resp.status_code if rag_resp else 'error'}",
            f"- time_s: {rag_time}",
            f"- response: {(rag_resp.text[:1200] if rag_resp else rag_err)}",
        ],
    )

    # 3) RAG stress query (with same file context)
    rag_stress = (
        "From the knowledge base, provide a structured brief: "
        "1) Core product purpose, 2) Top 5 features, 3) Target users, "
        "4) Any limitations/unknowns, 5) Provide 3 source-backed bullet points. "
        "If information is missing, explicitly say so."
    )
    rag_stress_resp = None
    rag_stress_err = None
    try:
        if rag_file.exists():
            with rag_file.open("rb") as f:
                files = {"files": (rag_file.name, f, "text/plain")}
                data = {"query": rag_stress}
                t0 = time.time()
                rag_stress_resp = requests.post(f"{base}/chat", headers=headers, files=files, data=data, timeout=120)
                rag_stress_time = round(time.time() - t0, 2)
        else:
            t0 = time.time()
            rag_stress_resp = requests.post(f"{base}/chat", headers=headers, json={"query": rag_stress}, timeout=120)
            rag_stress_time = round(time.time() - t0, 2)
    except Exception as e:
        rag_stress_time = round(time.time() - t0, 2)
        rag_stress_err = str(e)
    write_section(
        report_lines,
        "RAG Stress Query",
        [
            f"- file: {rag_file if rag_file.exists() else 'none'}",
            f"- query: {rag_stress}",
            f"- status: {rag_stress_resp.status_code if rag_stress_resp else 'error'}",
            f"- time_s: {rag_stress_time}",
            f"- response: {(rag_stress_resp.text[:1200] if rag_stress_resp else rag_stress_err)}",
        ],
    )

    # 4) BA query (simple)
    ba_dataset = Path("data") / "uploads" / "marketing_ecommerce_benchmark.csv"
    if not ba_dataset.exists():
        ba_dataset = Path("data") / "uploads" / "marketing_ecommerce_benchmark.csv"
    if not ba_dataset.exists():
        ba_dataset = Path("data") / "marketing_ecommerce_benchmark.csv"

    ba_query = "Identify the future trends of all products based on current data for every region individually."
    if ba_dataset.exists():
        ba_resp = None
        ba_err = None
        try:
            with ba_dataset.open("rb") as f:
                files = {"files": (ba_dataset.name, f, "text/csv")}
                data = {"query": ba_query, "session_id": "ba-verify-1"}
                t0 = time.time()
                ba_resp = requests.post(
                    f"{base}/business_analyst/chat",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=180,
                )
                ba_time = round(time.time() - t0, 2)
        except Exception as e:
            ba_time = round(time.time() - t0, 2)
            ba_err = str(e)
        write_section(
            report_lines,
            "BA Query",
            [
                f"- dataset: {ba_dataset}",
                f"- query: {ba_query}",
                f"- status: {ba_resp.status_code if ba_resp else 'error'}",
                f"- time_s: {ba_time}",
                f"- response: {(ba_resp.text[:1200] if ba_resp else ba_err)}",
            ],
        )
    else:
        write_section(
            report_lines,
            "BA Query",
            ["- dataset not found, skipped."],
        )

    # 5) BA stress query
    ba_stress = (
        "Using the dataset, build a full region-by-region trend analysis. "
        "Include top 3 products per region by forecast delta, overall regional trend, "
        "backtest MAE, confidence interval, and highlight any anomalies or risks."
    )
    if ba_dataset.exists():
        ba_stress_resp = None
        ba_stress_err = None
        try:
            with ba_dataset.open("rb") as f:
                files = {"files": (ba_dataset.name, f, "text/csv")}
                data = {"query": ba_stress, "session_id": "ba-verify-2"}
                t0 = time.time()
                ba_stress_resp = requests.post(
                    f"{base}/business_analyst/chat",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=180,
                )
                ba_stress_time = round(time.time() - t0, 2)
        except Exception as e:
            ba_stress_time = round(time.time() - t0, 2)
            ba_stress_err = str(e)
        write_section(
            report_lines,
            "BA Stress Query",
            [
                f"- dataset: {ba_dataset}",
                f"- query: {ba_stress}",
                f"- status: {ba_stress_resp.status_code if ba_stress_resp else 'error'}",
                f"- time_s: {ba_stress_time}",
                f"- response: {(ba_stress_resp.text[:1200] if ba_stress_resp else ba_stress_err)}",
            ],
        )

    # 6) SLA Metrics snapshot
    t0 = time.time()
    sla_resp = None
    sla_err = None
    try:
        sla_resp = requests.get(f"{base}/metrics/sla", headers=headers, timeout=30)
        sla_time = round(time.time() - t0, 2)
    except Exception as e:
        sla_time = round(time.time() - t0, 2)
        sla_err = str(e)
    write_section(
        report_lines,
        "SLA Metrics Snapshot",
        [
            f"- status: {sla_resp.status_code if sla_resp else 'error'}",
            f"- time_s: {sla_time}",
            f"- response: {(sla_resp.text[:1200] if sla_resp else sla_err)}",
        ],
    )

    out_path = Path("logs") / "verification_report.md"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
