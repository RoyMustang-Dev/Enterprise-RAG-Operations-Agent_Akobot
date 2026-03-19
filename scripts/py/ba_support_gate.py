import json
import time
import os
from pathlib import Path
import requests


def _check(resp, name, max_latency_s):
    if resp is None:
        return False, f"{name}: no response"
    if resp.status_code != 200:
        return False, f"{name}: status {resp.status_code}"
    if max_latency_s is not None and resp.elapsed.total_seconds() > max_latency_s:
        return False, f"{name}: latency {resp.elapsed.total_seconds():.2f}s > {max_latency_s}s"
    return True, f"{name}: ok"


def main():
    base = "http://localhost:8000/api/v1"
    headers = {"x-tenant-id": "aditya-ds"}
    max_latency = float(120)
    failures = []

    ba_dataset = Path("data") / "uploads" / "marketing_ecommerce_benchmark.csv"
    if ba_dataset.exists():
        with ba_dataset.open("rb") as f:
            files = {"files": (ba_dataset.name, f, "text/csv")}
            data = {"query": "Run a quick trend analysis for each region.", "session_id": "ba-gate-1"}
            ba_resp = requests.post(
                f"{base}/business_analyst/chat",
                headers=headers,
                files=files,
                data=data,
                timeout=180,
            )
        ok, msg = _check(ba_resp, "BA", max_latency)
        if not ok:
            failures.append(msg)
    else:
        failures.append("BA: dataset missing")

    support_headers = {"x-tenant-id": "aditya-ds"}
    support_payload = {
        "query": "Use CRM to list open tickets for customer email test@example.com. Send email to test@example.com subject: Support Summary body: Please send the summary.",
        "session_id": "support-gate-1",
    }
    support_resp = requests.post(
        f"{base}/support_data_analytics/chat",
        headers=support_headers,
        data=support_payload,
        timeout=180,
    )
    ok, msg = _check(support_resp, "Support DA", max_latency)
    if not ok:
        failures.append(msg)

    # SLA gate (latency/error rate)
    try:
        sla_resp = requests.get(f"{base}/metrics/sla", headers=headers, timeout=20)
        if sla_resp.status_code != 200:
            failures.append(f"SLA: status {sla_resp.status_code}")
        else:
            sla = sla_resp.json()
            p95 = (sla.get("latency_ms") or {}).get("p95")
            err = sla.get("error_rate", 0.0)
            max_p95_ms = float(os.getenv("SLA_P95_MS", "90000"))
            max_error = float(os.getenv("SLA_ERROR_RATE", "0.2"))
            if p95 is not None and p95 > max_p95_ms:
                failures.append(f"SLA: p95 {p95}ms > {max_p95_ms}ms")
            if err > max_error:
                failures.append(f"SLA: error_rate {err} > {max_error}")
    except Exception as e:
        failures.append(f"SLA: failed to fetch metrics ({e})")

    out_path = Path("logs") / "ba_support_gate.json"
    out_path.write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit("Gate failed: " + "; ".join(failures))
    print("Gate passed.")


if __name__ == "__main__":
    main()
