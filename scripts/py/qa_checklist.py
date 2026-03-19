import json
import time
from pathlib import Path
import requests


def record_case(lines, title, request_payload, response):
    lines.append(f"## {title}")
    lines.append("Request:")
    lines.append("```json")
    lines.append(json.dumps(request_payload, indent=2))
    lines.append("```")
    if response is None:
        lines.append("Status: error")
        lines.append("")
        return
    lines.append(f"Status: {response.status_code}")
    lines.append("Response:")
    lines.append("```json")
    try:
        parsed = response.json()
        lines.append(json.dumps(parsed, indent=2))
    except Exception:
        lines.append(response.text)
    lines.append("```")
    lines.append("")


def main():
    base = "http://localhost:8000/api/v1"
    headers = {"x-tenant-id": "aditya-ds"}
    lines = ["# QA Checklist - Production Handover", f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", ""]

    # RAG basic
    rag_payload = {
        "query": "Summarize what Akobot.ai offers based on the knowledge base.",
        "model_provider": "auto",
        "session_id": "rag-qa-live-1",
        "stream": False,
    }
    rag_resp = requests.post(f"{base}/chat", headers=headers, json=rag_payload, timeout=120)
    record_case(lines, "1. RAG Basic", rag_payload, rag_resp)

    # RAG email connect link -> Done -> retry
    rag_email_payload = {
        "query": "Summarize the sources and send email to adityamishra0996@gmail.com subject: RAG Summary body: Please send the summary.",
        "model_provider": "auto",
        "session_id": "rag-qa-live-2",
        "stream": False,
    }
    rag_email_resp = requests.post(f"{base}/chat", headers=headers, json=rag_email_payload, timeout=120)
    record_case(lines, "2. RAG Email Connect Link", rag_email_payload, rag_email_resp)

    rag_done_payload = {
        "query": "Done",
        "model_provider": "auto",
        "session_id": "rag-qa-live-2",
        "stream": False,
    }
    rag_done_resp = requests.post(f"{base}/chat", headers=headers, json=rag_done_payload, timeout=120)
    record_case(lines, "3. RAG Email Done Retry", rag_done_payload, rag_done_resp)

    # BA with CSV + email + done retry
    ba_dataset = Path("data") / "uploads" / "marketing_ecommerce_benchmark.csv"
    if ba_dataset.exists():
        ba_payload = {
            "query": "Find top product by revenue by region and forecast next 30 days. Send email to adityamishra0996@gmail.com subject: BA Summary body: Please send the BA summary.",
            "session_id": "ba-qa-live-1",
        }
        with ba_dataset.open("rb") as f:
            files = {"files": (ba_dataset.name, f, "text/csv")}
            ba_resp = requests.post(
                f"{base}/business_analyst/chat",
                headers=headers,
                files=files,
                data=ba_payload,
                timeout=180,
            )
        record_case(lines, "4. BA Email Connect Link", ba_payload, ba_resp)

        ba_done_payload = {"query": "Done", "session_id": "ba-qa-live-1"}
        with ba_dataset.open("rb") as f:
            files = {"files": (ba_dataset.name, f, "text/csv")}
            ba_done_resp = requests.post(
                f"{base}/business_analyst/chat",
                headers=headers,
                files=files,
                data=ba_done_payload,
                timeout=180,
            )
        record_case(lines, "5. BA Email Done Retry", ba_done_payload, ba_done_resp)
    else:
        lines.append("## 4. BA Email Connect Link")
        lines.append(f"- dataset missing: {ba_dataset}")
        lines.append("")

    # Support DA CRM + email + done retry
    support_payload = {
        "query": "Use CRM to find any open tickets for customer email adityamishra0996@gmail.com and summarize. Send email to adityamishra0996@gmail.com subject: Support Summary body: Please send the summary.",
        "session_id": "support-qa-live-1",
    }
    support_headers = {"x-tenant-id": "aditya-ds-crm-test"}
    support_resp = requests.post(
        f"{base}/support_data_analytics/chat",
        headers=support_headers,
        data=support_payload,
        timeout=180,
    )
    record_case(lines, "6. Support DA CRM + Email", support_payload, support_resp)

    support_done_payload = {"query": "Done", "session_id": "support-qa-live-1"}
    support_done_resp = requests.post(
        f"{base}/support_data_analytics/chat",
        headers=support_headers,
        data=support_done_payload,
        timeout=120,
    )
    record_case(lines, "7. Support DA Done Retry", support_done_payload, support_done_resp)

    out_path = Path("logs") / "qa_checklist.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote checklist to {out_path}")


if __name__ == "__main__":
    main()
