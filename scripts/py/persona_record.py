import argparse
import json
import time
from pathlib import Path

import requests


def call_chat(base: str, headers: dict, query: str) -> dict:
    t0 = time.time()
    resp = requests.post(f"{base}/chat", headers=headers, json={"query": query}, timeout=120)
    latency = round(time.time() - t0, 2)
    return {"status": resp.status_code, "latency_s": latency, "payload": resp.json()}


def call_ba(base: str, headers: dict, query: str, dataset_path: Path) -> dict:
    t0 = time.time()
    with dataset_path.open("rb") as f:
        files = {"files": (dataset_path.name, f, "text/csv")}
        data = {"query": query, "session_id": f"ba-persona-{int(time.time())}"}
        resp = requests.post(f"{base}/business_analyst/chat", headers=headers, files=files, data=data, timeout=300)
    latency = round(time.time() - t0, 2)
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": resp.text}
    return {"status": resp.status_code, "latency_s": latency, "payload": payload}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tenant", default="aditya-ds")
    args = parser.parse_args()

    base = "http://localhost:8000/api/v1"
    headers = {"x-tenant-id": args.tenant}

    greeting_query = "hii how can you help me"
    rag_query = "summarize all the data that we have in the sources & explain them properly"
    ba_query = "Identify the future trends of all products based on current data for every region individually."

    dataset_path = Path("data") / "uploads" / "marketing_ecommerce_benchmark.csv"
    if not dataset_path.exists():
        dataset_path = Path("data") / "marketing_ecommerce_benchmark.csv"

    results = {
        "label": args.label,
        "tenant": args.tenant,
        "greeting": call_chat(base, headers, greeting_query),
        "rag": call_chat(base, headers, rag_query),
        "ba": call_ba(base, headers, ba_query, dataset_path),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_payload(payload: dict) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False)[:4000]

    lines = []
    lines.append(f"## {args.label}")
    lines.append("")
    lines.append("### Greeting Query")
    lines.append(f"- query: {greeting_query}")
    lines.append(f"- status: {results['greeting']['status']}")
    lines.append(f"- latency_s: {results['greeting']['latency_s']}")
    lines.append("```json")
    lines.append(_format_payload(results["greeting"]["payload"]))
    lines.append("```")
    lines.append("")

    lines.append("### RAG Query")
    lines.append(f"- query: {rag_query}")
    lines.append(f"- status: {results['rag']['status']}")
    lines.append(f"- latency_s: {results['rag']['latency_s']}")
    lines.append("```json")
    lines.append(_format_payload(results["rag"]["payload"]))
    lines.append("```")
    lines.append("")

    lines.append("### BA Query")
    lines.append(f"- query: {ba_query}")
    lines.append(f"- status: {results['ba']['status']}")
    lines.append(f"- latency_s: {results['ba']['latency_s']}")
    lines.append("```json")
    lines.append(_format_payload(results["ba"]["payload"]))
    lines.append("```")
    lines.append("")

    existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    out_path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
