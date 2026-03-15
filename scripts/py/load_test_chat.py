import argparse
import json
import threading
import time
from statistics import mean
import requests


def worker(url, headers, payload, results, idx):
    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        latency = time.time() - t0
        results[idx] = (resp.status_code, latency)
    except Exception:
        results[idx] = ("error", time.time() - t0)


def main():
    parser = argparse.ArgumentParser(description="Lightweight load test for /api/v1/chat")
    parser.add_argument("--url", default="http://localhost:8000/api/v1/chat")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--query", default="What can you help me with?")
    parser.add_argument("--tenant", default="aditya-ds")
    args = parser.parse_args()

    headers = {"x-tenant-id": args.tenant}
    payload = {"query": args.query}

    results = [None] * args.requests
    threads = []
    start = time.time()
    for i in range(args.requests):
        t = threading.Thread(target=worker, args=(args.url, headers, payload, results, i))
        threads.append(t)
        t.start()
        while threading.active_count() > args.concurrency:
            time.sleep(0.01)
    for t in threads:
        t.join()
    total = time.time() - start

    latencies = [r[1] for r in results if r]
    status_counts = {}
    for r in results:
        status = r[0]
        status_counts[status] = status_counts.get(status, 0) + 1

    latencies_sorted = sorted(latencies)
    p95 = latencies_sorted[int(0.95 * len(latencies_sorted)) - 1] if latencies_sorted else 0
    p99 = latencies_sorted[int(0.99 * len(latencies_sorted)) - 1] if latencies_sorted else 0

    print(json.dumps({
        "requests": args.requests,
        "concurrency": args.concurrency,
        "total_time_s": round(total, 2),
        "avg_latency_s": round(mean(latencies), 3) if latencies else None,
        "p95_latency_s": round(p95, 3),
        "p99_latency_s": round(p99, 3),
        "status_counts": status_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
