import argparse
import json
import time
import requests


def main():
    parser = argparse.ArgumentParser(description="SLA monitor for /health and /chat")
    parser.add_argument("--health", default="http://localhost:8000/api/v1/health")
    parser.add_argument("--chat", default="http://localhost:8000/api/v1/chat")
    parser.add_argument("--tenant", default="aditya-ds")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--query", default="What can you help me with?")
    args = parser.parse_args()

    headers = {"x-tenant-id": args.tenant}
    stats = []
    for i in range(args.iterations):
        entry = {"iteration": i + 1}
        t0 = time.time()
        try:
            h = requests.get(args.health, timeout=10)
            entry["health_status"] = h.status_code
            entry["health_ms"] = round((time.time() - t0) * 1000, 2)
        except Exception as e:
            entry["health_status"] = "error"
            entry["health_error"] = str(e)

        t1 = time.time()
        try:
            r = requests.post(args.chat, headers=headers, json={"query": args.query}, timeout=120)
            entry["chat_status"] = r.status_code
            entry["chat_ms"] = round((time.time() - t1) * 1000, 2)
        except Exception as e:
            entry["chat_status"] = "error"
            entry["chat_error"] = str(e)

        stats.append(entry)
        print(json.dumps(entry))
        time.sleep(args.interval)

    print(json.dumps({"summary": stats}, indent=2))


if __name__ == "__main__":
    main()
