#!/usr/bin/env python3
"""Render Cron Job entry point: ask OrderScan PHP to prepare due jobs.

This script never calls Exotel. It only invokes the password-protected PHP
scheduler endpoint, which writes WAITING_FOR_EXOTEL_DISPATCH queue records.
"""
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    url = os.environ.get("VOICE_SCHEDULER_URL", "").strip()
    secret = os.environ.get("VOICE_SCHEDULER_SECRET", "").strip()

    if not url or not secret:
        print("ERROR: set VOICE_SCHEDULER_URL and VOICE_SCHEDULER_SECRET in Render.")
        return 2
    if not url.startswith("https://"):
        print("ERROR: VOICE_SCHEDULER_URL must use HTTPS.")
        return 2

    body = json.dumps({"source": "render-cron"}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Scheduler-Secret": secret,
            "User-Agent": "OrderScan-Scheduler/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = response.read(10000).decode("utf-8", errors="replace")
            print(f"Scheduler endpoint HTTP {response.status}")
            try:
                data = json.loads(payload)
                print(json.dumps(data, ensure_ascii=False))
                if response.status != 200 or not data.get("ok"):
                    return 1
            except json.JSONDecodeError:
                print(payload)
                return 1
        return 0
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode("utf-8", errors="replace")
        print(f"ERROR: scheduler endpoint returned HTTP {exc.code}: {detail}")
        return 1
    except Exception as exc:
        print(f"ERROR: could not contact scheduler endpoint: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
