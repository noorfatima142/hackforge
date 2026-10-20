import subprocess
import sys
import time
from datetime import datetime, timedelta

MINUTE = 5  # run at five past every hour (PKT hours match UTC hours)


def next_run(now: datetime) -> datetime:
    t = now.replace(minute=MINUTE, second=0, microsecond=0)
    if t <= now:
        t += timedelta(hours=1)
    return t


def main() -> None:
    print("Hourly runner started. Leave this window open. Ctrl+C stops it.")
    while True:
        target = next_run(datetime.now())
        print(f"Next run at {target:%Y-%m-%d %H:%M:%S}")
        while datetime.now() < target:
            time.sleep(20)
        print(f"Starting run at {datetime.now():%H:%M:%S}")
        result = subprocess.run([sys.executable, "-m", "src.flows", "--once"])
        print(f"Run finished with exit code {result.returncode}")


if __name__ == "__main__":
    main()