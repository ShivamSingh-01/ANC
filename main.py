import argparse

from offline_demo import run_offline_demo
from live_demo import run_live_demo

parser = argparse.ArgumentParser()

parser.add_argument(
    "--live",
    action="store_true",
    help="Run live microphone ANC demo"
)

args = parser.parse_args()

if args.live:
    run_live_demo()
else:
    run_offline_demo()