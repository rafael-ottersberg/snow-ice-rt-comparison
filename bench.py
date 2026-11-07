"""Wall-clock benchmarks of the run scripts: one JSON line per timed stage in RESULTS/benchmarks.jsonl."""
import json
import os
import time
from contextlib import contextmanager

from cases import RESULTS


def record(stage, seconds, **info):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "benchmarks.jsonl", "a") as f:
        f.write(json.dumps(dict(stage=stage, seconds=round(float(seconds), 2), cpus=os.cpu_count(), **info)) + "\n")


@contextmanager
def timed(stage, **info):
    t0 = time.perf_counter()
    yield
    record(stage, time.perf_counter() - t0, **info)
