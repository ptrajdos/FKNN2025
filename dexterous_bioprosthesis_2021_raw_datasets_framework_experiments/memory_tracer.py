#!/usr/bin/env python
"""
Memory Object Profiler – Built-in + User Classes + Tracemalloc

- Tracks memory per class (user-defined + built-in)
- Writes snapshots to `report_file`
- Traces allocation sites for dicts/lists to a separate file (`tracemalloc_file`)
- Passes stdout/stderr from the target script to the console
- Optional periodic snapshots
"""

import sys
import time
import threading
import runpy
from datetime import datetime
from collections import defaultdict
from pympler import muppy, asizeof
from contextlib import redirect_stdout, redirect_stderr
import tracemalloc

# -------------------- helpers --------------------

def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def class_key(obj):
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def summarize_by_class(objects):
    stats = defaultdict(lambda: {"count": 0, "size": 0})
    for obj in objects:
        try:
            key = class_key(obj)
            stats[key]["count"] += 1
            stats[key]["size"] += asizeof.asizeof(obj)
        except Exception:
            pass
    return sorted(stats.items(), key=lambda x: x[1]["size"], reverse=True)


def format_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


# -------------------- snapshot --------------------

def write_snapshot(outfile_path, limit=30):
    all_objects = muppy.get_objects()
    class_stats = summarize_by_class(all_objects)

    with open(outfile_path, "a", buffering=1) as f:
        f.write("\n" + "=" * 110 + "\n")
        f.write(f"Snapshot time: {current_timestamp()}\n")
        f.write("Memory usage per class (built-in + user-defined)\n")
        f.write("=" * 110 + "\n")
        f.write(f"{'Class':65} {'Count':>12} {'Total Size':>15}\n")
        f.write("-" * 110 + "\n")

        for cls, data in class_stats[:limit]:
            f.write(
                f"{cls:65} {data['count']:12d} {format_size(data['size']):>15}\n"
            )
        f.flush()


def write_tracemalloc_snapshot(tracemalloc_file, limit=10, filters=None):
    """Save top allocation sites filtered by types (dict/list/tuple)"""
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")

    with open(tracemalloc_file, "a", buffering=1) as f:
        f.write("\n" + "=" * 110 + "\n")
        f.write(f"Tracemalloc snapshot time: {current_timestamp()}\n")
        f.write("Top allocation sites\n")
        f.write("=" * 110 + "\n")

        count = 0
        for stat in stats:
            # Only show allocations matching filter types
            line_str = str(stat)
            if filters:
                if not any(ftype in line_str for ftype in filters):
                    continue
            f.write(line_str + "\n")
            count += 1
            if count >= limit:
                break
        f.flush()


# -------------------- periodic tracking --------------------

def periodic_tracker(report_file, tracemalloc_file, interval, tracemalloc_filters):
    while True:
        write_snapshot(report_file)
        # write_tracemalloc_snapshot(tracemalloc_file, filters=tracemalloc_filters)
        time.sleep(interval)


# -------------------- main --------------------

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python memory_object_profiler.py target_script.py report_file [interval_seconds] [tracemalloc_file]"
        )
        sys.exit(1)

    target_script = sys.argv[1]
    report_file = sys.argv[2]
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    tracemalloc_file = sys.argv[4] if len(sys.argv) > 4 else "tracemalloc_report.txt"

    # types to filter in tracemalloc
    tracemalloc_filters = ["dict", "list", "tuple"]

    # start tracemalloc
    tracemalloc.start(25)  # 25 frames

    # start periodic snapshots in background
    if interval > 0:
        t = threading.Thread(
            target=periodic_tracker,
            args=(report_file, tracemalloc_file, interval, tracemalloc_filters),
            daemon=True,
        )
        t.start()

    # run target script while passing stdout/stderr
    with redirect_stdout(sys.stdout), redirect_stderr(sys.stderr):
        runpy.run_path(target_script, run_name="__main__")

    # final snapshots
    write_snapshot(report_file)
    write_tracemalloc_snapshot(tracemalloc_file, filters=tracemalloc_filters)


if __name__ == "__main__":
    main()
