"""Generate registered X5 feature banks and fit the requested model configurations."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from time import time

from run import DESIGN, HERE, OUT, ROOT, read

LIMIT = 12 * 3600


def gpu_seconds(include_active=True):
    total = 0.
    for path in (OUT / "gpu_ledger").glob("*.json"):
        item = read(path)
        if item["end"] is None and not include_active:
            raise ValueError(f"GPU interval is still active: {path}")
        end = time() if item["end"] is None else item["end"]
        total += end - item["start"]
    return total


def execute(cohort, components):
    selected = set(components)
    configs = [c for c in DESIGN["core_configurations"] if c == "Original" or set(c) <= selected]
    if "D" in selected:
        configs.extend(DESIGN["formula_controls"])
    regular = [c for c in configs if "C" not in c]
    with_c = [c for c in configs if "C" in c]
    feature_python = os.environ.get("X5_FEATURE_PYTHON", sys.executable)
    logs = OUT / cohort / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def worker(tag, python, script, args):
        print("Starting", tag, flush=True)
        with (logs / (tag + ".log")).open("a", encoding="utf-8") as stream:
            subprocess.run([python, "-B", "-u", str(HERE / script), *args], cwd=ROOT,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("Finished", tag, flush=True)

    def host(group, model, seed, cells):
        worker(f"{group}_{model}_{seed}", sys.executable, "run.py", [cohort, model, str(seed), *cells])

    if "A" in selected:
        worker("prepare_A", sys.executable, "baskets.py", ["prepare"])
        worker("build_A", sys.executable, "baskets.py", ["build", "--cohort", cohort])
    if "B" in selected:
        worker("build_B", feature_python, "tabular.py", ["build", "--cohort", cohort])
    with ThreadPoolExecutor(max_workers=len(DESIGN["host_seeds"]["ptonet"])) as pool:
        seeds = DESIGN["host_seeds"]["ptonet"]
        native_seed, = DESIGN["host_seeds"]["causalpfn"]
        fits = [pool.submit(host, "regular", "ptonet", seed, regular) for seed in seeds]
        host("regular", "causalpfn", native_seed, regular)
        if with_c:
            worker("build_C", feature_python, "qwen.py", ["build", "--cohort", cohort])
        for fit in fits:
            fit.result()
        if with_c:
            fits = [pool.submit(host, "with_C", "ptonet", seed, with_c) for seed in seeds]
            host("with_C", "causalpfn", native_seed, with_c)
            for fit in fits:
                fit.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("development", "full"))
    parser.add_argument("--components", choices=list("ABCD"), nargs="*", default=list("ABCD"))
    args = parser.parse_args()
    execute(args.cohort, args.components)
