#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_all.py - runs the tests of the job runner on CPU, each with its interpreter and a timeout, and prints a summary table."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parents[2]
SYS_PY = "/usr/bin/python3"
ORDER = [("a", "test_a_locks.py", 420, sys.executable), ("b", "test_b_pipelines_cache.py", 300, sys.executable),
         ("c", "test_c_equivalence.py", 900, sys.executable), ("d", "test_d_prefix_original.py", 900, sys.executable),
         ("e", "test_e_windowed_smoke.py", 1200, sys.executable), ("f", "test_f_postproc_harness.py", 900, SYS_PY),
         ("gp", "test_g_pipelines_cic_aci.py", 600, sys.executable), ("g", "test_g_runner_v2.py", 1200, sys.executable),
         ("i", "test_i_mlp_ctx.py", 900, sys.executable)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    sel = set(a.only.split(",")) if a.only else ({"a", "b", "c", "d", "f", "gp", "g", "i"} if a.quick else
                                                 {"a", "b", "c", "d", "e", "f", "gp", "g", "i"})
    table = []
    t_all = time.time()
    for tid, script, tout, py in ORDER:
        if tid not in sel:
            continue
        print(f"\n######## ({tid}) {script} [{py}] ########", flush=True)
        t0 = time.time()
        try:
            p = subprocess.run([py, str(TESTS / script)], timeout=tout, cwd=str(REPO))
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
        table.append((tid, script, rc, time.time() - t0))
    print("\n== Resumen ==")
    for tid, script, rc, dt in table:
        print(f"  ({tid}) {script:<30} {'PASS' if rc == 0 else 'FAIL'} rc={rc} {dt:6.1f} s")
    total = time.time() - t_all
    print(f"  total {total:.1f} s")
    return 0 if all(rc == 0 for _, _, rc, _ in table) else 1


if __name__ == "__main__":
    sys.exit(main())
