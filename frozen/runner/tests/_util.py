#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilidades de los tests del runner v1 (sin pytest). Raices temporales en data/cache/_tests/<ts>-<pid>-<tag>/."""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
RUNNER_DIR = TESTS_DIR.parent
sys.path.insert(0, str(RUNNER_DIR))
import common as C

PY = sys.executable
SYS_PY = "/usr/bin/python3"
REPO = C.MAIN_REPO
CODE_ROOT = C.DEFAULT_CODE_ROOT
KNAD_OS = C.DATA_ROOT / "KITSUNE" / "OS Scan"


class Checker:
    def __init__(self, name):
        self.name = name
        self.results = []
        self.t0 = time.time()

    def check(self, cid, ok, detail=""):
        ok = bool(ok)
        self.results.append((cid, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {detail}", flush=True)
        return ok

    def warn(self, cid, detail):
        self.results.append((cid, None, detail))
        print(f"[WARN] {cid}: {detail}", flush=True)

    def info(self, msg):
        print(f"       {msg}", flush=True)

    @property
    def ok(self):
        return all(r[1] is not False for r in self.results)

    def finish(self):
        n_ok = sum(1 for r in self.results if r[1] is True)
        n_fail = sum(1 for r in self.results if r[1] is False)
        n_warn = sum(1 for r in self.results if r[1] is None)
        print(f"== {self.name}: {n_ok} PASS, {n_fail} FAIL, {n_warn} WARN en {time.time() - self.t0:.1f} s ==", flush=True)
        return 0 if n_fail == 0 else 1


def make_root(tag):
    base = C.TEST_ROOT_BASE / f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{tag}"
    for sub in ("results", "cache", "logs", "data"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def rmtree_rw(path):
    path = Path(path)
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o755)
            except OSError:
                pass
        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o644)
            except OSError:
                pass
    os.chmod(path, 0o755)
    shutil.rmtree(path, ignore_errors=True)


def cleanup(root, ok):
    if ok and not os.environ.get("JISA_KEEP_TEST_ROOT"):
        rmtree_rw(root)
        print(f"       raiz temporal borrada: {root}")
    else:
        print(f"       raiz temporal CONSERVADA para diagnostico: {root}")


def make_row(block="TEST", priority=1, dataset="KNAD:OS_Scan", variant="_fake_ok", ar=True, pipeline="orig",
             seed="None", rep=1, window=800, save_ckpt=0, timing_ref=0, notes="test"):
    r = {"block": block, "priority": str(priority), "dataset": dataset, "variant": variant, "ar": str(bool(ar)),
         "paradigm": "AR" if ar else "TSR", "pipeline": pipeline, "pipeline_version": "v1", "window": str(window),
         "k_clusters": "4", "grouping": "dbscan", "partition_seed": "NA", "warmup_frac": "NA", "contamination": "0",
         "decontaminate": "0", "seed": str(seed), "rep": str(rep), "save_ckpt": str(save_ckpt),
         "timing_ref": str(timing_ref), "notes": notes}
    r["job_id"] = C.derive_job_id(r)
    return r


def write_manifest(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=C.MANIFEST_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in C.MANIFEST_COLUMNS})


def base_env(**extra):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1", TF_CPP_MIN_LOG_LEVEL="2",
               MPLBACKEND="Agg", JISA_HEARTBEAT_S="1", PYTHONUNBUFFERED="1")
    for k in ("PYTHONHASHSEED", "JISA_SEED_REEXEC", "JISA_NOW"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def job_cmd(root, manifest, job_id, extra=(), test_mode=True):
    cmd = [PY, str(RUNNER_DIR / "run_job.py"), "--job-id", job_id, "--manifest", str(manifest),
           "--results-root", str(Path(root) / "results"), "--gpu", "cpu", "--cache-root", str(Path(root) / "cache")]
    if test_mode:
        cmd.append("--test-mode")
    return cmd + [str(e) for e in extra]


def run_job(root, manifest, job_id, extra=(), env=None, timeout=900, test_mode=True):
    p = subprocess.run(job_cmd(root, manifest, job_id, extra, test_mode), env=env or base_env(), capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def popen_job(root, manifest, job_id, extra=(), env=None, log_path=None, test_mode=True):
    fh = open(log_path, "w") if log_path else subprocess.DEVNULL
    return subprocess.Popen(job_cmd(root, manifest, job_id, extra, test_mode), env=env or base_env(), stdout=fh,
                            stderr=subprocess.STDOUT)


def worker_cmd(root, manifest, extra=()):
    return [PY, str(RUNNER_DIR / "worker.py"), "--gpu", "cpu", "--manifest", str(manifest),
            "--results-root", str(Path(root) / "results"), "--logs-root", str(Path(root) / "logs"),
            "--cache-root", str(Path(root) / "cache"), "--once", "--test-mode", "--idle-sleep-s", "0.5",
            "--env-sleep-s", "0.5", "--dirty-sleep-s", "0.5", "--progress-every-s", "5"] + [str(e) for e in extra]


def run_worker(root, manifest, extra=(), env=None, timeout=300):
    p = subprocess.run(worker_cmd(root, manifest, extra), env=env or base_env(), capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def wait_for(pred, timeout=30, step=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(step)
    return False


def load_json(p):
    return json.loads(Path(p).read_text())


def copy_head_lines(src, dst, n_lines):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for _ in range(n_lines):
            ln = fi.readline()
            if not ln:
                break
            fo.write(ln)
