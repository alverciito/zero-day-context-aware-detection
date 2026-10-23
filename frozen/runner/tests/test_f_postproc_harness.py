#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(f) Regression gate of the original post-processing (check_postproc_regression.py) exercised with synthetic data."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _util as U
from _util import C

import numpy as np

HARNESS = U.TESTS_DIR / "check_postproc_regression.py"
D25 = U.REPO / "postprocess" / "detectors_2025"
NAME = "__Autoregressive - STAT_metrics.json"


def run(args, timeout=900):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([U.SYS_PY, str(HARNESS)] + [str(x) for x in args], capture_output=True, text=True,
                       timeout=timeout, env=env)
    return p.returncode, p.stdout + p.stderr


def summary(out):
    p = Path(out) / "regression_summary.json"
    return U.load_json(p) if p.exists() else {}


def main():
    ck = U.Checker("test_f_postproc_harness")
    try:
        import torch
    except ImportError:
        ck.check("f.torch", False, f"{sys.executable} sin torch: ejecutar con {U.SYS_PY}")
        return ck.finish()
    root = U.make_root("f")
    try:
        N = 230_000
        rng = np.random.default_rng(7)
        scores = np.abs(rng.normal(0.05, 0.01, N))
        scores[:50_000] = 0.0
        labels = np.zeros(N + 1, dtype=np.int64)
        labels[210_000:216_000] = 1
        scores[210_000:216_000] += 0.2
        J = root / "results" / "X2__fake"
        J.mkdir(parents=True)
        np.save(J / "scores.npy", scores)
        C.write_json_atomic(J / "config.json", {"job_id": "X2__fake", "manifest_row": {
            "pipeline": "orig", "dataset": "KNAD:OS_Scan", "variant": "stat", "ar": "True"}})
        ref = root / "data" / "ref.npy"
        tgt = root / "data" / "target.npy"
        np.save(ref, scores)
        np.save(tgt, labels)
        base = ["--job-dir", J, "--reference", ref, "--target", tgt, "--d25-root", D25]

        out1 = root / "out1"
        rc, out = run(base + ["--out-dir", out1, "--no-published", "--detectors", "MVD,KMD"])
        s1 = summary(out1)
        res = s1.get("results", {})
        t2 = {d: (res.get(d, {}).get("T2_detector_arrays") or {}) for d in ("MVD", "KMD")}
        t2ok = (t2["MVD"].get("pass") and t2["MVD"].get("raw_sha256_equal") and t2["MVD"].get("smooth_sha256_equal")
                and t2["KMD"].get("pass") and (t2["KMD"].get("raw_max_abs_diff") or 0.0) <= 1e-9)
        ck.info(f"KMD reajustado sobre entrada identica: max|d| raw={t2['KMD'].get('raw_max_abs_diff')} "
                f"smooth={t2['KMD'].get('smooth_max_abs_diff')} (sha256 igual: {t2['KMD'].get('raw_sha256_equal')})")
        written = [out1 / sub / "os_scan" / NAME for sub in ("experiments", "experiments_kmeans")]
        ck.check("f1.identical_input", rc == 0 and s1.get("pass") is True and t2ok and all(p.exists() for p in written),
                 f"rc={rc} pass={s1.get('pass')} T2={t2ok} json={[p.exists() for p in written]}"
                 + ("" if rc == 0 else f"\n{out[-2500:]}"))
        ck.check("f1.d25_code_verified", all(isinstance(v, dict) and v.get("match") is True
                                             for v in (s1.get("d25_code") or {}).values()) and s1.get("d25_code"),
                 "sha256 de execute.py y src/detectors/*.py = SHA256SUMS de detectors_2025")
        leaked = []
        for p in written:
            if p.exists():
                js = json.loads(p.read_text())
                for k in ("F1-Score Raw", "F1-Score Smooth", "Precision Smooth", "Recall Smooth", "Window Diff Smooth",
                          "Pk Smooth", "Threshold Smooth"):
                    v = float(js[k])
                    if 0 < v < 1:
                        leaked += [s for s in (repr(v), f"{v:.4f}", f"{v:.6f}") if s in out]
        ck.check("f1.no_metric_values_printed", not leaked, f"valores de metricas en stdout: {leaked[:3]}")

        pub = root / "pub"
        for sub in ("experiments", "experiments_kmeans"):
            (pub / sub / "os_scan").mkdir(parents=True)
            if (out1 / sub / "os_scan" / NAME).exists():
                shutil.copy(out1 / sub / "os_scan" / NAME, pub / sub / "os_scan" / NAME)
        out2 = root / "out2"
        rc, out = run(base + ["--out-dir", out2, "--json-root", pub, "--detectors", "MVD"])
        t1 = (summary(out2).get("results", {}).get("MVD") or {}).get("T1_published_json") or {}
        ck.check("f2.published_pass", rc == 0 and t1.get("pass") is True and t1.get("n_keys") == 20 and t1.get("n_exact") == 20,
                 f"rc={rc} T1={ {k: t1.get(k) for k in ('pass', 'n_keys', 'n_exact', 'keys_over_tol')} }")

        spec = importlib.util.spec_from_file_location("cpr", str(HARNESS))
        cpr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cpr)
        js = json.loads((pub / "experiments" / "os_scan" / NAME).read_text())
        tampered = dict(js, **{"F1-Score Smooth": float(js["F1-Score Smooth"]) + 1e-3})
        cmp_ok = cpr.compare_published(js, js, 1e-6)
        cmp_bad = cpr.compare_published(js, tampered, 1e-6)
        ck.check("f3.compare_detects_field", cmp_ok["pass"] and not cmp_bad["pass"] and
                 cmp_bad["keys_over_tol"] == ["F1-Score Smooth"] and cmp_bad["n_exact"] == 19,
                 f"ok={cmp_ok['pass']} bad={ {k: cmp_bad[k] for k in ('pass', 'keys_over_tol', 'n_exact')} }")

        ref2 = root / "data" / "ref2.npy"
        r2 = scores.copy()
        r2[205_000:] *= 1.5
        np.save(ref2, r2)
        out4 = root / "out4"
        rc, out = run(["--job-dir", J, "--reference", ref2, "--target", tgt, "--d25-root", D25, "--out-dir", out4,
                       "--arrays-only", "--detectors", "MVD,KMD"])
        r4 = summary(out4).get("results", {})
        ck.check("f4.detects_input_diff", rc == 3 and all((r4.get(d, {}).get("T2_detector_arrays") or {}).get("pass") is False
                                                           for d in ("MVD", "KMD")),
                 f"rc={rc} T2 pass={[(r4.get(d, {}).get('T2_detector_arrays') or {}).get('pass') for d in ('MVD', 'KMD')]}")

        J5 = root / "results" / "T1__fake"
        J5.mkdir(parents=True)
        np.save(J5 / "scores.npy", scores)
        C.write_json_atomic(J5 / "config.json", {"job_id": "T1__fake", "manifest_row": {
            "pipeline": "corrected", "dataset": "KNAD:OS_Scan", "variant": "stat", "ar": "True"}})
        rc5, _ = run(["--job-dir", J5, "--reference", ref, "--target", tgt, "--d25-root", D25, "--out-dir", root / "out5"])
        bad_out = D25 / "_nunca_test_f"
        rc6, _ = run(base + ["--out-dir", bad_out, "--arrays-only"])
        ck.check("f5.guards", rc5 == 2 and rc6 == 1 and not bad_out.exists(), f"corrected rc={rc5}; out-dir en D25 rc={rc6}")
    except Exception:
        import traceback
        traceback.print_exc()
        ck.check("f.exception", False, "excepcion en el test")
    rc = ck.finish()
    U.cleanup(root, rc == 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
