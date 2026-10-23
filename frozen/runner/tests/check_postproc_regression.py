#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_postproc_regression.py - puerta G-PP antes de lanzar T1-A-KNAD.
SIN imprimir valores de metricas: regresion INTERNA de cifras ya publicadas.

  /usr/bin/python3 check_postproc_regression.py --job-dir ABS [--reference ABS] [--target ABS] [--d25-root ABS]
      [--json-root ABS] [--dataset-name os_scan] [--out-dir ABS] [--detectors MVD,KMD] [--tol 1e-6]
      [--arrays-only] [--no-published]

Interprete: /usr/bin/python3 (numpy 2.4, scipy 1.17, sklearn 1.8, torch 2.10 cpu), el entorno con el que se
reprodujeron 56/56 JSON del envio. envs/tf215 no tiene torch.

Que comprueba, con el codigo ORIGINAL de postprocess/detectors_2025 importado sin modificar (sha256 frente a su
SHA256SUMS):
  T2 (detector reajustado desde las entradas guardadas): stat_detector / kmeans_detector (W = 10 000) y
     median_smoothing / mean_smoothing (50 000) sobre scores.npy del runner y sobre el .npy original: series raw y
     suavizada con |d| <= tol (execute.py:311-312,318-319).
  T1 (metricas publicadas): compute_scores + compute_metrics ORIGINALES (execute.py:53-78,215-281; WindowDiff/Pk con
     torch uint8) sobre scores.npy del runner reproducen los 20 campos de __<Paradigma> - <VARIANTE>_metrics.json del
     envio (experiments = MVD, experiments_kmeans = KMD) con |d| <= tol.
Alineacion de etiquetas identica a execute.py:302-305: target = target[:len(frame)]; si el frame es mas largo, se
recorta por delante. Solo aplica a jobs con pipeline orig (X2): es regresion del pipeline enviado.

Salida: 0 PASS; 3 alguna diferencia > tol (o JSON ausente); 2 no aplicable; 1 faltan ficheros o error.
Coste (OS_Scan completo, N = 1 697 850): ~10-15 min de CPU y ~20 GB de RAM de pico (conv1d uint8 de torch con
k = 11 779) -> tmux. Escribe en --out-dir (por defecto data/cache/postproc_regression/<job_id>/) los
_metrics.json y _values.npy de compute_metrics y regression_summary.json (sin valores de metricas).
Lectura de un FAIL: T2 PASS + T1 FAIL -> entorno o post-proceso (no el runner); T2 FAIL -> scores distintos del
original (ver check_x2_integrity.py y el arbol de decision S2).
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C

W_DET, W_SMOOTH, W_LABEL = 10_000, 50_000, 10_000
KEYS = ["Precision Raw", "Recall Raw", "F1-Score Raw", "Window Diff Raw", "Pk Raw", "TP Raw", "FP Raw", "TN Raw",
        "FN Raw", "Threshold Raw", "Precision Smooth", "Recall Smooth", "F1-Score Smooth", "Window Diff Smooth",
        "Pk Smooth", "TP Smooth", "FP Smooth", "TN Smooth", "FN Smooth", "Threshold Smooth"]
D25_FILES = ["execute.py", "src/detectors/__init__.py", "src/detectors/second_order_detector.py",
             "src/detectors/median_smoother.py", "src/detectors/metrics.py"]
SUBDIR = {"MVD": "experiments", "KMD": "experiments_kmeans"}
NOTE = "regresion de cifras ya publicadas; sin valores de metricas"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--d25-root", default=str(C.MAIN_REPO / "postprocess" / "detectors_2025"))
    ap.add_argument("--json-root", default=None, help="Raiz con experiments/ y experiments_kmeans/ (por defecto --d25-root).")
    ap.add_argument("--dataset-name", default=None, help="Carpeta del post-proceso (por defecto el ataque en minusculas).")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--detectors", default="MVD,KMD")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--arrays-only", action="store_true", help="Solo T2 (sin umbrales ni metricas).")
    ap.add_argument("--no-published", action="store_true", help="Calcula las metricas pero no compara con los JSON.")
    return ap.parse_args(argv)


def sha256_array(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def verify_d25(d25):
    """sha256 de los ficheros del post-proceso frente a su SHA256SUMS (el codigo debe ser el enviado)."""
    sums = {}
    p = d25 / "SHA256SUMS"
    if p.exists():
        for ln in p.read_text(errors="replace").splitlines():
            parts = ln.strip().split(None, 1)
            if len(parts) == 2 and len(parts[0]) == 64:
                sums[parts[1].lstrip("*").strip().lstrip("./")] = parts[0]
    res, ok = {}, True
    for f in D25_FILES:
        fp = d25 / f
        if not fp.exists():
            res[f] = "falta"
            ok = False
            continue
        h = C.sha256_file(fp)
        exp = sums.get(f)
        res[f] = {"sha256": h, "in_SHA256SUMS": exp is not None, "match": (exp == h) if exp else None}
        if exp is not None and exp != h:
            ok = False
    return ok, res


def load_d25(d25):
    os.environ.setdefault("MPLBACKEND", "Agg")
    for mod in [m for m in list(sys.modules) if m == "src" or m.startswith("src.") or m == "detectors" or m.startswith("detectors.")]:
        del sys.modules[mod]
    for p in (str(d25 / "src"), str(d25)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("d25_execute", str(d25 / "execute.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    for fn in (mod.stat_detector, mod.kmeans_detector, mod.median_smoothing, mod.mean_smoothing, mod.label_smoother,
               mod.wd_pk):
        src = sys.modules[fn.__module__].__file__
        if not C.is_under(src, d25):
            raise RuntimeError(f"{fn.__name__} importado desde {src}, fuera de {d25}")
    return mod


def max_abs_diff(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None, False
    eq = bool(np.array_equal(a, b, equal_nan=True))
    if eq:
        return 0.0, True
    with np.errstate(invalid="ignore"):
        d = np.abs(a - b)
    nan_mismatch = bool((np.isnan(a) != np.isnan(b)).any())
    fin = d[np.isfinite(d)]
    return (float(fin.max()) if fin.size else 0.0), (not nan_mismatch)


def value_diff(x, y):
    x, y = float(x), float(y)
    if math.isnan(x) and math.isnan(y):
        return 0.0
    return abs(x - y)


def compare_published(m, js, tol):
    """Compara los 20 campos de compute_metrics con un JSON publicado. Devuelve solo diferencias (sin valores)."""
    diffs = {k: value_diff(m[k], js[k]) for k in KEYS if k in js and k in m}
    missing = [k for k in KEYS if k not in js or k not in m]
    worst = max(diffs, key=diffs.get) if diffs else None
    over = [k for k in KEYS if k in diffs and diffs[k] > tol]
    return {"n_keys": len(diffs), "n_exact": sum(1 for v in diffs.values() if v == 0),
            "max_abs_diff": diffs.get(worst) if worst else None, "worst_key": worst, "keys_over_tol": over,
            "missing_keys": missing, "pass": bool(not missing and not over)}


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    J = Path(a.job_dir)
    for pth, what in ((a.job_dir, "--job-dir"), (a.reference, "--reference"), (a.target, "--target"),
                      (a.d25_root, "--d25-root"), (a.json_root, "--json-root"), (a.out_dir, "--out-dir")):
        if pth is not None and not os.path.isabs(pth):
            print(f"ERROR: {what} debe ser absoluta: {pth}")
            return 1
    cfg = C.read_json(J / "config.json")
    if not cfg or not (J / "scores.npy").exists():
        print(f"ERROR: faltan {J / 'config.json'} o {J / 'scores.npy'}")
        return 1
    row = cfg.get("manifest_row") or {}
    if row.get("pipeline") != "orig":
        print(f"NO APLICABLE: pipeline={row.get('pipeline')!r}; G-PP es regresion del pipeline enviado (solo X2/orig)")
        return 2
    ds = row.get("dataset", "")
    if not ds.upper().startswith("KNAD:") and not a.dataset_name:
        print(f"NO APLICABLE en v1: dataset {ds!r} (solo KNAD; para otros, --dataset-name y --reference explicitos)")
        return 2
    attack = ds.split(":", 1)[1] if ":" in ds else ds
    dsname = a.dataset_name or attack.lower()
    variant, ar = row["variant"], C.parse_bool(row["ar"])
    name = f"{'Autoregressive' if ar else 'Full Sequence'} - {variant.upper()}"
    d25 = Path(a.d25_root)
    json_root = Path(a.json_root) if a.json_root else d25
    ref_p = Path(a.reference) if a.reference else (C.ORIG_RESULTS / "kitsune" / "results" / attack / f"{variant}_ar_{ar}.npy")
    tgt_p = Path(a.target) if a.target else (C.MAIN_REPO / "data" / "remote_chiquitin" / "detectors_2025" / "data" / dsname / "target.npy")
    out = Path(a.out_dir) if a.out_dir else (C.MAIN_REPO / "data" / "cache" / "postproc_regression" / J.name)
    dets = [d.strip().upper() for d in a.detectors.split(",") if d.strip()]
    if any(d not in SUBDIR for d in dets):
        print(f"ERROR: detectores validos {sorted(SUBDIR)}")
        return 1
    try:
        C.check_not_forbidden(out, "--out-dir")
        for bad in (d25, C.MAIN_REPO / "postprocess", C.MAIN_REPO / "src", C.MAIN_REPO / "experiments"):
            if C.is_under(out, bad):
                raise C.EnvError(f"--out-dir {out} no puede estar bajo {bad}.")
    except C.EnvError as e:
        print(f"ERROR: {e}")
        return 1
    for p in (ref_p, tgt_p, d25 / "execute.py"):
        if not p.exists():
            print(f"ERROR: falta {p}")
            return 1
    d25_ok, d25_hashes = verify_d25(d25)
    if not d25_ok:
        print(f"ERROR: el codigo del post-proceso no coincide con su SHA256SUMS: {json.dumps(d25_hashes)[:600]}")
        return 1

    import numpy as np
    E = load_d25(d25)
    import scipy
    import sklearn
    import torch
    run = np.load(J / "scores.npy")
    ref = np.load(ref_p, allow_pickle=True)
    tgt_full = np.load(tgt_p, allow_pickle=True).astype(int)
    frames = {}
    for tag, fr in (("runner", run), ("reference", ref)):
        t = tgt_full[:len(fr)]
        if len(fr) - len(t) > 0:
            fr = fr[len(fr) - len(t):]
        frames[tag] = (fr, t)
    summary = {"note": NOTE, "job_dir": str(J), "job_id": cfg.get("job_id"), "name": name, "dataset_name": dsname,
               "tol": a.tol, "detectors": dets, "arrays_only": a.arrays_only, "no_published": a.no_published,
               "inputs": {"runner_scores": {"path": str(J / "scores.npy"), "shape": list(run.shape), "sha256": C.sha256_file(J / "scores.npy")},
                          "reference_npy": {"path": str(ref_p), "shape": list(ref.shape), "sha256": C.sha256_file(ref_p)},
                          "target": {"path": str(tgt_p), "shape": list(tgt_full.shape), "sha256": C.sha256_file(tgt_p)},
                          "runner_equals_reference_bytes": bool(run.shape == ref.shape and np.array_equal(run, ref))},
               "d25_code": d25_hashes, "json_root": str(json_root), "out_dir": str(out),
               "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                            "sklearn": sklearn.__version__, "torch": torch.__version__},
               "alignment": "execute.py:302-305 (target[:len(frame)]; recorte del frame por delante)", "results": {}}
    if frames["runner"][0].shape != frames["reference"][0].shape or not np.array_equal(frames["runner"][1], frames["reference"][1]):
        summary["results"]["shapes"] = {"pass": False, "runner": list(frames["runner"][0].shape),
                                        "reference": list(frames["reference"][0].shape)}
        print(json.dumps(summary["results"], indent=2))
        print("FAIL: formas o etiquetas alineadas distintas entre runner y referencia")
        _write_summary(out, summary)
        return 3
    target = frames["runner"][1]
    target_smooth = E.label_smoother(target, W_LABEL)
    all_pass = True
    for det in dets:
        fn = E.stat_detector if det == "MVD" else E.kmeans_detector
        sm_fn = E.median_smoothing if det == "MVD" else E.mean_smoothing
        td = time.time()
        d_run = fn(frames["runner"][0], window_length=W_DET)
        s_run = sm_fn(d_run, window_length=W_SMOOTH)
        d_ref = fn(frames["reference"][0], window_length=W_DET)
        s_ref = sm_fn(d_ref, window_length=W_SMOOTH)
        md_raw, nan_ok_raw = max_abs_diff(d_run, d_ref)
        md_sm, nan_ok_sm = max_abs_diff(s_run, s_ref)
        t2 = {"raw_max_abs_diff": md_raw, "smooth_max_abs_diff": md_sm,
              "raw_sha256_equal": sha256_array(d_run) == sha256_array(d_ref),
              "smooth_sha256_equal": sha256_array(s_run) == sha256_array(s_ref),
              "pass": bool(md_raw is not None and md_sm is not None and nan_ok_raw and nan_ok_sm and
                           md_raw <= a.tol and md_sm <= a.tol), "seconds": round(time.time() - td, 1)}
        rec = {"T2_detector_arrays": t2}
        all_pass &= t2["pass"]
        del d_ref, s_ref
        if not a.arrays_only:
            tm = time.time()
            th_raw = E.compute_scores(d_run, target)[4]
            th_sm = E.compute_scores(s_run, target_smooth)[4]
            odir = out / SUBDIR[det] / dsname
            odir.mkdir(parents=True, exist_ok=True)
            m = E.compute_metrics(d_run, s_run, target, target_smooth, th_raw, th_sm, str(odir), name)
            t1 = {"metrics_json_written": str(odir / f"__{name}_metrics.json"), "seconds": round(time.time() - tm, 1)}
            if not a.no_published:
                pub = json_root / SUBDIR[det] / dsname / f"__{name}_metrics.json"
                t1["published_json"] = str(pub)
                if not pub.exists():
                    t1.update(pass_=False, error="JSON publicado no encontrado")
                    t1["pass"] = t1.pop("pass_")
                else:
                    t1.update(compare_published(m, json.loads(pub.read_text()), a.tol))
                all_pass &= bool(t1["pass"])
            rec["T1_published_json"] = t1
            del m
        summary["results"][det] = rec
        del d_run, s_run
    summary["pass"] = bool(all_pass)
    summary["seconds"] = round(time.time() - t0, 1)
    _write_summary(out, summary)
    print(json.dumps({"results": summary["results"], "runner_equals_reference_bytes":
                      summary["inputs"]["runner_equals_reference_bytes"], "seconds": summary["seconds"],
                      "summary": str(out / "regression_summary.json"), "note": NOTE}, indent=2, ensure_ascii=False))
    print("G-PP PASS" if all_pass else "G-PP FAIL (ver regression_summary.json; no bloquea por si solo si T2 PASS: "
                                       "diferencia de entorno o post-proceso)")
    return 0 if all_pass else 3


def _write_summary(out, summary):
    out.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(out / "regression_summary.json", summary)


if __name__ == "__main__":
    sys.exit(main())
