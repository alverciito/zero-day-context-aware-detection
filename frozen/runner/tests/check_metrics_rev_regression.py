#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_metrics_rev_regression.py - puerta de regresion NUMERICA de metrics_rev.py sobre los metrics.json REALES ya
calculados. SIN imprimir valores de metricas: es regresion interna, no una lectura de resultados.

  envs/postproc/bin/python check_metrics_rev_regression.py --results-root ABS [--out-root ABS] [--max-jobs N]
      [--blocks T1-A,T1-B] [--json] [--keep]

Por que existe
--------------
Los tests h (metrics_rev, stats) corren sobre entradas SINTETICAS. Cuando se cambia metrics_rev.py o common.py,
version_key cambia (helper_hashes cubre los dos ficheros) y TODOS los metrics.json del arbol se recalculan; la
afirmacion "nada confirmatorio cambia" no queda comprobada contra datos reales por ningun test. Esta puerta
recalcula cada metrics.json existente con el codigo de trabajo, en un directorio temporal, y exige que TODOS los
campos coincidan salvo los que por construccion cambian.

Campos ignorados en la comparacion (y por que)
----------------------------------------------
  created                      marca de tiempo de la escritura
  provenance                   incluye sha256 del propio metrics_rev.py, version_key y la cronologia
  deviations_proposed          texto de las desviaciones declaradas
  job.exploratory_arm          bandera nueva; en un job confirmatorio es False y solo se ANADE
  job.exploratory_arm_rule     idem (None en confirmatorios)
  exploratory_job              solo existe en jobs T7
  timing.*                     tiempos de pared del propio calculo, no son metricas
  *_seconds, *.seconds         idem (primary.sklearn_seconds, sensitivities.<S>.seconds)

Cualquier otra diferencia, numerica o no, es un FAIL. Un FAIL significa que el cambio en el post-proceso NO es
neutro para las cifras ya calculadas.

Salida: 0 PASS; 3 alguna diferencia; 1 error o ningun job.

Coste: un metrics.json por job; con --max-jobs se muestrea. La puerta completa (todos los jobs) va en tmux
y es obligatoria ANTES del tag runner-v3 y antes del siguiente aggregate.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C
import metrics_rev as M

IGNORED_TOP = ("created", "provenance", "deviations_proposed", "exploratory_job", "timing")
IGNORED_JOB = ("exploratory_arm", "exploratory_arm_rule")


def _ignored_path(path):
    """Fields that are not metrics and vary between runs of the computation: wall times and the in-process memoisation flag."""
    return (path.endswith("_seconds") or path.endswith(".seconds") or path.startswith(".timing.")
            or path == ".regions.fit_set.memo")


def _strip(m):
    """Copia de metrics.json sin los campos que por construccion cambian entre versiones."""
    out = {k: v for k, v in (m or {}).items() if k not in IGNORED_TOP}
    if isinstance(out.get("job"), dict):
        out["job"] = {k: v for k, v in out["job"].items() if k not in IGNORED_JOB}
    return out


def _diff(a, b, path="", out=None, tol=0.0):
    """Diferencias recursivas entre dos estructuras JSON. NaN == NaN; los floats se comparan con |d| <= tol."""
    out = [] if out is None else out
    if path and _ignored_path(path):
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if _ignored_path(f"{path}.{k}"):
                continue
            if k not in a:
                out.append(f"{path}.{k}: solo en el NUEVO")
            elif k not in b:
                out.append(f"{path}.{k}: solo en el GUARDADO")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: longitud {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _diff(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return out
        if math.isinf(fa) or math.isinf(fb):
            if fa != fb:
                out.append(f"{path}: {a!r} != {b!r}")
        elif abs(fa - fb) > tol:
            out.append(f"{path}: difiere (|d| > {tol:g})")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=str(C.RESULTS_ROOT))
    ap.add_argument("--out-root", default=None, help="donde recalcular (por defecto, un temporal que se borra)")
    ap.add_argument("--blocks", default=None, help="filtro de bloques por prefijo, separados por comas")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--tol", type=float, default=0.0, help="tolerancia absoluta; 0 = identidad exacta")
    ap.add_argument("--keep", action="store_true", help="no borrar el directorio temporal")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    R = Path(a.results_root)
    if not R.is_absolute():
        print("ERROR: --results-root debe ser absoluto")
        return 1
    blocks = [b.strip() for b in a.blocks.split(",")] if a.blocks else None
    jobs = []
    for d in sorted(R.iterdir()) if R.exists() else []:
        if not d.is_dir() or d.name.startswith((".", "_")) or not (d / "metrics.json").exists():
            continue
        if blocks and not C.block_selected(d.name.split("__")[0], blocks):
            continue
        jobs.append(d)
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    if not jobs:
        print("ERROR: ningun job con metrics.json")
        return 1
    try:
        prov = M.base_provenance()
    except (M.PlanGuardError, M.EnvGuardError) as e:
        print(f"ABORT: {e}")
        return 1
    opts = {"bootstrap_ci": False}
    tmp = Path(a.out_root) if a.out_root else Path(tempfile.mkdtemp(prefix="mrreg_"))
    tmp.mkdir(parents=True, exist_ok=True)
    res, n_fail, n_skip, t0 = [], 0, 0, time.time()
    try:
        for i, J in enumerate(jobs, 1):
            name = J.name
            old = C.read_json(J / "metrics.json") or {}
            out_dir = tmp / name
            try:
                state, new = M.run_one(J, out_dir, opts, prov, force=True, log=lambda *_a, **_k: None,
                                       out_root_for_first=tmp)
            except M.NotApplicable as e:
                res.append({"job_id": name, "state": "no_aplicable", "detail": str(e)[:200]})
                n_skip += 1
                continue
            except Exception as e:
                res.append({"job_id": name, "state": "error", "detail": f"{type(e).__name__}: {e}"[:300]})
                n_fail += 1
                continue
            if state != "ok" or new is None:
                res.append({"job_id": name, "state": f"recalculo {state}"})
                n_fail += 1
                continue
            d = _diff(_strip(new), _strip(old), tol=a.tol)
            if d:
                n_fail += 1
            res.append({"job_id": name, "state": "PASS" if not d else "FAIL", "n_diffs": len(d),
                        "diffs": d[:25], "old_sha256": C.sha256_file(J / "metrics.json"),
                        "new_sha256": C.sha256_file(out_dir / "metrics.json")})
            if not a.json:
                print(f"[{i}/{len(jobs)}] {name}: {'PASS' if not d else f'FAIL ({len(d)} campos)'}", flush=True)
                for x in d[:10]:
                    print(f"    {x}")
    finally:
        if not a.keep and not a.out_root:
            shutil.rmtree(tmp, ignore_errors=True)
    summary = {"results_root": str(R), "n_jobs": len(jobs), "n_fail": n_fail, "n_no_aplicable": n_skip,
               "tol": a.tol, "seconds": round(time.time() - t0, 1),
               "metrics_rev_version": M.METRICS_REV_VERSION,
               "metrics_rev_sha256": prov.get("metrics_rev_sha256"), "helpers_sha256": prov.get("helpers_sha256"),
               "ignored_fields": {"top": list(IGNORED_TOP), "job": list(IGNORED_JOB),
                                  "paths": ["*_seconds", "*.seconds", ".timing.*"]},
               "rule": "ninguna cifra confirmatoria puede cambiar al cambiar el post-proceso",
               "jobs": res}
    if a.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'PASS' if not n_fail else 'FAIL'}: {len(jobs)} jobs, {n_fail} con diferencias, "
              f"{n_skip} no aplicables, {summary['seconds']} s")
    return 0 if not n_fail else 3


if __name__ == "__main__":
    sys.exit(main())
