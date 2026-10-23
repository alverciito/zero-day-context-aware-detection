#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(d) Prefix of the runner with the original pipeline and real hyper-parameters against the original OS Scan scores."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _util as U
from _util import C

import numpy as np

PHASES = [(0, 50_000, "FM"), (50_000, 200_000, "entrenamiento (in-sample)"), (200_000, 10**12, "ejecucion")]


def phase_of(i):
    return next(name for a, b, name in PHASES if a <= i < b)


def compare(ck, cid, run, ref, n, strict_equal):
    ref = ref[:n] if n else ref
    if run.shape != ref.shape:
        ck.check(cid, False, f"forma runner {run.shape} != original {ref.shape}")
        return
    if np.array_equal(run, ref):
        ck.check(cid, True, f"array_equal con el .npy original en {len(run)} filas")
        return
    diff = np.flatnonzero(run != ref)
    i0 = int(diff[0])
    md = float(np.nanmax(np.abs(run - ref)))
    by_phase = {name: int(((diff >= a) & (diff < b)).sum()) for a, b, name in PHASES}
    detail = (f"DIFIERE: primer indice {i0} ({phase_of(i0)}), max|d|={md:.3e}, filas distintas por tramo {by_phase}. "
              "Arbol: FM -> mapa DBSCAN/carga distintos; entrenamiento -> capa stat/AEs; ejecucion -> bloques/BLAS. "
              "Testigo: x2_wrapper.py --dataset KNAD:OS_Scan --variant <v> --ar True")
    if by_phase["FM"] > 0:
        ck.check(cid, False, detail)
    elif strict_equal:
        ck.warn(cid, detail + " -> NO bloquea T1 (S2: procedencia del codigo)")
    else:
        ck.warn(cid, detail + " (informativo)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--variants", default="stat,original")
    a = ap.parse_args()
    ck = U.Checker("test_d_prefix_original")
    root = U.make_root("d")
    R = root / "results"
    n = None if a.full else 260_000
    try:
        rows = [U.make_row(variant=v, priority=i) for i, v in enumerate(a.variants.split(","))]
        man = root / "manifest.csv"
        U.write_manifest(man, rows)
        extra = ["--data-root", str(C.DATA_ROOT)] + ([] if a.full else ["--max-rows", n])
        t0 = time.time()
        procs = {r["variant"]: U.popen_job(root, man, r["job_id"], extra, log_path=root / "logs" / f"{r['variant']}.log")
                 for r in rows}
        rcs = {v: p.wait(timeout=3600) for v, p in procs.items()}
        ck.info(f"runner: {rcs} en {time.time() - t0:.0f} s")
        for r in rows:
            v = r["variant"]
            J = R / r["job_id"]
            if rcs[v] != 0 or not (J / "DONE").exists():
                ck.check(f"d.{v}.done", False, f"rc={rcs[v]}\n{C.tail_text(root / 'logs' / f'{v}.log', 2000)}")
                continue
            run = np.load(J / "scores.npy")
            ref = np.load(C.ORIG_RESULTS / "kitsune" / "results" / "OS_Scan" / f"{v}_ar_True.npy", mmap_mode="r")
            if a.full:
                ck.check(f"d.{v}.length", len(run) == len(ref), f"runner {len(run)} vs original {len(ref)}")
            ok_zero = bool(np.all(run[:50_000] == 0))
            ck.check(f"d.{v}.fm_zeros", ok_zero, "[0, 50000) ceros exactos")
            fs = U.load_json(J / "fm_stats.json")
            ck.info(f"{v}: grupos DBSCAN {[len(g) for g in fs['feature_map']]} en {fs['iterations']} iteraciones "
                    f"(mapa_modelo.md §8.3: OS Scan [82, 15, 8, 10] en 15)")
            compare(ck, f"d{'1' if v == 'stat' else '2'}.{v}", run, np.asarray(ref), len(run), strict_equal=(v == "stat"))
    except Exception:
        import traceback
        traceback.print_exc()
        ck.check("d.exception", False, "excepcion en el test")
    rc = ck.finish()
    U.cleanup(root, rc == 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
