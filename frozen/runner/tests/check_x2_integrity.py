#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_x2_integrity.py - integridad de un job X2 del runner frente al .npy original (SIN metricas).

  check_x2_integrity.py --job-dir ABS [--reference ABS]
  (por defecto la referencia se deduce: data/original_results/kitsune/results/<Ataque>/<variante>_ar_<ar>.npy)

Imprime forma, dtype, n_nan, n_inf, sha256 de ambos, array_equal y, si difieren, primer indice distinto, tramo
(FM [0,50k), entrenamiento [50k,200k), ejecucion) y max|d|. Salida 0 si array_equal, 3 si difieren (no es un fallo del
runner: S2, procedencia del codigo), 1 si faltan ficheros o las formas no casan.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--reference", default=None)
    a = ap.parse_args()
    J = Path(a.job_dir)
    cfg = json.loads((J / "config.json").read_text())
    row = cfg["manifest_row"]
    ref = Path(a.reference) if a.reference else (C.ORIG_RESULTS / "kitsune" / "results" / row["dataset"].split(":", 1)[1] /
                                                f"{row['variant']}_ar_{row['ar']}.npy")
    if not (J / "scores.npy").exists() or not ref.exists():
        print(f"faltan ficheros: {J / 'scores.npy'} / {ref}")
        return 1
    run, orig = np.load(J / "scores.npy"), np.load(ref)
    info = {}
    for name, arr, p in (("runner", run, J / "scores.npy"), ("original", orig, ref)):
        info[name] = {"path": str(p), "shape": list(arr.shape), "dtype": str(arr.dtype), "n_nan": int(np.isnan(arr).sum()),
                      "n_inf": int(np.isinf(arr).sum()), "sha256": C.sha256_file(p)}
    print(json.dumps(info, indent=2))
    if run.shape != orig.shape:
        print("FORMAS DISTINTAS")
        return 1
    if np.array_equal(run, orig):
        print("array_equal: True (bit a bit)")
        return 0
    d = np.flatnonzero(run != orig)
    i0 = int(d[0])
    tramo = "FM" if i0 < 50_000 else ("entrenamiento" if i0 < 200_000 else "ejecucion")
    print(f"array_equal: False; primer indice {i0} ({tramo}); filas distintas {len(d)}; max|d|={float(np.nanmax(np.abs(run - orig))):.3e}")
    print("Arbol de decision (S2): FM -> mapa DBSCAN o carga; entrenamiento -> AEs/capa stat; ejecucion -> bloques/BLAS. "
          "Testigo: x2_wrapper.py con el codigo de submitted-v1. No bloquea T1-A-KNAD.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
