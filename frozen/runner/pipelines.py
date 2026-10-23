#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipelines.py - loading and construction of the (x, y) streams of the job runner, with a sha256-checked cache."""
import ast
import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

PIPELINE_FORMAT = 1
TRAIN_ROWS = 200_000
ORIG_PREFIX = "/mnt/8A3A82BA3A82A335/database"
ORIG_SCRIPTS = {"cic2017": ("experiments/frames/CIC2017/cic2017.py", 8),
                "cic2018": ("experiments/frames/CIC2018/cic2018.py", 7),
                "aci": ("experiments/frames/ACI-IOT-2023/aci_iot_2023_experiment.py", 1)}
ORIG_EXPECTED_N = {"aci": 1_531_411, "cic2017": 3_130_743, "cic2018": 7_640_018}

SEMANTICS = {
    ("knad", "orig"): ("KitsuneDatabase(regularized=False).get(<ataque>) con KITSUNE_DATABASE_PATHS=[data_root/KITSUNE/"
                       "<ataque>] (src/database/test_database/database.py:10-20,26-31,51-58): pd.read_csv con header=0 "
                       "consume el paquete 0 como cabecera (x: m-1 filas); y = ultima columna de *_labels.csv (m filas) "
                       "truncada a len(x) como launch_all.py:40-42. Desfase +1: x[i] = paquete i+1, y[i] = etiqueta "
                       "del paquete i (mapa_datos.md §1.1)."),
    ("knad", "corrected"): ("x = pd.read_csv(dataset, header=None).to_numpy() (m filas, 115 columnas); y = "
                            "pd.read_csv(labels)['x'] (m filas), alineadas fila a fila; entrenamiento [0, 200000) = "
                            "FM [0, 50000) + AE [50000, 200000) validado con etiquetas como 100 % benigno (solo "
                            "aserción; el tramo se fija a priori); tramo posterior al entrenamiento post_train = "
                            "[200000, m), que NO es una region de evaluacion (los pares calib/eval disjuntos estan "
                            "en config.json.regions.detector_protocol_candidates); orden de captura sin reordenar."),
    ("aci", "orig"): "aci_iot_2023_experiment.py:26-30: data_loader -> grant_first_label(300000) -> regularize_data",
    ("cic2017", "orig"): "cic2017.py:44-57: data_loader por fichero (orden de PATHS), concatenacion, grant_first_label"
                         "(300000), regularize_data con are_categories del ultimo fichero",
    ("cic2018", "orig"): "cic2018.py:36-50: data_loader(path, skip=1_000_000) sobre 7 rutas, concatenacion, "
                         "grant_first_label(300000), regularize_data",
}
_CORR_COMMON = (
    " Etapa 1 (solo etiquetas/timestamps) con warmup_audit.py de T0 (labels_csv, corrected_variants, build_corrected): "
    "exclusiones, orden, corte de entrenamiento [0, T) y guarda de empates identicos a t0/stream_cuts.json. Guarda "
    "benigna: 0 ataques en [0, T), 0 filas descartadas por el filtro de etiqueta y T = common.CORRECTED_TRAIN_ROWS; si "
    "no, PipelineError (HALT). Sin copias grant_first_label (C1). Etapa 2: features crudas sin identificadores; "
    "preprocesado ajustado SOLO con [0, T) (C5): fillna(0) (constante, launch_all.py:98), +-inf -> maximo finito de la "
    "columna en [0, T) (regla de launch_all.py:118-120), z-score (x - media[0,T)) / (desv[0,T) + 1e-6) "
    "(launch_all.py:150-153) y, en categoricas, vocabulario de [0, T) con codigo reservado para lo no visto, uint8 y "
    "unpackbits (launch_all.py:146-148); columnas numericas y despues los bits. Columnas constantes en [0, T): la misma "
    "formula da (x - mu) / 1e-6 en la ejecucion (se cuentan las filas afectadas y se declara; los AEs Keras recortan a "
    "min/max de entrenamiento, p. ej. multilayer.py:98). y[i] = etiqueta de la fila x[i] "
    "(label != benigno), alineadas fila a fila como KNAD corrected (sin desfase). post_train = [T, N) NO es una region "
    "de evaluacion (pares calib/eval en config.json.regions; M1). aux_dup_train: fila de [T, N) cuyo vector crudo es "
    "identico (-0 = +0, NaN = NaN; hash de T0 + comprobacion exacta) a alguna fila de [0, T) (plan 2.1 F_cal; P-14), "
    "con la referencia de common.CORRECTED_DUP_REFERENCE registrada sin HALT. Con los datos de produccion, N, T, primer "
    "ataque, ataques, exclusiones, empates y dimension deben coincidir con common.CORRECTED_EXPECTED (si no, HALT).")
SEMANTICS.update({
    ("cic2017", "corrected"): "variante mon_first: 8 CSV MachineLearningCVE en orden cronologico "
                              "(lunes como preludio; orden de filas del CSV dentro de cada fichero), "
                              "sin timestamp." + _CORR_COMMON,
    ("cic2018", "corrected"): "variante days7: 7 dias del manuscrito, reloj de 12 h corregido (01-07 "
                              "h -> 13-19 h), sin la cabecera repetida ni las 14 filas de 1970, orden "
                              "por (timestamp corregido, fichero, fila); entrenamiento = 200.000 "
                              "primeras benignas; la evaluacion empieza en el primer flujo "
                              "estrictamente posterior (3 empates fuera)." + _CORR_COMMON,
    ("aci", "corrected"): "variante oct31_train_chrono: timestamps resueltos (3 formatos), orden "
                          "cronologico sin el 30-oct; entrenamiento = dia 31-oct completo (72.264 "
                          "filas: FM 50.000 + AE 22.264); sin Flow ID, Src IP, Dst IP, Timestamp ni "
                          "Src Port; Connection Type categorica." + _CORR_COMMON,
})


def knad_files(data_root, spec):
    d = Path(data_root) / "KITSUNE" / spec.attack_dir
    return d / f"{spec.attack}_dataset.csv", d / f"{spec.attack}_labels.csv"


def orig_script_paths(code_root, kind, data_root):
    rel, expected = ORIG_SCRIPTS[kind]
    src = (Path(code_root) / rel).read_text()
    paths = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PATHS" for t in node.targets):
            paths = ast.literal_eval(node.value)
    if not paths or len(paths) != expected:
        raise C.PipelineError(f"{rel}: PATHS con {0 if not paths else len(paths)} rutas activas, se esperaban {expected}")
    out = []
    for p in paths:
        if not p.startswith(ORIG_PREFIX):
            raise C.PipelineError(f"{rel}: ruta sin el prefijo original: {p}")
        out.append(str(Path(data_root) / p[len(ORIG_PREFIX):].lstrip("/")))
    return out


def count_label_rows(path):
    """Filas de datos de un CSV con cabecera (lineas - 1), contando saltos de linea."""
    n, last = 0, b"\n"
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            n += b.count(b"\n")
            last = b[-1:]
    if last != b"\n":
        n += 1
    return n - 1


def build_knad_orig_class(data_root, spec):
    """Clase original (database.py) + truncado de launch_all.py:40-42."""
    import src.database.test_database.database as kdb
    old = kdb.KITSUNE_DATABASE_PATHS
    kdb.KITSUNE_DATABASE_PATHS = [str(Path(data_root) / "KITSUNE" / spec.attack_dir)]
    try:
        db = kdb.KitsuneDatabase(regularized=False)
        x, y = db.get(spec.attack)
    finally:
        kdb.KITSUNE_DATABASE_PATHS = old
    if len(x) != len(y):
        y = y[:len(x)]
    return x, y, {}


def build_knad_orig_replica(data_root, spec, max_rows):
    """Replica de database.py:51-55 con nrows (solo test-mode; nunca se cachea). Test b1: replica == clase."""
    dataset, labels = knad_files(data_root, spec)
    x_df = pd.read_csv(dataset, nrows=max_rows)
    y_df = pd.read_csv(labels, nrows=None if max_rows is None else max_rows + 1)
    x = x_df.to_numpy()
    y = y_df.to_numpy()[:, -1]
    if len(x) != len(y):
        y = y[:len(x)]
    return x, y, {}


def build_knad_corrected(data_root, spec, train_rows=TRAIN_ROWS, max_rows=None):
    dataset, labels = knad_files(data_root, spec)
    x = pd.read_csv(dataset, header=None, nrows=max_rows).to_numpy()
    y = pd.read_csv(labels, nrows=max_rows)["x"].to_numpy()
    if len(x) != len(y):
        raise C.PipelineError(f"corrected {spec.dataset}: len(x)={len(x)} != len(y)={len(y)} (x/y deben alinearse)")
    if x.ndim != 2 or x.shape[1] != 115:
        raise C.PipelineError(f"corrected {spec.dataset}: x.shape={x.shape}, se esperaban 115 columnas")
    vals = np.unique(y)
    if not set(vals.tolist()) <= {0, 1}:
        raise C.PipelineError(f"corrected {spec.dataset}: etiquetas fuera de {{0, 1}}: {vals[:10].tolist()}")
    if len(y) < train_rows:
        raise C.PipelineError(f"corrected {spec.dataset}: el stream tiene {len(y)} filas < tramo de entrenamiento "
                              f"{train_rows}")
    tr = y[:train_rows]
    n_attack_train = int(tr.sum())
    if n_attack_train:
        first = int(np.flatnonzero(tr)[0])
        raise C.PipelineError(
            f"el tramo de entrenamiento [0, {train_rows}) de {spec.dataset} contiene {n_attack_train} etiquetas de ataque "
            f"(primera en la fila {first}): D-7 exige un tramo benigno contiguo, anterior y excluido de la evaluacion; "
            f"no se lanza")
    att = np.flatnonzero(y)
    extra = {"m": int(len(y)), "n_attack": int(len(att)), "first_attack_index": int(att[0]) if len(att) else None,
             "train_benign_check": {"rows": [0, int(train_rows)], "n_attack": 0, "passed": True,
                                    "note": "las etiquetas solo validan un tramo fijado a priori; no seleccionan filas"},
             "regions": {"train": [0, int(train_rows)], "post_train": [int(train_rows), int(len(y))],
                         "note": "post_train NO es una region de evaluacion: los pares (calib, eval) los emite "
                                 "run_job.regions_for en config.json"}}
    return x, y, extra


def build_orig_cic_aci(code_root, data_root, spec):
    import experiments.frames.launch_all as la
    paths = orig_script_paths(code_root, spec.kind, data_root)
    if spec.kind == "aci":
        (x, y), (_enc, cat) = la.data_loader(paths[0])
    else:
        x = y = None
        cat = []
        for path in paths:
            if spec.kind == "cic2018":
                (_x, _y), (_enc, cat) = la.data_loader(path, skip=1_000_000)
            else:
                (_x, _y), (_enc, cat) = la.data_loader(path)
            if x is None:
                x, y = _x, _y
            else:
                x = np.concatenate((x, _x), axis=0)
                y = np.concatenate((y, _y), axis=0)
    x, y = la.grant_first_label(x, y, 300_000)
    x, y = la.regularize_data(x, y, cat)
    if len(x) != len(y):
        y = y[:len(x)]
    return x, y, {"paths": paths}


T0_SCRIPT = "revision/analysis/warmup_audit.py"
T0_CUTS = "revision/analysis/t0/stream_cuts.json"
CORRECTED_DIRS = {"cic2017": "CIC2017", "cic2018": "CIC2018", "aci": "ACI-IOT-2023"}
ZSCORE_EPS = 1e-6
HASH_CHUNK = 1_000_000
NA_TOKEN = "<NA>"
CALIB_FRACS = (5, 10, 20)
_T0_MODULES = {}


def t0_module(code_root):
    """warmup_audit.py (T0) de code_root como modulo PRIVADO (no se registra en sys.modules). En esta copia cached_npz
    se sustituye por el calculo directo: el constructor no lee ni escribe data/cache/t0 (hermetico; la huella del
    constructor incluye el sha256 del script)."""
    path = Path(code_root) / T0_SCRIPT
    sha = C.sha256_file(path)
    mod = _T0_MODULES.get((str(path), sha))
    if mod is None:
        mspec = importlib.util.spec_from_file_location(f"_jisa_t0_warmup_audit_{sha[:12]}", path)
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
        mod.cached_npz = lambda _path, _deps, compute: compute()
        _T0_MODULES[(str(path), sha)] = mod
    return mod


def corrected_variant_spec(spec, code_root):
    wa = t0_module(code_root)
    variant = C.CORRECTED_VARIANT[spec.dataset]
    return wa, variant, wa.corrected_variants(spec.kind)[variant]


def corrected_paths(spec, data_root, code_root):
    _wa, _variant, vspec = corrected_variant_spec(spec, code_root)
    return [Path(data_root) / CORRECTED_DIRS[spec.kind] / f for f in vspec["files"]]


def corrected_plan(spec, data_root, code_root, train_rows, max_rows=None, labels_reader=None):
    """Stage 1 of the corrected CIC and ACI streams, from labels and timestamps only: files, order, training cut [0, T) and guards."""
    wa, variant, vspec = corrected_variant_spec(spec, code_root)
    ds = spec.kind
    paths = corrected_paths(spec, data_root, code_root)
    read = labels_reader or (lambda p: wa.labels_csv(ds, p, max_rows or 0, "."))
    per = [read(p) for p in paths]
    S = wa.Source()
    S.ds, S.files, S.benign = ds, list(vspec["files"]), wa.BENIGN[ds]
    S.n_phys = [int(p[1]["n_rows"]) for p in per]
    S.offsets = np.concatenate([[0], np.cumsum(S.n_phys)]).astype(np.int64)
    S.vocab = sorted(set(v for p in per for v in p[1]["vocab"]))
    lut = [np.array([S.vocab.index(v) for v in p[1]["vocab"]], dtype=np.int16) for p in per]
    S.lcode = np.concatenate([lut[i][p[0]["codes"]] for i, p in enumerate(per)]).astype(np.int16)
    S.ybin = ((S.lcode != S.vocab.index(S.benign)).astype(np.int8) if S.benign in S.vocab
              else np.ones(len(S.lcode), np.int8))
    if "Label" in S.vocab:
        S.ybin[S.lcode == S.vocab.index("Label")] = 0
    S.ts = np.concatenate([p[0]["ts"] for p in per]) if ds != "cic2017" else None
    S.ts_raw = np.concatenate([p[0]["ts_raw"] for p in per]) if ds == "cic2018" else None
    S.excl = (np.concatenate([p[0]["excl"] for p in per]) if ds != "cic2017"
              else np.zeros(len(S.lcode), np.int8))
    S.file_info = [p[1] for p in per]
    consts = wa.Consts(1)
    if vspec["train_rule"] != "days":
        consts.TRAIN = int(train_rows)
    try:
        st, _sorted = wa.build_corrected(S, variant, vspec, consts)
    except AssertionError as e:
        raise C.PipelineError(f"corrected {spec.dataset} ({variant}): aserto cronologico de T0: {e}.")
    except ValueError as e:
        raise C.PipelineError(f"corrected {spec.dataset} ({variant}): {e}")
    cuts = st.meta["cuts"]
    T, n = int(st.train_end), int(st.N)
    y = st.ybin.astype(np.int64)
    n_att_train = int(y[:T].sum())
    discarded = int(cuts["discarded_label_filter"]["n"])
    problems = []
    if T != int(train_rows):
        problems.append(f"tiene {T} filas y se esperaban {int(train_rows)} (common.CORRECTED_TRAIN_ROWS; DEV-006)")
    if n_att_train:
        problems.append(f"contiene {n_att_train} etiquetas de ataque (primera en la fila {int(np.flatnonzero(y[:T])[0])})")
    if discarded:
        problems.append(f"el filtro por etiqueta descartaria {discarded} filas ("
                        f"{cuts['discarded_label_filter']['types']}); se declaran 0")
    if problems:
        raise C.PipelineError(
            f"el tramo de entrenamiento [0, {T}) de {spec.dataset} ({variant}) no cumple la guarda benigna: " +
            "; ".join(problems) + ". D-7 exige un tramo benigno contiguo, anterior y excluido de la evaluacion; no se lanza")
    fidx = (np.searchsorted(S.offsets, st.gid, side="right") - 1).astype(np.int64)
    att = np.flatnonzero(y)
    return {"wa": wa, "variant": variant, "vspec": vspec, "paths": paths, "S": S, "gid": st.gid,
            "src_file": fidx.astype(np.int8), "src_row": (st.gid - S.offsets[fidx]).astype(np.int32),
            "T": T, "n": n, "y": y, "label_code": st.lcode.astype(np.int16), "vocab": list(S.vocab),
            "ts": None if S.ts is None else S.ts[st.gid].astype(np.int64), "cuts": cuts,
            "excluded_before_sort": st.meta["excluded_before_sort"], "flags": st.meta["flags"],
            "n_attack": int(len(att)), "first_attack_index": int(att[0]) if len(att) else None,
            "n_attack_train": n_att_train, "discarded_label_filter": discarded}


def corrected_raw_features(spec, plan):
    """Stage 2a: raw features (float64, NaN and inf untouched) in stream order, read by physical row and checked against the labels."""
    wa, S, paths = plan["wa"], plan["S"], plan["paths"]
    headers = [pd.read_csv(p, nrows=0).columns.tolist() for p in paths]
    if any(h != headers[0] for h in headers):
        raise C.PipelineError(f"corrected {spec.dataset}: cabeceras distintas entre ficheros "
                              f"{[p.name for p, h in zip(paths, headers) if h != headers[0]]}")
    label_col = wa.LABEL_COL[spec.kind]
    drop = list(wa.DROP_CORR[spec.kind]) + list(C.CORRECTED_EXTRA_DROP.get(spec.dataset, []))
    feats = [c for c in headers[0] if c not in drop and c != label_col]
    if label_col not in headers[0]:
        raise C.PipelineError(f"corrected {spec.dataset}: falta la columna de etiqueta {label_col!r} en la cabecera")
    read_cols = [c for c in headers[0] if c in set(feats) or c == label_col]
    vocab_arr = np.array(plan["vocab"], dtype=object)
    label_mismatch = {}
    bad_cols = [c for c in feats if c.strip() in C.CORRECTED_FORBIDDEN_COLUMNS or c.strip() in drop]
    if bad_cols:
        raise C.PipelineError(f"corrected {spec.dataset}: columnas identificadoras en las features {bad_cols}.")
    cats = list(C.CORRECTED_CATEGORICAL.get(spec.dataset, []))
    if [c for c in cats if c not in feats]:
        raise C.PipelineError(f"corrected {spec.dataset}: faltan las categoricas {cats} en la cabecera")
    n = plan["n"]
    pos = np.full(int(S.offsets[-1]), -1, np.int64)
    pos[plan["gid"]] = np.arange(n, dtype=np.int64)
    xr = np.empty((n, len(feats)), np.float64)
    cat_text = {c: np.empty(n, dtype=object) for c in cats}
    filled = np.zeros(n, bool)
    coerced, header_rows = {}, {}
    for fi, p in enumerate(paths):
        a, b = int(S.offsets[fi]), int(S.offsets[fi + 1])
        hdr = np.flatnonzero(S.excl[a:b] == 1)
        keep = np.delete(np.arange(b - a, dtype=np.int64), hdr)
        if len(hdr):
            header_rows[p.name] = hdr.tolist()
        df = pd.read_csv(p, usecols=read_cols, nrows=len(keep), dtype={label_col: str},
                         skiprows=[int(r) + 1 for r in hdr] if len(hdr) else None)
        if len(df) != len(keep) or list(df.columns) != read_cols:
            raise C.PipelineError(f"corrected {spec.dataset}: {p.name} dio {len(df)} filas/{len(df.columns)} columnas, "
                                  f"se esperaban {len(keep)}/{len(read_cols)} (etapa 1 de etiquetas)")
        tgt = pos[a + keep]
        sel = tgt >= 0
        dst = tgt[sel]
        filled[dst] = True
        lab = df[label_col].fillna(NA_TOKEN).to_numpy(dtype=object)[sel]
        bad = np.flatnonzero(lab != vocab_arr[plan["label_code"][dst]])
        if len(bad):
            label_mismatch[p.name] = {"n": int(len(bad)), "first_stream_row": int(dst[bad[0]]),
                                      "csv": str(lab[bad[0]]), "stage1": str(vocab_arr[plan["label_code"][dst[bad[0]]]])}
        for j, c in enumerate(feats):
            col = df[c]
            if c in cats:
                v = col.to_numpy(dtype=object)[sel]
                cat_text[c][dst] = np.where(pd.isna(v), None, v.astype(str))
                continue
            if not pd.api.types.is_numeric_dtype(col):
                num = pd.to_numeric(col, errors="coerce")
                nbad = int((num.isna() & col.notna()).to_numpy()[sel].sum())
                if nbad:
                    coerced[c] = coerced.get(c, 0) + nbad
                col = num
            xr[dst, j] = col.to_numpy(dtype=np.float64)[sel]
        del df
    if label_mismatch:
        raise C.PipelineError(f"corrected {spec.dataset}: la etiqueta releida junto a las features no coincide con la "
                              f"de la etapa 1 (x/y desalineados): {label_mismatch}")
    if coerced:
        raise C.PipelineError(f"corrected {spec.dataset}: valores no numericos en columnas numericas {coerced}")
    if not filled.all():
        raise C.PipelineError(f"corrected {spec.dataset}: {int((~filled).sum())} filas del stream sin features")
    cat_vocab_all = {}
    for c in cats:
        codes, uniq = pd.factorize(pd.Series(cat_text[c], dtype=object), sort=True)
        xr[:, feats.index(c)] = codes.astype(np.float64)
        cat_vocab_all[c] = [str(u) for u in uniq]
    return xr, {"columns": feats, "label_column": label_col, "dropped_columns": drop, "categorical": cats,
                "categorical_values_in_stream": cat_vocab_all, "cat_text": cat_text,
                "repeated_header_rows_skipped": header_rows,
                "labels_rechecked_with_features": {"rows": int(n), "mismatches": 0,
                                                   "note": "etiqueta leida en la misma lectura que x, fila a fila (F11)"}}


def train_duplicate_mask(xr, train_end, hash_fn, chunk=HASH_CHUNK):
    """Boolean mask (N,) of the rows of [train_end, N) whose raw feature vector equals some row of [0, train_end)."""
    n, T = len(xr), int(train_end)
    mask = np.zeros(n, bool)
    info = {"method": "warmup_audit.hash_matrix por bloques + comparacion exacta de valores (-0 == +0, NaN == NaN)",
            "train_rows": [0, T], "hash_matches": 0, "hash_matches_rejected": 0}
    if T <= 0 or n <= T:
        info["n"] = 0
        return mask, info
    h = np.empty(n, np.uint64)
    for a in range(0, n, chunk):
        h[a:a + chunk] = hash_fn(xr[a:a + chunk])
    order = np.argsort(h[:T], kind="stable")
    sh = h[:T][order]
    eh = h[T:]
    k = np.searchsorted(sh, eh)
    kc = np.minimum(k, T - 1)
    hit = (k < T) & (sh[kc] == eh)
    ex = np.flatnonzero(hit) + T
    tr = order[kc[hit]]
    eq = np.zeros(len(ex), bool)
    step = 100_000
    for a in range(0, len(ex), step):
        A, B = xr[ex[a:a + step]], xr[tr[a:a + step]]
        eq[a:a + step] = ((A == B) | (np.isnan(A) & np.isnan(B))).all(axis=1)
    for i in np.flatnonzero(~eq):
        lo, hi = np.searchsorted(sh, h[ex[i]], "left"), np.searchsorted(sh, h[ex[i]], "right")
        B, A = xr[order[lo:hi]], xr[ex[i]]
        eq[i] = bool(((B == A) | (np.isnan(B) & np.isnan(A))).all(axis=1).any())
    mask[ex[eq]] = True
    info.update(hash_matches=int(len(ex)), hash_matches_rejected=int((~eq).sum()), n=int(mask.sum()))
    return mask, info


def corrected_preprocess(xr, raw, T):
    """Etapa 2b (C5): todo objeto ajustado usa SOLO las filas [0, T) del stream del modelo; la ejecucion no ajusta nada.
    Numericas, en este orden y por columna (in situ sobre xr si no hay categoricas):
      fillna(0) (constante, no ajustada; launch_all.py:98);
      +inf/-inf -> maximo finito de la columna en [0, T) (misma regla que launch_all.py:118-120, que usaba el fichero
      entero; 0.0 si la columna no tiene valores finitos en [0, T), declarado);
      z-score (x - media[0,T)) / (desv[0,T) + 1e-6) con np.mean/np.std (ddof 0) como launch_all.py:150-153.
    Categoricas: vocabulario ordenado de [0, T) (NaN = '<NA>'), codigo reservado len(vocab) para lo no visto, uint8
    (sin desbordamiento: len(vocab) + 1 <= 256) y np.unpackbits (launch_all.py:146-148). Salida: [numericas, bits]
    en float64, como regularize_data."""
    cols, cats = raw["columns"], raw["categorical"]
    num_idx = [j for j, c in enumerate(cols) if c not in cats]
    xn = xr if len(num_idx) == len(cols) else np.ascontiguousarray(xr[:, num_idx])
    if not xn.flags.c_contiguous:
        raise C.PipelineError("corrected_preprocess: matriz numerica no contigua en orden C")
    d = xn.shape[1]
    nan_train, nan_exec, inf_train, inf_exec = (np.zeros(d, np.int64) for _ in range(4))
    inf_fill = np.zeros(d, np.float64)
    no_finite = []
    for j in range(d):
        col = xn[:, j]
        m = np.isnan(col)
        if m.any():
            nan_train[j], nan_exec[j] = int(m[:T].sum()), int(m[T:].sum())
            col[m] = 0.0
        tr = col[:T]
        fin = np.isfinite(tr)
        if fin.any():
            inf_fill[j] = tr[fin].max()
        else:
            no_finite.append(cols[num_idx[j]])
        mi = np.isinf(col)
        if mi.any():
            inf_train[j], inf_exec[j] = int(mi[:T].sum()), int(mi[T:].sum())
            col[mi] = inf_fill[j]
    mu = xn[:T].mean(axis=0)
    sd = xn[:T].std(axis=0)
    names = [cols[j] for j in num_idx]
    const_exec = {names[j]: int((xn[T:, j] != mu[j]).sum()) for j in range(d) if sd[j] == 0.0}
    xn -= mu
    xn /= (sd + ZSCORE_EPS)
    prep = {"fit_rows": [0, int(T)], "fillna": 0.0,
            "numeric_columns": names,
            "nan_to_zero": {"train": int(nan_train.sum()), "exec": int(nan_exec.sum())},
            "inf_to_train_max": {"train": int(inf_train.sum()), "exec": int(inf_exec.sum()),
                                 "fill": {names[j]: float(inf_fill[j]) for j in range(d) if inf_train[j] or inf_exec[j]},
                                 "columns_without_finite_train_values_fill_0": no_finite},
            "zscore": {"eps": ZSCORE_EPS, "mean": [float(v) for v in mu], "std": [float(v) for v in sd],
                       "constant_in_train": [names[j] for j in range(d) if sd[j] == 0.0],
                       "constant_in_train_exec_rows_differing": const_exec,
                       "note_constant_in_train": "std = 0 en [0, T): las filas de ejecucion con otro valor quedan "
                                                 "multiplicadas por 1e6 (launch_all.py:153 con estadisticos de "
                                                 "entrenamiento, C5); se declara, no se corrige"},
            "categorical": {}}
    if not cats:
        return xn, prep, names
    codes_all = []
    bit_names = []
    for c in cats:
        text = np.where(pd.isna(raw["cat_text"][c]), NA_TOKEN, raw["cat_text"][c]).astype(str)
        vocab = sorted(set(text[:T].tolist()))
        if len(vocab) + 1 > 256:
            raise C.PipelineError(f"categorica {c}: {len(vocab)} valores en entrenamiento + reservado > 256 (uint8)")
        codes = pd.Categorical(text, categories=vocab).codes.astype(np.int64)
        unseen = codes < 0
        codes[unseen] = len(vocab)
        codes_all.append(codes.astype(np.uint8))
        bit_names += [f"{c}[bit{b}]" for b in range(8)]
        prep["categorical"][c] = {"train_vocab": vocab, "reserved_code_unseen": len(vocab),
                                  "unseen_train": int(unseen[:T].sum()), "unseen_exec": int(unseen[T:].sum()),
                                  "unseen_values_exec": sorted(set(text[unseen].tolist()))[:50],
                                  "encoding": "uint8 + np.unpackbits (big-endian), como launch_all.py:146-148"}
    bits = np.unpackbits(np.stack(codes_all, axis=1), axis=1)
    x = np.concatenate([xn, bits], axis=1)
    return x, prep, names + bit_names


def build_cic_aci_corrected(code_root, data_root, spec, train_rows, max_rows=None, labels_reader=None):
    """Build the corrected stream of CIC2017, CIC2018 or ACI; returns (x, y, extra) with the per-row auxiliary arrays in extra."""
    t0 = time.time()
    plan = corrected_plan(spec, data_root, code_root, train_rows, max_rows=max_rows, labels_reader=labels_reader)
    t_plan = time.time() - t0
    T, n, y = plan["T"], plan["n"], plan["y"]
    xr, raw = corrected_raw_features(spec, plan)
    nonfinite = (~np.isfinite(xr)).any(axis=1)
    dup, dup_info = train_duplicate_mask(xr, T, plan["wa"].hash_matrix)
    by_frac = {}
    for pct in CALIB_FRACS:
        c = int(math.floor(pct / 100.0 * (n - T)))
        seg = slice(T, T + c)
        by_frac[f"{pct:02d}"] = {"calib": [T, T + c], "n_dup": int(dup[seg].sum()),
                                 "n_dup_attack": int((dup[seg] & (y[seg] == 1)).sum()),
                                 "n_fit_rows_F_cal": int(c - dup[seg].sum())}
    dup_info.update(n_attack=int((dup & (y == 1)).sum()), by_calib_fraction=by_frac)
    production = (max_rows is None and C.realpath(data_root) == C.realpath(C.DATA_ROOT) and
                  int(train_rows) == C.corrected_train_rows(spec))
    ref = C.CORRECTED_DUP_REFERENCE.get(spec.dataset)
    if production and ref is not None:
        obtained = {"n": dup_info["n"], "n_attack": dup_info["n_attack"], "calib10": by_frac["10"]["n_dup"]}
        dup_info["reference"] = dict(ref, obtained=obtained, matches=all(obtained[k] == ref[k] for k in obtained),
                                     note="sin HALT: una discrepancia se declara (common.CORRECTED_DUP_REFERENCE)")
    x, prep, model_cols = corrected_preprocess(xr, raw, T)
    del xr
    x = np.ascontiguousarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise C.PipelineError(f"corrected {spec.dataset}: quedan valores no finitos tras el preprocesado")
    got = {"n_rows": n, "train_end": T, "first_attack_index": plan["first_attack_index"],
           "n_attack_train": plan["n_attack_train"], "discarded_label_filter": plan["discarded_label_filter"],
           "guard_ties_dropped": int(plan["cuts"]["guard_ties_dropped"]["n"]),
           "excluded_before_sort": int(sum(v["n"] for v in plan["excluded_before_sort"].values())),
           "n_attack": plan["n_attack"], "dim": int(x.shape[1])}
    expected = None
    if production:
        exp = C.CORRECTED_EXPECTED[spec.dataset]
        diffs = {k: {"esperado": exp[k], "obtenido": got.get(k)} for k in exp if got.get(k) != exp[k]}
        if diffs:
            raise C.PipelineError(f"corrected {spec.dataset}: el stream no coincide con la auditoria T0 "
                                  f"(common.CORRECTED_EXPECTED, t0/stream_cuts.json): {diffs}")
        expected = {"checked": True, "values": exp}
    t0_sha = C.sha256_file(Path(code_root) / T0_SCRIPT)
    try:
        cuts_sha = json.loads((Path(code_root) / T0_CUTS).read_text(encoding="utf-8"))["_meta"]["script_sha256"]
    except (OSError, ValueError, KeyError):
        cuts_sha = None
    aux = {"dup_train": dup, "src_file": plan["src_file"], "src_row": plan["src_row"], "label_code": plan["label_code"]}
    if plan["ts"] is not None:
        aux["ts"] = plan["ts"]
    extra = {
        "m": n, "n_rows": n, "n_attack": plan["n_attack"], "first_attack_index": plan["first_attack_index"],
        "variant": plan["variant"], "dataset": spec.dataset,
        "t0": {"script": T0_SCRIPT, "script_sha256": t0_sha, "stream_cuts": T0_CUTS,
               "stream_cuts_script_sha256": cuts_sha, "script_matches_stream_cuts": cuts_sha == t0_sha,
               "files": list(plan["vspec"]["files"]), "order": plan["vspec"]["order"],
               "train_rule": plan["vspec"]["train_rule"], "flags": plan["flags"], "cuts": plan["cuts"],
               "excluded_before_sort": plan["excluded_before_sort"],
               "n_rows_by_file": [int(v) for v in plan["S"].n_phys], "max_rows_per_file": max_rows},
        "train_benign_check": {"rows": [0, T], "n_attack": plan["n_attack_train"], "passed": True,
                               "discarded_label_filter": plan["discarded_label_filter"],
                               "note": "las etiquetas solo validan un tramo fijado por la regla T0 de la variante "
                                       "(no seleccionan filas: 0 descartadas); PipelineError si no"},
        "regions": {"train": [0, T], "post_train": [T, n],
                    "note": "post_train NO es una region de evaluacion: los pares (calib, eval) los emite "
                            "run_job.regions_for en config.json"},
        "labels": {"semantics": "y[i] = 1 si la etiqueta del flujo x[i] != benigno; alineadas fila a fila (sin desfase)",
                   "benign": plan["S"].benign, "vocab": plan["vocab"],
                   "attack_types_in_stream": plan["wa"]._types(plan["S"], plan["gid"])},
        "features": {"raw_columns": raw["columns"], "dropped_columns": raw["dropped_columns"],
                     "label_column": raw["label_column"], "categorical": raw["categorical"],
                     "categorical_values_in_stream": raw["categorical_values_in_stream"],
                     "repeated_header_rows_skipped": raw["repeated_header_rows_skipped"],
                     "labels_rechecked_with_features": raw["labels_rechecked_with_features"],
                     "model_columns": model_cols, "dim": int(x.shape[1])},
        "preprocessing": prep,
        "nonfinite_rows": {"train": int(nonfinite[:T].sum()), "exec": int(nonfinite[T:].sum())},
        "duplicates_train_exec": dup_info,
        "expected_production": expected,
        "aux_semantics": {"dup_train": "bool: fila de [T, N) identica en features crudas a una fila de [0, T)",
                          "src_file": "int8: indice en t0.files", "src_row": "int32: fila de datos del CSV (0-based)",
                          "label_code": "int16: indice en labels.vocab (tipo de ataque; R2.4)",
                          "ts": "int64: segundos desde epoch del timestamp corregido (R1.11; R-4)"},
        "timing": {"plan_s": t_plan, "total_s": time.time() - t0},
        "_aux_arrays": aux,
    }
    return x, y, extra


def default_train_rows(spec):
    return C.corrected_train_rows(spec) if spec.pipeline == "corrected" else TRAIN_ROWS


def builder_for(spec, max_rows):
    if spec.kind == "knad" and spec.pipeline == "orig":
        return build_knad_orig_replica if max_rows is not None else build_knad_orig_class
    if spec.kind == "knad" and spec.pipeline == "corrected":
        return build_knad_corrected
    if spec.pipeline == "orig":
        return build_orig_cic_aci
    C.check_supported(spec)
    if spec.pipeline == "corrected" and spec.kind in CORRECTED_DIRS:
        return build_cic_aci_corrected
    raise C.InvalidJob(f"sin constructor para {spec.dataset} {spec.pipeline}")


_CICACI_FUNCS = (t0_module, corrected_variant_spec, corrected_paths, corrected_plan, corrected_raw_features,
                 train_duplicate_mask, corrected_preprocess, build_cic_aci_corrected)
_CICACI_CONSTS = ("CORRECTED_VARIANT", "CORRECTED_TRAIN_ROWS", "CORRECTED_EXTRA_DROP", "CORRECTED_CATEGORICAL",
                  "CORRECTED_FORBIDDEN_COLUMNS", "CORRECTED_EXPECTED", "CORRECTED_DUP_REFERENCE")


def builder_fingerprint(builder, code_root):
    parts = [f"format={PIPELINE_FORMAT}", inspect.getsource(builder)]
    if builder in (build_knad_orig_class, build_knad_orig_replica):
        parts.append(C.sha256_file(Path(code_root) / "src/database/test_database/database.py"))
    if builder is build_orig_cic_aci:
        parts.append(C.sha256_file(Path(code_root) / "experiments/frames/launch_all.py"))
        parts.append(inspect.getsource(orig_script_paths))
    if builder is build_cic_aci_corrected:
        parts += [inspect.getsource(f) for f in _CICACI_FUNCS if f is not builder]
        parts.append(f"ZSCORE_EPS={ZSCORE_EPS!r} NA_TOKEN={NA_TOKEN} CORRECTED_DIRS={sorted(CORRECTED_DIRS.items())}")
        parts.append(json.dumps({k: getattr(C, k) for k in _CICACI_CONSTS}, sort_keys=True))
        parts.append(C.sha256_file(Path(code_root) / T0_SCRIPT))
    return C.sha256_text("\n".join(parts))


def source_files(spec, data_root, code_root):
    if spec.kind == "knad":
        return [str(p) for p in knad_files(data_root, spec)]
    if spec.pipeline == "corrected":
        return [str(p) for p in corrected_paths(spec, data_root, code_root)]
    return orig_script_paths(code_root, spec.kind, data_root)


def expected_rows(spec, data_root):
    if spec.kind == "knad":
        m = count_label_rows(knad_files(data_root, spec)[1])
        return m - 1 if spec.pipeline == "orig" else m
    if spec.pipeline == "corrected":
        if C.realpath(data_root) == C.realpath(C.DATA_ROOT):
            return C.CORRECTED_EXPECTED[spec.dataset]["n_rows"]
        return None
    return ORIG_EXPECTED_N.get(spec.kind)


def cache_key(spec):
    return f"{C.ds_slug(spec.dataset)}__{spec.pipeline}__{spec.pipeline_version}"


def load_stream(spec, *, cache_root, data_root, code_root, test_mode, max_rows=None, overrides=None, log=print,
                cache_wait_s=3600, hb=None):
    """Devuelve (x, y, meta_publica). Lanza EnvError, PipelineError, CacheMismatch, CacheIntegrityError."""
    C.check_supported(spec)
    overrides = dict(overrides or {})
    prod_cache = C.realpath(cache_root) == C.realpath(C.CACHE_ROOT)
    if test_mode and (C.is_under(cache_root, C.CACHE_ROOT) or C.is_under(C.CACHE_ROOT, cache_root)):
        raise C.EnvError(f"--test-mode no puede usar la cache de produccion {C.CACHE_ROOT} ni rutas que la contengan "
                         f"o esten dentro de ella: {cache_root} (M3)")
    if not test_mode:
        if C.realpath(data_root) != C.realpath(C.DATA_ROOT):
            raise C.EnvError(f"en produccion data_root debe ser {C.DATA_ROOT} (M3), no {data_root}")
        if not prod_cache:
            raise C.EnvError(f"en produccion cache_root debe ser {C.CACHE_ROOT} (M3), no {cache_root}")
        if max_rows is not None or overrides:
            raise C.EnvError("max_rows/overrides solo con --test-mode")
    C.check_not_forbidden(cache_root, "cache_root")
    train_rows = int(overrides.get("train_rows", default_train_rows(spec)))
    builder = builder_for(spec, max_rows)
    semantics = SEMANTICS[(spec.kind, spec.pipeline)]
    base = {"key": cache_key(spec), "pipeline": spec.pipeline, "pipeline_version": spec.pipeline_version,
            "semantics": semantics, "builder": builder.__name__, "test_mode": bool(test_mode),
            "data_root": C.realpath(data_root)}
    uncached_reason = None
    if max_rows is not None:
        uncached_reason = f"max_rows={max_rows}"
    elif "train_rows" in overrides:
        uncached_reason = f"override train_rows={train_rows}"
    if uncached_reason:
        t0 = time.time()
        x, y, extra = _call_builder(builder, spec, data_root, code_root, max_rows, train_rows)
        aux = extra.pop("_aux_arrays", None)
        out = dict(base, cached=False, uncached_reason=uncached_reason, build_seconds=time.time() - t0,
                   x={"shape": list(x.shape), "dtype": str(x.dtype), "fortran_order": bool(x.flags.f_contiguous and not x.flags.c_contiguous)},
                   y={"shape": list(y.shape), "dtype": str(y.dtype)}, extra=extra)
        if aux:
            import hashlib
            out["aux_inmemory"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                       "sha256_bytes": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
                                   for k, v in aux.items()}
        return x, y, out
    return _get_or_build(spec, builder, base, Path(cache_root), data_root, code_root, test_mode, train_rows, log,
                         cache_wait_s, hb)


def _call_builder(builder, spec, data_root, code_root, max_rows, train_rows):
    if builder is build_knad_orig_class:
        return builder(data_root, spec)
    if builder is build_knad_orig_replica:
        return builder(data_root, spec, max_rows)
    if builder is build_knad_corrected:
        return builder(data_root, spec, train_rows=train_rows, max_rows=max_rows)
    if builder is build_cic_aci_corrected:
        return builder(code_root, data_root, spec, train_rows=train_rows, max_rows=max_rows)
    return builder(code_root, data_root, spec)


def _get_or_build(spec, builder, base, cache_root, data_root, code_root, test_mode, train_rows, log, cache_wait_s, hb):
    key = base["key"]
    final = cache_root / key
    fp = builder_fingerprint(builder, code_root)
    if (final / "DONE").exists():
        return _verify_and_load(spec, final, fp, data_root, test_mode)
    locks = cache_root / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / key
    t_wait = time.time()
    host, pid = C.hostname(), os.getpid()
    while True:
        try:
            os.mkdir(lock)
            C.write_json_atomic(lock / "info.json", {"pid": pid, "host": host, "start_ticks": C.proc_start_ticks(pid),
                                                     "start": C.now_iso(), "key": key})
            break
        except FileExistsError:
            info = C.read_lock_info(lock) or {}
            if info.get("host") == host and info.get("pid") and not C.pid_alive(info["pid"], info.get("start_ticks")):
                stale = locks / f"{key}.stale-{C.ts_compact()}-{pid}"
                try:
                    os.rename(lock, stale)
                except OSError:
                    continue
                moved = C.read_lock_info(stale) or {}
                if (str(moved.get("pid")) != str(info.get("pid")) or
                        str(moved.get("start_ticks")) != str(info.get("start_ticks"))):
                    try:
                        os.rename(stale, lock)
                        log(f"cache {key}: el lock cambio entre la comprobacion y el movimiento (pid {moved.get('pid')} "
                            f"!= {info.get('pid')}): devuelto")
                    except OSError as e:
                        log(f"cache {key}: AVISO lock ajeno movido por carrera y no devuelto ({e}); queda en {stale.name}")
                else:
                    log(f"cache {key}: lock de constructor obsoleto (PID {info.get('pid')} muerto) movido a {stale.name}")
                continue
            if (final / "DONE").exists():
                return _verify_and_load(spec, final, fp, data_root, test_mode)
            if time.time() - t_wait > cache_wait_s:
                raise C.EnvError(f"cache {key}: otro proceso la construye desde hace > {cache_wait_s} s ({info})")
            if hb:
                hb.set_phase("cache_espera")
            time.sleep(min(10.0, max(0.5, cache_wait_s / 100)))
    try:
        if (final / "DONE").exists():
            return _verify_and_load(spec, final, fp, data_root, test_mode)
        tmp_root = cache_root / ".tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp = tmp_root / f"{key}.{host}.{pid}"
        if tmp.exists():
            _rmtree_rw(tmp)
        tmp.mkdir()
        if hb:
            hb.set_phase("cache_construccion")
        t0 = time.time()
        srcs = source_files(spec, data_root, code_root)
        x, y, extra = _call_builder(builder, spec, data_root, code_root, None, train_rows)
        build_s = time.time() - t0
        np.save(tmp / "x.npy", x)
        np.save(tmp / "y.npy", y)
        sx, sy = C.sha256_file(tmp / "x.npy"), C.sha256_file(tmp / "y.npy")
        aux_meta = {}
        for name, arr in (extra.pop("_aux_arrays", None) or {}).items():
            if len(arr) != len(x):
                raise C.CacheIntegrityError(f"cache {key}: aux {name} con {len(arr)} filas != {len(x)}")
            np.save(tmp / f"aux_{name}.npy", arr)
            aux_meta[name] = {"file": f"aux_{name}.npy", "shape": list(arr.shape), "dtype": str(arr.dtype),
                              "sha256": C.sha256_file(tmp / f"aux_{name}.npy")}
        sources = []
        for s in srcs:
            st = os.stat(s)
            sources.append({"path": C.realpath(s), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                            "sha256": C.sha256_file(s)})
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            rss = None
        meta = dict(base, format=PIPELINE_FORMAT, dataset=spec.dataset, kind=spec.kind, sources=sources,
                    builder={"function": builder.__name__, "fingerprint_sha256": fp,
                             "runner_commit": C.head_commit(C.RUNNER_DIR), "code_root": str(code_root)},
                    versions={"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
                    x={"shape": list(x.shape), "dtype": str(x.dtype), "fortran_order": bool(np.isfortran(x)),
                       "sha256": sx},
                    y={"shape": list(y.shape), "dtype": str(y.dtype), "sha256": sy},
                    expected_rows=expected_rows(spec, data_root), created=C.now_iso(), host=host,
                    build_seconds=build_s, rss_peak_bytes=rss, extra=extra)
        if aux_meta:
            meta["aux"] = aux_meta
        if meta["expected_rows"] is not None and x.shape[0] != meta["expected_rows"]:
            raise C.CacheIntegrityError(f"cache {key}: x tiene {x.shape[0]} filas, se esperaban {meta['expected_rows']}")
        del x, y
        C.write_json_atomic(tmp / "meta.json", meta)
        C.write_text_atomic(tmp / "DONE", C.now_iso() + "\n")
        for f in tmp.iterdir():
            os.chmod(f, 0o444)
        C.fsync_tree(tmp)
        try:
            os.rename(tmp, final)
        except OSError as e:
            if (final / "DONE").exists():
                log(f"cache {key}: {final} ya existia al renombrar ({e}); se carga la existente y se borra {tmp.name}")
                rmtree_rw(tmp)
                built_elsewhere = True
            else:
                raise
        else:
            os.chmod(final, 0o555)
            built_elsewhere = False
            log(f"cache {key}: construida en {build_s:.1f} s -> {final}")
    finally:
        try:
            info = C.read_lock_info(lock) or {}
            if str(info.get("pid")) == str(pid):
                (lock / "info.json").unlink(missing_ok=True)
                os.rmdir(lock)
        except OSError:
            pass
    out = _verify_and_load(spec, final, fp, data_root, test_mode, skip_hash=not built_elsewhere)
    out[2]["built_now"] = not built_elsewhere
    return out


def _verify_and_load(spec, final, fp, data_root, test_mode, skip_hash=False):
    """M3: la cache debe corresponder a esta invocacion. Errores -> CacheMismatch / CacheIntegrityError (HALT)."""
    t0 = time.time()
    meta = C.read_json(final / "meta.json")
    if not meta:
        raise C.CacheIntegrityError(f"cache {final}: meta.json ilegible")
    checks = {}
    key = final.name
    if meta.get("format") != PIPELINE_FORMAT:
        raise C.CacheMismatch(f"cache {key}: formato {meta.get('format')} != {PIPELINE_FORMAT}")
    if meta["builder"]["fingerprint_sha256"] != fp:
        raise C.CacheMismatch(f"cache {key}: el constructor ha cambiado (fingerprint distinto). Sube "
                              "pipeline_version o reconstruye con permiso; la cache antigua no se borra: renombrala "
                              f"a mano a {key}.superseded-<ts>")
    checks["builder_fingerprint"] = True
    if bool(meta.get("test_mode")) != bool(test_mode):
        raise C.CacheMismatch(f"cache {key}: construida con test_mode={meta.get('test_mode')} y se carga con "
                              f"test_mode={test_mode} (M3)")
    checks["test_mode"] = bool(test_mode)
    if C.realpath(meta.get("data_root", "")) != C.realpath(data_root):
        raise C.CacheMismatch(f"cache {key}: data_root {meta.get('data_root')} != {C.realpath(data_root)} (M3)")
    checks["data_root"] = C.realpath(data_root)
    for s in meta.get("sources", []):
        if not C.is_under(s["path"], data_root):
            raise C.CacheMismatch(f"cache {key}: fuente {s['path']} fuera de {data_root} (M3)")
        try:
            st = os.stat(s["path"])
        except OSError as e:
            raise C.CacheMismatch(f"cache {key}: fuente {s['path']} no accesible: {e}")
        if st.st_size != s["size"] or st.st_mtime_ns != s["mtime_ns"]:
            raise C.CacheMismatch(f"cache {key}: fuente {s['path']} cambio (tamano/mtime) desde la construccion (M3)")
    checks["sources_size_mtime"] = len(meta.get("sources", []))
    if not skip_hash:
        for name in ("x", "y"):
            h = C.sha256_file(final / f"{name}.npy")
            if h != meta[name]["sha256"]:
                raise C.CacheIntegrityError(f"cache {key}: sha256 de {name}.npy no coincide ({h[:12]} != "
                                            f"{meta[name]['sha256'][:12]})")
        checks["sha256_recomputed"] = True
    else:
        checks["sha256_recomputed"] = "calculado al construir (mismo fichero renombrado)"
    aux_pub = {}
    for name, a in (meta.get("aux") or {}).items():
        p = final / a["file"]
        if not p.is_file():
            raise C.CacheIntegrityError(f"cache {key}: falta {a['file']}")
        if not skip_hash and C.sha256_file(p) != a["sha256"]:
            raise C.CacheIntegrityError(f"cache {key}: sha256 de {a['file']} no coincide")
        aux_pub[name] = dict(a, path=str(p))
    if aux_pub:
        checks["aux_sha256_recomputed"] = sorted(aux_pub) if not skip_hash else "calculado al construir"
    t_hash = time.time() - t0
    x = np.load(final / "x.npy")
    y = np.load(final / "y.npy")
    exp = expected_rows(spec, data_root)
    if x.shape[0] != y.shape[0] or (exp is not None and x.shape[0] != exp):
        raise C.CacheIntegrityError(f"cache {key}: filas x={x.shape[0]}, y={y.shape[0]}, esperadas={exp} (M3)")
    if any(a["shape"][0] != x.shape[0] for a in aux_pub.values()):
        raise C.CacheIntegrityError(f"cache {key}: arrays aux con filas distintas de x ({x.shape[0]})")
    checks["rows_vs_labels"] = {"x": int(x.shape[0]), "expected": exp}
    public = {k: meta[k] for k in ("key", "pipeline", "pipeline_version", "semantics", "builder", "test_mode",
                                   "data_root", "sources", "x", "y", "created", "extra") if k in meta}
    public.update(cached=True, path=str(final), y_path=str(final / "y.npy"), y_sha256=meta["y"]["sha256"],
                  verified_on_load=checks, verify_seconds=t_hash, load_seconds=time.time() - t0 - t_hash)
    if aux_pub:
        public["aux"] = aux_pub
    return x, y, public


def _rmtree_rw(path):
    def onerror(func, p, _exc):
        try:
            os.chmod(p, 0o755 if os.path.isdir(p) else 0o644)
            parent = os.path.dirname(p)
            os.chmod(parent, 0o755)
            func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=onerror)


def rmtree_rw(path):
    """Borra un arbol con ficheros en solo lectura (solo temporales de tests/runner)."""
    for root, dirs, _files in os.walk(path):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o755)
            except OSError:
                pass
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    _rmtree_rw(path)


def cli_spec(dataset, pipeline="corrected"):
    """Spec minima (sin manifiesto) para la CLI: la carga del stream solo usa dataset y pipeline."""
    row = {"job_id": f"CLI__{C.ds_slug(dataset)}", "block": "CLI", "priority": "0", "dataset": dataset,
           "variant": "stat", "ar": "True", "paradigm": "AR", "pipeline": pipeline, "pipeline_version": "v1",
           "window": "800", "k_clusters": "4", "grouping": "dbscan", "partition_seed": "NA", "warmup_frac": "NA",
           "contamination": "0", "decontaminate": "0", "seed": "None", "rep": "1", "save_ckpt": "0",
           "timing_ref": "0", "notes": ""}
    return C.Spec(row, test_mode=True)


def build_cache_provenance_problems(code_root):
    """Guardas de procedencia de --build-cache (las de run_job en produccion, M4): code_root distinto del arbol
    principal, este pipelines.py dentro de code_root, worktree registrado, HEAD con tag runner-vN exacto y worktree
    limpio. Devuelve la lista de problemas (vacia = se puede construir)."""
    code_root = Path(code_root).resolve()
    probs = []
    if C.realpath(code_root) == C.realpath(C.MAIN_REPO):
        probs.append("code_root es el arbol principal (se construye desde el worktree ../jisa-runner en runner-vN)")
    if not C.is_under(C.RUNNER_DIR, code_root):
        probs.append(f"pipelines.py ({C.RUNNER_DIR}) no esta dentro de code_root {code_root}")
    if not C.is_registered_worktree(code_root):
        probs.append(f"{code_root} no es un worktree registrado de {C.MAIN_REPO}")
    if not C.runner_tag_at_head(code_root):
        probs.append(f"HEAD de {code_root} no tiene un tag runner-vN exacto")
    dirty = C.worktree_dirty(code_root)
    if dirty:
        probs.append(f"worktree sucio: {' '.join(dirty.split())[:200]}")
    return probs


def summary_view(extra):
    """Resumen de extra sin listas largas (media/desviacion por columna, columnas del modelo)."""
    e = json.loads(json.dumps({k: v for k, v in extra.items() if k != "_aux_arrays"}, default=C._json_default))
    z = e.get("preprocessing", {}).get("zscore", {})
    for k in ("mean", "std"):
        if k in z:
            z[k] = f"<{len(z[k])} valores>"
    f = e.get("features", {})
    for k in ("model_columns", "raw_columns"):
        if k in f:
            f[k] = f"<{len(f[k])} columnas>"
    if "numeric_columns" in e.get("preprocessing", {}):
        e["preprocessing"]["numeric_columns"] = f"<{len(e['preprocessing']['numeric_columns'])} columnas>"
    return e


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Streams del runner: semantica, resumen en memoria o cache de produccion.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--summary", metavar="DATASET", help="construye en memoria (sin cache ni escritura) el stream "
                   "corrected de CIC2017/CIC2018/ACI-IOT-2023 e imprime el resumen (solo datos y etiquetas)")
    g.add_argument("--build-cache", metavar="DATASET", help="construye la cache de PRODUCCION del stream corrected "
                   "(> 5 min en CIC2018: en tmux, desde el worktree del tag runner-vN y con --code-root de ese worktree)")
    ap.add_argument("--code-root", default=str(C.DEFAULT_CODE_ROOT))
    ap.add_argument("--data-root", default=str(C.DATA_ROOT))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--train-rows", type=int, default=None)
    ap.add_argument("--out-json", default=None, help="con --summary: escribe el resumen tambien en este fichero")
    a = ap.parse_args(argv)
    if not a.summary and not a.build_cache:
        print(json.dumps({"semantics": {f"{k[0]}:{k[1]}": v for k, v in SEMANTICS.items()}}, indent=2, ensure_ascii=False))
        return 0
    sys.path.insert(0, str(a.code_root))
    if a.summary:
        spec = cli_spec(a.summary)
        t0 = time.time()
        x, y, extra = build_cic_aci_corrected(a.code_root, a.data_root, spec,
                                              train_rows=a.train_rows or C.corrected_train_rows(spec),
                                              max_rows=a.max_rows)
        out = {"dataset": spec.dataset, "x_shape": list(x.shape), "x_dtype": str(x.dtype), "y_shape": list(y.shape),
               "aux": {k: [list(v.shape), str(v.dtype)] for k, v in extra["_aux_arrays"].items()},
               "seconds": time.time() - t0, "extra": summary_view(extra)}
        text = json.dumps(out, indent=1, ensure_ascii=False, default=C._json_default)
        print(text)
        if a.out_json:
            C.write_text_atomic(a.out_json, text + "\n")
        return 0
    spec = cli_spec(a.build_cache)
    if a.max_rows is not None or a.train_rows is not None or C.realpath(a.data_root) != C.realpath(C.DATA_ROOT):
        print("NO SE CONSTRUYE: --build-cache es de produccion (sin --max-rows/--train-rows y con data_root de "
              f"produccion {C.DATA_ROOT}; M3)")
        return 2
    problems = build_cache_provenance_problems(a.code_root)
    if problems:
        print("NO SE CONSTRUYE la cache de produccion: " + "; ".join(problems))
        return 8
    x, y, meta = load_stream(spec, cache_root=C.CACHE_ROOT, data_root=C.DATA_ROOT, code_root=a.code_root,
                             test_mode=False, log=print)
    print(json.dumps({"key": meta["key"], "path": meta.get("path"), "x": meta["x"], "y": meta["y"],
                      "aux": sorted(meta.get("aux", {})), "built_now": meta.get("built_now", False)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
