#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_manifest.py - deterministic generator and checker of manifest.csv, the list of planned jobs (standard library only)."""
import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = Path(__file__).resolve().parent / "manifest.csv"
CANDIDATE_OUT = Path(__file__).resolve().parent / "manifest_v2_candidate.csv"
CANDIDATE_OUT_V3 = Path(__file__).resolve().parent / "manifest_v3_candidate.csv"
TRAIN_ROWS = 200_000

KNAD_M = {"OS_Scan": 1_697_851, "SSL_Renegotiation": 2_207_571, "Fuzzing": 2_244_139, "Active_Wiretap": 2_278_689,
          "Video_Injection": 2_472_401, "SSDP_Flood": 4_077_266}
STREAM_N = {**{ds: C.CORRECTED_EXPECTED[ds]["n_rows"] for ds in C.OTHER_DATASETS},
            **{f"KNAD:{a}": m for a, m in KNAD_M.items()}}
ACTIVE = "v6"
ACTIVE_SECOND = "CIC2017"
V1_TAG = "runner-v1"
AW = "KNAD:Active_Wiretap"
ALL9_ORDER = [AW] + sorted([d for d in STREAM_N if d != AW], key=lambda d: STREAM_N[d])
T1_WITHIN_SEED = [("mlp", True), ("mlp", False), ("conv2d", True), ("conv2d", False), ("conv1d", True), ("conv1d", False),
                  ("mha", True), ("mha", False), ("lstm", True), ("lstm", False)]
T1_WITHIN_A = [("stat", True), ("original", True)] + T1_WITHIN_SEED


def row(block, priority, dataset, variant, ar, pipeline, seed, rep=1, window=800, save_ckpt=0, timing_ref=0, notes="",
        k_clusters=4, grouping="dbscan", partition_seed="NA", contamination=0):
    r = {"block": block, "priority": str(priority), "dataset": dataset, "variant": variant, "ar": str(bool(ar)),
         "paradigm": "AR" if ar else "TSR", "pipeline": pipeline, "pipeline_version": "v1", "window": str(window),
         "k_clusters": str(k_clusters), "grouping": grouping, "partition_seed": str(partition_seed), "warmup_frac": "NA",
         "contamination": str(contamination), "decontaminate": "0", "seed": str(seed), "rep": str(rep),
         "save_ckpt": str(save_ckpt), "timing_ref": str(timing_ref), "notes": notes}
    r["job_id"] = C.derive_job_id(r)
    return r


def rows_x2():
    """Rows of block X2 (original pipeline, seed None)."""
    ref = "ref data/original_results/kitsune/results/{a}/{v}_ar_True.npy"
    n = []
    n.append(row("X2", 100, "KNAD:OS_Scan", "stat", True, "orig", None,
                 notes="canario determinista; " + ref.format(a="OS_Scan", v="stat") + "; bit a bit NO garantizado "
                       "(procedencia del codigo): evidencia de reproduccion, no bloquea T1"))
    n.append(row("X2", 101, "KNAD:OS_Scan", "original", True, "orig", None,
                 notes="canario determinista; " + ref.format(a="OS_Scan", v="original") + "; salvedad BLAS"))
    n.append(row("X2", 600, "KNAD:SSDP_Flood", "stat", True, "orig", None,
                 notes="determinista a escala completa (4.077.265 filas, 10 bloques); " + ref.format(a="SSDP_Flood", v="stat")))
    n.append(row("X2", 610, "KNAD:OS_Scan", "mlp", True, "orig", None,
                 notes="no determinista; integridad (forma, dtype, NaN/Inf); " + ref.format(a="OS_Scan", v="mlp")))
    n.append(row("X2", 611, "KNAD:SSDP_Flood", "mlp", True, "orig", None,
                 notes="dataset KNAD grande; integridad; " + ref.format(a="SSDP_Flood", v="mlp")))
    n.append(row("X2", 620, "KNAD:OS_Scan", "conv1d", True, "orig", None, rep=1,
                 notes="variante media; " + ref.format(a="OS_Scan", v="conv1d")))
    n.append(row("X2", 621, "KNAD:OS_Scan", "conv1d", True, "orig", None, rep=2,
                 notes="repeticion con s0 para medir el no determinismo"))
    return n


def rows_s():
    aw = "KNAD:Active_Wiretap"
    n = []
    for i, v in enumerate(["lstm", "mha", "conv1d", "conv2d", "mlp"]):
        n.append(row("S-w1", 200 + i, aw, v, True, "corrected", "A", notes="fase=w1; sonda smoke.py (sin scores)"))
    n.append(row("S-w1", 205, "KNAD:SSDP_Flood", "lstm", True, "corrected", "A",
                 notes="fase=w1; carga y RSS con N maximo; sonda smoke.py"))
    k = 300
    for v in ("lstm", "mha"):
        for rep in (1, 2):
            n.append(row("S-w2", k, aw, v, True, "corrected", "A", rep=rep, notes="fase=w2; lanzar con 2 workers"))
            k += 1
    k = 400
    for v in ("lstm", "mha"):
        for rep in (1, 2, 3, 4):
            n.append(row("S-w4", k, aw, v, True, "corrected", "A", rep=rep, notes="fase=w4; lanzar con 4 workers"))
            k += 1
    for i, v in enumerate(["mha", "conv1d", "lstm"]):
        n.append(row("S-L2000", 500 + i, aw, v, True, "corrected", "A", window=2000,
                     notes="fase=L2000; OOM = resultado (probe.json); batch 1 y EXECUTION_BS=512"))
    return n


def rows_t1():
    order = ["Active_Wiretap"] + sorted([a for a in KNAD_M if a != "Active_Wiretap"], key=lambda a: KNAD_M[a])
    within = [("stat", True), ("original", True), ("mlp", True), ("mlp", False), ("conv2d", True), ("conv2d", False),
              ("conv1d", True), ("conv1d", False), ("mha", True), ("mha", False), ("lstm", True), ("lstm", False)]
    n = []
    for ai, att in enumerate(order):
        base = 1000 + 100 * ai
        for j, (v, ar) in enumerate(within):
            note = []
            if v in ("stat", "original"):
                note.append("determinista; AR = TSR por construccion (original_kitsune.py:16-17, stdev.py:14); "
                            "una ejecucion por ataque, no por semilla ni paradigma")
            if v == "mlp":
                note.append("MLPAutoencoder ignora ar (multilayer.py:16-17): T/F = replica gratuita con la misma semilla")
            if att == "Active_Wiretap":
                note.append("timing_ref: 1 worker")
            n.append(row("T1-A-KNAD", base + j, f"KNAD:{att}", v, ar, "corrected", "A",
                         save_ckpt=0 if v == "stat" else 1, timing_ref=1 if att == "Active_Wiretap" else 0,
                         notes="; ".join(note)))
    return n


def generate_v1():
    rows = rows_x2() + rows_s() + rows_t1()
    return sorted(rows, key=lambda r: (int(r["priority"]), r["job_id"]))


def generate():
    """Filas de manifest.csv segun ACTIVE (M2): v1 = generate_v1(); v2 = generate_v2(ACTIVE_SECOND)."""
    if ACTIVE == "v1":
        return generate_v1()
    if ACTIVE == "v2":
        return generate_v2(ACTIVE_SECOND)
    if ACTIVE == "v3":
        return generate_v3(ACTIVE_SECOND)
    if ACTIVE == "v4":
        return generate_v4(ACTIVE_SECOND)
    if ACTIVE == "v5":
        return generate_v5(ACTIVE_SECOND)
    if ACTIVE == "v6":
        return generate_v6(ACTIVE_SECOND)
    raise ValueError(f"ACTIVE={ACTIVE!r} no previsto (v1 | v2 | v3 | v4 | v5 | v6)")


MLP_NOTE = "MLPAutoencoder ignora ar (multilayer.py:16-17): una fila con ar=True"


def _corpus(dataset):
    return "KNAD" if dataset.startswith("KNAD:") else {"ACI-IOT-2023": "ACI"}.get(dataset, dataset)


def _ckpt(variant):
    return 0 if variant == "stat" else 1


def rows_t1_a_v2():
    """T1-A de ACI, CIC-IDS2017 y CSE-CIC-IDS2018 (12 filas cada uno, semilla A; §4.3, §13.2 P1)."""
    n = []
    for base, ds in ((1600, "ACI-IOT-2023"), (1700, "CIC2017"), (1800, "CIC2018")):
        for j, (v, ar) in enumerate(T1_WITHIN_A):
            note = ["§13.2 P1 (runner v2: pipeline corrected de CIC/ACI)"]
            if v in ("stat", "original"):
                note.append("determinista; una ejecucion por dataset (§4.3)")
            if v == "mlp":
                note.append("MLPAutoencoder ignora ar (multilayer.py:16-17): T/F = replica con la misma semilla")
            n.append(row(f"T1-A-{_corpus(ds)}", base + j, ds, v, ar, "corrected", "A", save_ckpt=_ckpt(v),
                         notes="; ".join(note)))
    return n


def rows_t1_seed(letter, base, plan_p):
    """T1-B / T1-C: 9 datasets x 10 filas (sin original ni stat, §4.3), AW primero y despues por tamano creciente."""
    n = []
    for i, ds in enumerate(ALL9_ORDER):
        for j, (v, ar) in enumerate(T1_WITHIN_SEED):
            note = [f"semilla {letter} (§4, §13.2 {plan_p})"]
            if v == "mlp":
                note.append("MLPAutoencoder ignora ar (multilayer.py:16-17): T/F = replica con la misma semilla")
            n.append(row(f"T1-{letter}-{_corpus(ds)}", base + 100 * i + j, ds, v, ar, "corrected", letter,
                         save_ckpt=_ckpt(v), notes="; ".join(note)))
    return n


def rows_t2(block, base, ar, seed, plan_p, second):
    """T2: {LSTM, MHA} x L en {200, 2000} x {AW, segundo dataset} (§7.2)."""
    n = []
    within = [("mha", 200), ("lstm", 200), ("mha", 2000), ("lstm", 2000)]
    for i, ds in enumerate([AW, second]):
        for j, (v, L) in enumerate(within):
            ref = C.ablation_reference_block(block, ds)
            n.append(row(block, base + 10 * i + j, ds, v, ar, "corrected", seed, window=L, save_ckpt=1,
                         notes=f"{block}: L={L} (referencia {ref} L=800); §7.2, §13.2 {plan_p}"))
    return n


def rows_t3_k(block, base, variants, datasets, plan_p):
    """T3 k: variantes x k en {1, 2, 8} con el DBSCAN iterativo del codigo y objetivo k (§7.3)."""
    n = []
    for i, ds in enumerate(datasets):
        j = 0
        for v, ar in variants:
            for k in C.T3_KS:
                note = f"{block}: k={k} (referencia T1-A k=4, DBSCAN); §7.3, §13.2 {plan_p}"
                if v == "mlp":
                    note += "; " + MLP_NOTE
                n.append(row(block, base + 10 * i + j, ds, v, ar, "corrected", "A", save_ckpt=_ckpt(v), k_clusters=k,
                             notes=note))
                j += 1
    return n


def rows_t3_random(block, base, variants, datasets, plan_p):
    """T3 agrupacion aleatoria: variantes x partition_seed en {101, 202, 303} con k = 4 y los tamanos DBSCAN (§7.3)."""
    n = []
    for i, ds in enumerate(datasets):
        j = 0
        for v, ar in variants:
            for ps in C.T3_PARTITION_SEEDS:
                note = (f"{block}: agrupacion aleatoria partition_seed={ps} con los tamanos del mapa DBSCAN (referencia "
                        f"T1-A); §7.3, §13.2 {plan_p}")
                if v == "mlp":
                    note += "; " + MLP_NOTE
                n.append(row(block, base + 10 * i + j, ds, v, ar, "corrected", "A", save_ckpt=_ckpt(v),
                             grouping="random", partition_seed=ps, notes=note))
                j += 1
    return n


def rows_t4e():
    """T4-E: AW x {original, MLP, LSTM-AR} x c en {5, 20} % de [fm, T), semilla A (§8.5; P3 condicionado)."""
    n = []
    j = 0
    for v in ("original", "mlp", "lstm"):
        for c in C.T4E_CONTAMINATION_PCT:
            note = (f"T4-E: c={c} % de las filas AE [fm, T) (referencia c=0 = T1-A semilla A sobre E sin el pool con "
                    f"guardas); §8.5, §13.2 P3 (condicionado al soporte del runner el 22 sep)")
            if v == "mlp":
                note += "; " + MLP_NOTE
            n.append(row("T4-E", 3400 + j, AW, v, True, "corrected", "A", save_ckpt=1, contamination=c, notes=note))
            j += 1
    return n


def generate_v2(second="CIC2017"):
    """Manifest of runner v2: the v1 rows unchanged plus the new rows; `second` is the second dataset of the ablations."""
    if second not in ("CIC2017", "KNAD:OS_Scan"):
        raise ValueError(f"segundo dataset de ablacion no previsto por el plan: {second}")
    abl = [AW, second]
    new = (rows_t1_a_v2() + rows_t1_seed("B", 2000, "P2") +
           rows_t2("T2-min", 3000, True, "A", "P3", second) +
           rows_t3_k("T3-min-a", 3100, [("stat", True), ("original", True), ("mlp", True)], ALL9_ORDER, "P3") +
           rows_t3_random("T3-min-b", 3200, [("stat", True), ("original", True), ("mlp", True)], abl, "P3") +
           rows_t3_k("T3-min-c", 3300, [("lstm", True)], abl, "P3") +
           rows_t4e() +
           rows_t1_seed("C", 4000, "P4") +
           rows_t2("T2-ext", 5000, False, "A", "P5", second) +
           rows_t3_random("T3-ext-a", 5100, [("lstm", True)], abl, "P5") +
           rows_t3_k("T3-ext-b", 5200, [("mha", True)], abl, "P5") +
           rows_t2("T2-B", 5300, True, "B", "P5", second))
    return sorted(generate_v1() + new, key=lambda r: (int(r["priority"]), r["job_id"]))


CTX_NOTE = ("[EXPLORATORIO]: variante nueva con ventana de contexto (revision/experiments/networks_ctx.py, "
            "registrada en caliente en NETWORK_TYPES; src/ intacto). Fuera del plan congelado: no entra en H1-H4, ni "
            "en S*_d, ni en R1.4, ni en el abstract. Referencia descriptiva: la misma celda de T1-A")


def rows_t7_mlpw(base=3500):
    """Rows of block T7-MLPW ('mlp_ctx'): 9 datasets x {AR, TSR}, seed A, corrected pipeline, L = 800, k = 4 (18 rows)."""
    n = []
    for i, ds in enumerate(ALL9_ORDER):
        for j, ar in enumerate((True, False)):
            n.append(row("T7-MLPW", base + 2 * i + j, ds, "mlp_ctx", ar, "corrected", "A", save_ckpt=1,
                         notes=f"{CTX_NOTE} ({'mlp-AR' if ar else 'mlp-TSR'}); el MLP publicado cambiando SOLO la "
                               f"entrada a la ventana aplanada (multilayer.py con Flatten; error MSE como el original)"))
    return n


def rows_t7_mlpdeep(base=3600):
    """Rows of block T7-MLPDEEP ('mlp_deep_ctx'): Active Wiretap and CIC2017 x {AR, TSR}, seed A, no checkpoint (4 rows)."""
    n = []
    for j, ar in enumerate((True, False)):
        for i, ds in enumerate(("KNAD:Active_Wiretap", "CIC2017")):
            n.append(row("T7-MLPDEEP", base + 2 * j + i, ds, "mlp_deep_ctx", ar, "corrected", "A", save_ckpt=0,
                         notes=f"{CTX_NOTE}; Deep MLP de la Tabla 3 con ventana "
                               "(archive/kitsune-zd-rewrite-2026/src/tf/autoencoders/deep_mlp_ae.py); sin "
                               "checkpoint por tamano"))
    return n


T8_NOTE = ("[EXPLORATORIO]: par RE-ENTRENADO. Dos factores declarados y aplicados por igual a las dos variantes: "
           "regla Pi de lr por capa (lr_capa = lr0 * n_v / fan_in_capa, elegida por la aritmetica de Adam y no "
           "ajustada con metricas) y LayerNormalization alrededor del cuello de botella. Validacion reservada "
           "(ultimo 10 % del tramo de AE) y puertas CG-0..CG-4. Fuera del plan congelado: no entra en H1-H4 ni en el "
           "abstract. El par (mlp_fix, mlp_ctx_fix) aisla la ventana")


def rows_t8(base=3700):
    """Rows of block T8: the pair (mlp_fix, mlp_ctx_fix) on the 9 datasets, 2 seeds and {TSR, AR} (72 rows)."""
    n, p = [], base
    for ar, seed in ((False, "A"), (False, "B"), (True, "A"), (True, "B")):
        for ds in sorted(ALL9_ORDER):
            for variant, block in (("mlp_fix", "T8-BASE"), ("mlp_ctx_fix", "T8-CTX")):
                n.append(row(block, p, ds, variant, ar, "corrected", seed, save_ckpt=1,
                             notes=f"{T8_NOTE} ({'AR' if ar else 'TSR'}, semilla {seed})"))
                p += 1
    return n


EPS_NOTE = ("[EXPLORATORIO] S-EPS: sensibilidad del estabilizador del dA de referencia (C10). eps 1e-16 en "
            "entrenamiento Y ejecucion, como el dA de Mirsky (external/Kitsune-py/KitNET/dA.py:87,119) y como las "
            "otras seis redes; el envio usa 1e-6 al entrenar (original_kitsune.py:93) y 1e-16 al ejecutar (:130). "
            "Unico factor cambiado frente a C10. FUERA de H1-H4 y de S*_d")


def rows_t8_eps(base=3800):
    """Rows of block T8-EPS: the stabiliser sensitivity of the reference autoencoder on the 9 datasets (9 rows)."""
    n, p = [], base
    for ds in sorted(ALL9_ORDER):
        n.append(row("T8-EPS", p, ds, "original_eps", True, "corrected", "A", save_ckpt=0, timing_ref=0,
                     notes=EPS_NOTE))
        p += 1
    return n


EP5_NOTE = ("[EXPLORATORIO]: ablacion de presupuesto sobre el par ya inicializado por contencion. CINCO epocas en "
            "vez de una, en las DOS variantes para que el presupuesto sea igual en los dos miembros y el contraste "
            "siga aislando la ventana. Unico factor cambiado frente a T8-BASE/T8-CTX. Mide si el empate del par "
            "re-entrenado (mediana del ratio 1,000 sobre 90 AE) es falta de presupuesto y no de capacidad")


def rows_t8_ep5(base=3900):
    """Rows of blocks T8-EP5 and T8-EP5B: the pair with five epochs on ACI, CIC-IDS2017 and CSE-CIC-IDS2018, TSR, seed A (6 rows)."""
    n, p = [], base
    for ds in ("ACI-IOT-2023", "CIC2017", "CIC2018"):
        for variant, block in (("mlp_fix_e5", "T8-EP5B"), ("mlp_ctx_fix_e5", "T8-EP5")):
            n.append(row(block, p, ds, variant, False, "corrected", "A", save_ckpt=1, notes=EP5_NOTE))
            p += 1
    return n


def rows_t8_ep5_knad(base=5400):
    """Rows that extend the five-epoch ablation to the six KNAD packet captures, TSR, seed A (12 rows)."""
    n, p = [], base
    for ds in ("KNAD:Active_Wiretap", "KNAD:Fuzzing", "KNAD:OS_Scan", "KNAD:SSDP_Flood",
               "KNAD:SSL_Renegotiation", "KNAD:Video_Injection"):
        for variant, block in (("mlp_fix_e5", "T8-EP5B"), ("mlp_ctx_fix_e5", "T8-EP5")):
            n.append(row(block, p, ds, variant, False, "corrected", "A", save_ckpt=1, notes=EP5_NOTE))
            p += 1
    return n


def generate_v6(second="CIC2017"):
    """Manifest of runner v6: generate_v5(second) unchanged plus the five-epoch ablation."""
    return sorted(generate_v5(second) + rows_t8_ep5(), key=lambda r: (int(r["priority"]), r["job_id"]))


def generate_v5(second="CIC2017"):
    """Manifest of runner v5: generate_v4(second) unchanged plus block T8-EPS."""
    return sorted(generate_v4(second) + rows_t8_eps(), key=lambda r: (int(r["priority"]), r["job_id"]))


def generate_v4(second="CIC2017"):
    """Manifest of runner v4: generate_v3(second) unchanged plus the T8 blocks."""
    return sorted(generate_v3(second) + rows_t8(), key=lambda r: (int(r["priority"]), r["job_id"]))


def generate_v3(second="CIC2017"):
    """Manifest of runner v3: generate_v2(second) unchanged plus the T7 blocks."""
    new = rows_t7_mlpw() + rows_t7_mlpdeep()
    return sorted(generate_v2(second) + new, key=lambda r: (int(r["priority"]), r["job_id"]))


def check_rows_preserved(reference_text, candidate_rows, candidate_text, label):
    """Toda fila de `reference_text` aparece identica (campos, job_id, linea CSV y orden relativo) en el candidato."""
    errs = []
    old = list(csv.DictReader(io.StringIO(reference_text)))
    cand = {r["job_id"]: r for r in candidate_rows}
    for r in old:
        c = cand.get(r["job_id"])
        if c is None:
            errs.append(f"fila {label} ausente en el candidato: {r['job_id']}")
        elif {k: r[k] for k in C.MANIFEST_COLUMNS} != {k: c[k] for k in C.MANIFEST_COLUMNS}:
            errs.append(f"fila {label} cambiada: {r['job_id']}")
    old_ids = [r["job_id"] for r in old]
    order = [r["job_id"] for r in candidate_rows if r["job_id"] in set(old_ids)]
    if order != old_ids:
        errs.append(f"el orden relativo de las filas {label} cambia")
    cand_lines = set(candidate_text.splitlines())
    missing = [ln for ln in reference_text.splitlines() if ln not in cand_lines]
    if missing:
        errs.append(f"{len(missing)} lineas {label} no aparecen literales en el candidato: {missing[:2]}")
    return errs


def v1_reference_text(code_root=None):
    """(texto, origen) de las filas v1: manifest.csv del tag runner-v1 (git show) o, sin tag legible, generate_v1()."""
    root = code_root or C.MAIN_REPO
    rc, out, _err = C.git(root, "show", f"{V1_TAG}:revision/experiments/manifest.csv")
    if rc == 0 and out.strip():
        return out.rstrip("\n") + "\n", f"git show {V1_TAG}:revision/experiments/manifest.csv"
    return to_csv(generate_v1()), "generate_v1() (sin tag runner-v1 legible)"


def check_v1_preserved(v1_csv, candidate_rows, candidate_text):
    """Toda fila v1 aparece identica en el candidato: mismos campos, mismo job_id, misma linea CSV y mismo orden
    relativo. v1_csv = ruta a un CSV v1 o None (= manifest.csv del tag runner-v1, v1_reference_text; M2: tras la
    promocion manifest.csv ya no es v1). Devuelve lista de errores."""
    errs = []
    if v1_csv is None:
        v1_text, _src = v1_reference_text()
    elif Path(v1_csv).exists():
        v1_text = Path(v1_csv).read_text()
    else:
        return [f"{v1_csv} no existe"]
    old = list(csv.DictReader(io.StringIO(v1_text)))
    cand = {r["job_id"]: r for r in candidate_rows}
    for r in old:
        c = cand.get(r["job_id"])
        if c is None:
            errs.append(f"fila v1 ausente en el candidato: {r['job_id']}")
        elif {k: r[k] for k in C.MANIFEST_COLUMNS} != {k: c[k] for k in C.MANIFEST_COLUMNS}:
            errs.append(f"fila v1 cambiada: {r['job_id']}")
    old_ids = [r["job_id"] for r in old]
    order = [r["job_id"] for r in candidate_rows if r["job_id"] in set(old_ids)]
    if order != old_ids:
        errs.append("el orden relativo de las filas v1 cambia")
    cand_lines = set(candidate_text.splitlines())
    v1_lines = v1_text.splitlines()
    missing = [ln for ln in v1_lines if ln not in cand_lines]
    if missing:
        errs.append(f"{len(missing)} lineas v1 no aparecen literales en el candidato: {missing[:2]}")
    return errs


def to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=C.MANIFEST_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: ("None" if r[c] == "None" else r[c]) for c in C.MANIFEST_COLUMNS})
    return buf.getvalue()


def benign_guard(data_root, attacks):
    """S14: primeras 200.000 etiquetas == 0 en cada ataque con filas corrected. Solo lee *_labels.csv."""
    res = {}
    for att in sorted(attacks):
        p = Path(data_root) / "KITSUNE" / C.KNAD_ATTACKS[att] / f"{att}_labels.csv"
        n_att, first, n = 0, None, 0
        with open(p, newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd)
            if header[-1] != "x":
                raise SystemExit(f"{p}: cabecera inesperada {header}")
            for i, rec in enumerate(rd):
                if i >= TRAIN_ROWS:
                    break
                n += 1
                if int(rec[-1]) != 0:
                    n_att += 1
                    first = i if first is None else first
        res[att] = {"rows_checked": n, "n_attack": n_att, "first_attack": first}
    return res


T0_DIR = Path(__file__).resolve().parents[1] / "analysis" / "t0"
T0_KEYS = {ds: C.label_counts_key(ds) for ds in C.OTHER_DATASETS}


def benign_guard_t0(datasets, cuts_path=T0_DIR / "stream_cuts.json", decisions_path=T0_DIR / "stream_cuts_decisions.json",
                    t0_script=T0_DIR.parent / "warmup_audit.py"):
    """Benign-training guard of the corrected CIC and ACI streams, checked against stream_cuts.json without reading any CSV."""
    import json
    cuts = json.loads(Path(cuts_path).read_text(encoding="utf-8"))
    dec = json.loads(Path(decisions_path).read_text(encoding="utf-8"))
    meta = cuts.get("_meta", {})
    t0_sha = C.sha256_file(t0_script) if Path(t0_script).is_file() else None
    res = {}
    for ds in sorted(datasets):
        key, variant = T0_KEYS.get(ds), C.CORRECTED_VARIANT.get(ds)
        probs = []
        if key is None or variant is None:
            res[ds] = {"variant": variant, "rows_checked": 0, "n_attack": None,
                       "problems": [f"sin variante T0 para {ds}"]}
            continue
        if meta.get("mode") != ["full"] or meta.get("constants", {}).get("scale_div") != 1:
            probs.append(f"stream_cuts.json no es de la pasada completa (mode={meta.get('mode')})")
        if meta.get("script_sha256") != t0_sha:
            probs.append(f"stream_cuts.json no lo genero el warmup_audit.py actual ({t0_script}: sha256 "
                         f"{str(t0_sha)[:12]} != {str(meta.get('script_sha256'))[:12]})")
        chosen = str(dec.get("datasets", {}).get(key, {}).get("variante", ""))
        default = cuts.get(key, {}).get("variante_por_defecto_D7")
        if not (chosen.split(" ")[0] == variant or (chosen.startswith("default") and default == variant)):
            probs.append(f"variante {variant!r} no es la elegida en stream_cuts_decisions.json ({chosen!r})")
        v = cuts.get(key, {}).get("variantes", {}).get(variant)
        if v is None:
            res[ds] = {"variant": variant, "rows_checked": 0, "n_attack": None,
                       "problems": probs + [f"stream_cuts.json sin la variante {variant}"]}
            continue
        tr, ev = v.get("entrenamiento", {}), v.get("evaluacion", {})
        exp = C.CORRECTED_EXPECTED[ds]
        checks = [("entrenamiento.ataques_en_entrenamiento", tr.get("ataques_en_entrenamiento"), 0),
                  ("entrenamiento.filas_mantenidas", tr.get("filas_mantenidas"), C.CORRECTED_TRAIN_ROWS[ds]),
                  ("descartadas_por_filtro_de_etiqueta.n", v.get("descartadas_por_filtro_de_etiqueta", {}).get("n"), 0),
                  ("evaluacion.filas_de_entrenamiento_reevaluadas", ev.get("filas_de_entrenamiento_reevaluadas"), 0),
                  ("evaluacion.inicio_modelo", ev.get("inicio_modelo"), exp["train_end"]),
                  ("evaluacion.fin_modelo", ev.get("fin_modelo"), exp["n_rows"]),
                  ("entrenamiento.fm_modelo", tr.get("fm_modelo"), [0, 50_000]),
                  ("entrenamiento.ae_modelo", tr.get("ae_modelo"), [50_000, C.CORRECTED_TRAIN_ROWS[ds]])]
        try:
            n_att_eval = round(float(ev["n_filas"]) * float(ev["pct_malicioso"]) / 100.0)
        except (KeyError, TypeError, ValueError):
            n_att_eval = None
        checks.append(("evaluacion.n_filas * pct_malicioso", n_att_eval, exp["n_attack"]))
        if v.get("flags", {}).get("chronological"):
            checks.append(("evaluacion.filas_ejec_con_ts_<=_max_entrenamiento",
                           ev.get("filas_ejec_con_ts_<=_max_entrenamiento"), 0))
        probs += [f"{name}={got!r} (se exige {want!r})" for name, got, want in checks if got != want]
        res[ds] = {"variant": variant, "source": "revision/analysis/t0/stream_cuts.json",
                   "rows_checked": tr.get("filas_mantenidas"), "n_attack": tr.get("ataques_en_entrenamiento"),
                   "discarded_label_filter": v.get("descartadas_por_filtro_de_etiqueta", {}).get("n"), "problems": probs}
    return res


def validate(rows, data_root=C.DATA_ROOT, check_labels=True):
    errs = []
    ids = [r["job_id"] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("job_id duplicados")
    miss = [c for c in C.GATES_MANIFEST_COLS if c not in C.MANIFEST_COLUMNS]
    if miss:
        errs.append(f"faltan columnas de gates: {miss}")
    for r in rows:
        try:
            s = C.validate_row(r, test_mode=False)
        except C.InvalidJob as e:
            errs.append(f"{r['job_id']}: {e}")
            continue
        if s.is_fake:
            errs.append(f"{r['job_id']}: variante de test")
        if s.variant == "stat" and s.save_ckpt:
            errs.append(f"{r['job_id']}: stat con save_ckpt")
        try:
            C.check_supported(s)
        except NotImplementedError as e:
            errs.append(f"{r['job_id']}: {e}")
    second = sorted({r["dataset"] for r in rows if r["block"] in C.ABLATION_BLOCKS
                     and not C.ABLATION_BLOCKS[r["block"]].get("all_datasets")
                     and not C.ABLATION_BLOCKS[r["block"]].get("datasets") and r["dataset"] != AW})
    if len(second) > 1:
        errs.append(f"ablaciones T2/T3 con mas de un segundo dataset {second} (analysis_plan.md §7.1: CIC2017 o el "
                    f"respaldo declarado KNAD:OS_Scan, no ambos)")
    guard = None
    if check_labels:
        attacks = {r["dataset"].split(":", 1)[1] for r in rows if r["pipeline"] == "corrected" and r["dataset"].startswith("KNAD:")}
        guard = benign_guard(data_root, attacks)
        for att, g in guard.items():
            if g["rows_checked"] < TRAIN_ROWS or g["n_attack"]:
                errs.append(f"guarda benigna {att}: {g} (se exige [0, {TRAIN_ROWS}) benigno)")
        others = {r["dataset"] for r in rows if r["pipeline"] == "corrected" and not r["dataset"].startswith("KNAD:")}
        if others:
            g_t0 = benign_guard_t0(others)
            guard.update(g_t0)
            for ds, g in g_t0.items():
                if g["problems"]:
                    errs.append(f"guarda benigna {ds} ({g['variant']}): {'; '.join(g['problems'])}.")
    return errs, guard


def immutability(existing_csv, new_rows, results_root):
    errs = []
    if not existing_csv.exists():
        return errs
    old = {r["job_id"]: r for r in C.read_manifest(existing_csv)}
    new = {r["job_id"]: r for r in new_rows}
    R = Path(results_root)
    for jid, r in old.items():
        touched = (R / jid).exists() or (R / ".locks" / jid).exists() or (R / ".state" / jid).exists()
        if not touched:
            continue
        if jid not in new:
            errs.append(f"{jid}: tiene resultados/lock/.state y desapareceria del manifiesto")
        elif {c: r.get(c, "") for c in C.MANIFEST_COLUMNS} != {c: new[jid].get(c, "") for c in C.MANIFEST_COLUMNS}:
            errs.append(f"{jid}: tiene resultados/lock/.state y su fila cambiaria")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Genera o comprueba manifest.csv del runner (JISA; version ACTIVE).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--print", action="store_true")
    g.add_argument("--write-candidate", action="store_true",
                   help="Runner v2: escribe el manifiesto CANDIDATO (--candidate-out); nunca manifest.csv.")
    g.add_argument("--check-candidate", action="store_true")
    g.add_argument("--print-candidate", action="store_true")
    g.add_argument("--write-candidate-v3", action="store_true",
                   help="Runner v3: escribe manifest_v3_candidate.csv (--candidate-out-v3); nunca manifest.csv.")
    g.add_argument("--check-candidate-v3", action="store_true")
    g.add_argument("--print-candidate-v3", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--candidate-out", default=str(CANDIDATE_OUT))
    ap.add_argument("--candidate-out-v3", default=str(CANDIDATE_OUT_V3))
    ap.add_argument("--ablation-second-dataset", default="CIC2017", choices=["CIC2017", "KNAD:OS_Scan"],
                    help="Segundo dataset de T2/T3 (analysis_plan.md §7.1; KNAD:OS_Scan = respaldo declarado).")
    ap.add_argument("--results-root", default=str(C.RESULTS_ROOT))
    ap.add_argument("--data-root", default=str(C.DATA_ROOT))
    ap.add_argument("--no-label-guard", action="store_true", help="Omite la guarda benigna (solo depuracion).")
    ap.add_argument("--allow-remove", action="store_true",
                    help="--write: permite quitar filas del manifest.csv actual sin resultados (por defecto se niega).")
    a = ap.parse_args(argv)
    if a.write_candidate_v3 or a.check_candidate_v3 or a.print_candidate_v3:
        return main_candidate_v3(a)
    if a.write_candidate or a.check_candidate or a.print_candidate:
        return main_candidate(a)
    rows = generate()
    text = to_csv(rows)
    errs, guard = validate(rows, a.data_root, check_labels=not a.no_label_guard)
    if errs:
        print("ERRORES DE VALIDACION:\n  " + "\n  ".join(errs))
        return 1
    fam = {}
    for r in rows:
        fam[r["block"]] = fam.get(r["block"], 0) + 1
    summary = f"{len(rows)} filas: " + ", ".join(f"{b}={n}" for b, n in fam.items())
    if guard:
        summary += " | guarda benigna [0, 200000): " + ", ".join(f"{k}={v['n_attack']}" for k, v in guard.items())
    out = Path(a.out)
    if a.print:
        print(text, end="")
        return 0
    if a.check:
        cur = out.read_text() if out.exists() else ""
        if cur != text:
            print(f"DIFIERE: {out} no coincide con lo generado ({summary})")
            return 1
        print(f"OK: {out} coincide ({summary})")
        return 0
    ierr = immutability(out, rows, a.results_root)
    if out.exists() and not a.allow_remove:
        gone = sorted({r["job_id"] for r in C.read_manifest(out)} - {r["job_id"] for r in rows})
        if gone:
            ierr.append(f"{len(gone)} filas de {out.name} desaparecerian (ACTIVE={ACTIVE!r}; ¿promocion sin cambiar "
                        f"ACTIVE?); usa --allow-remove si es intencionado: {gone[:3]}")
    if ierr:
        print("NO SE ESCRIBE (inmutabilidad):\n  " + "\n  ".join(ierr))
        return 1
    C.write_text_atomic(out, text)
    print(f"escrito {out} ({summary})")
    return 0


def main_candidate(a):
    """Entry point of --write-candidate, --check-candidate and --print-candidate (runner v2), with the validation of --write."""
    out = Path(a.candidate_out)
    if C.realpath(out) == C.realpath(a.out) or out.name == OUT.name:
        print(f"NO SE ESCRIBE: el candidato no puede ser {OUT.name} ({out})")
        return 1
    rows = generate_v2(a.ablation_second_dataset)
    text = to_csv(rows)
    errs, guard = validate(rows, a.data_root, check_labels=not a.no_label_guard)
    errs += check_v1_preserved(None, rows, text)
    errs += immutability(Path(a.out), rows, a.results_root)
    if errs:
        print("ERRORES DE VALIDACION DEL CANDIDATO:\n  " + "\n  ".join(errs[:40]))
        return 1
    fam = {}
    for r in rows:
        fam[r["block"]] = fam.get(r["block"], 0) + 1
    summary = (f"{len(rows)} filas (v1 = {len(generate_v1())} identicas al tag {V1_TAG}): " + ", ".join(f"{b}={n}" for b, n in fam.items())
               + f" | ablaciones: AW + {a.ablation_second_dataset}")
    if guard:
        summary += " | guarda benigna [0, T) (T = C.CORRECTED_TRAIN_ROWS): " + ", ".join(
            f"{k}={v['n_attack']}" for k, v in guard.items())
    if a.print_candidate:
        print(text, end="")
        return 0
    if a.check_candidate:
        cur = out.read_text() if out.exists() else ""
        if cur != text:
            print(f"DIFIERE: {out} no coincide con lo generado ({summary})")
            return 1
        print(f"OK: {out} coincide ({summary})")
        return 0
    C.write_text_atomic(out, text)
    print(f"escrito {out} ({summary})")
    return 0


def main_candidate_v3(a):
    """Entry point of --write-candidate-v3, --check-candidate-v3 and --print-candidate-v3 (runner v3)."""
    out = Path(a.candidate_out_v3)
    cur = Path(a.out)
    if C.realpath(out) == C.realpath(cur) or out.name in (OUT.name, CANDIDATE_OUT.name):
        print(f"NO SE ESCRIBE: el candidato v3 no puede ser {OUT.name} ni {CANDIDATE_OUT.name} ({out})")
        return 1
    rows = generate_v3(a.ablation_second_dataset)
    text = to_csv(rows)
    errs, guard = validate(rows, a.data_root, check_labels=not a.no_label_guard)
    errs += check_v1_preserved(None, rows, text)
    if cur.exists():
        errs += check_rows_preserved(cur.read_text(), rows, text, f"vigente ({cur.name})")
    else:
        errs.append(f"{cur} no existe: no se puede comprobar que las filas vigentes no cambian")
    errs += immutability(cur, rows, a.results_root)
    if errs:
        print("ERRORES DE VALIDACION DEL CANDIDATO v3:\n  " + "\n  ".join(errs[:40]))
        return 1
    fam = {}
    for r in rows:
        fam[r["block"]] = fam.get(r["block"], 0) + 1
    n_cur = len(C.read_manifest(cur)) if cur.exists() else 0
    summary = (f"{len(rows)} filas ({n_cur} de {cur.name} identicas byte a byte; +{len(rows) - n_cur} nuevas): "
               + ", ".join(f"{b}={n}" for b, n in fam.items()))
    if guard:
        summary += " | guarda benigna [0, T): " + ", ".join(f"{k}={v['n_attack']}" for k, v in guard.items())
    if a.print_candidate_v3:
        print(text, end="")
        return 0
    if a.check_candidate_v3:
        have = out.read_text() if out.exists() else ""
        if have != text:
            print(f"DIFIERE: {out} no coincide con lo generado ({summary})")
            return 1
        print(f"OK: {out} coincide ({summary})")
        return 0
    C.write_text_atomic(out, text)
    print(f"escrito {out} ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
