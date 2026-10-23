#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate.py - agregacion de metrics.json por celda (dataset x configuracion) de la revision JISAS-D-26-02300.

Especificacion: revision/analysis/analysis_plan.md (tag analysis-plan-frozen) §4 (semillas, configuraciones C1-C11,
regla S*_d, MLP = media de replicas, original/stat una vez), §7.1 (referencia de ruido de las ablaciones), §12.2
(celdas perdidas, divergencia, intentos conservados, versiones del runner) y §14.2 (sha256 de entradas y salidas).
Contrato de entrada: <job_dir>/metrics.json (schema "jisa-metrics-rev/1", METRICS_REV.md) + manifest.csv + config.json
+ .state/.locks del runner. No calcula ninguna metrica nueva: solo medias de las del job y diferencias por celda.

Salidas (por defecto en --results-root):
    runs.csv                      una fila por job y analisis/candidata/sensibilidad/serie (todas las ejecuciones;
                                  tambien perdidas, no ejecutadas y superadas, con su uso)
    aggregate/cells.csv           tabla (dataset, configuracion): valores por semilla, media sin redondear, SD, min-max
    aggregate/aggregate.json      todo lo anterior en JSON (lo lee stats.py): S*_d, celdas perdidas, conjuntos de
                                  valores (main, s_itt, s_div_noimp, s_seed), SD_pool, Delta_MLP, ablaciones, O1
    aggregate/SHA256SUMS          sha256 de las salidas (y de aggregate.json, que ademas lista las entradas)

Uso (entorno del plan, RF-2; R-7 como metrics_rev: sin tag o con plan distinto aborta con codigo 3):
    R=<raiz del repositorio>
    nice -n 19 env OMP_NUM_THREADS=4 $R/envs/postproc/bin/python $R/revision/experiments/aggregate.py \
        [--results-root ABS] [--manifest ABS] [--out-dir ABS] [--ladder-json ABS] [--compute-missing [--blocks B]]
Salida: 0 OK; 1 salidas escritas pero incompletas o inconsistentes (jobs DONE sin metrics.json valido, versiones de
metrics_rev mezcladas...); 2 uso; 3 guarda R-7 o de entorno (no se escribe nada).
Lecturas literales de huecos del plan: AG_DEVIATIONS_PROPOSED (tambien en aggregate.json y AGG_STATS.md).
"""
import argparse
import csv
import datetime as dt
import io
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
import metrics_rev as M

SCHEMA_AGG = "jisa-aggregate/1"
AGGREGATE_VERSION = "1.3.0"
SEED_ORDER = ("A", "B", "C")
DATASETS = [("D1", "KNAD:Active_Wiretap"), ("D2", "KNAD:Fuzzing"), ("D3", "KNAD:OS_Scan"), ("D4", "KNAD:SSDP_Flood"),
            ("D5", "KNAD:SSL_Renegotiation"), ("D6", "KNAD:Video_Injection"), ("D7", "ACI-IOT-2023"),
            ("D8", "CIC2017"), ("D9", "CIC2018")]
DS_ID = {name: did for did, name in DATASETS}
DS_NAME = {did: name for did, name in DATASETS}
CORPUS = {"D1": "KNAD", "D2": "KNAD", "D3": "KNAD", "D4": "KNAD", "D5": "KNAD", "D6": "KNAD", "D7": "ACI",
          "D8": "CIC2017", "D9": "CIC2018"}
T1_CORPUS_BLOCK = {"D7": "ACI", "D8": "CIC2017", "D9": "CIC2018"}
ROWCFGS = ["conv1d-AR", "conv1d-TSR", "conv2d-AR", "conv2d-TSR", "lstm-AR", "lstm-TSR", "mha-AR", "mha-TSR",
           "mlp-AR", "mlp-TSR", "original", "stat"]
ROWCFG_TO_CONFIG = {"conv1d-AR": "C1", "conv1d-TSR": "C2", "conv2d-AR": "C3", "conv2d-TSR": "C4", "lstm-AR": "C5",
                    "lstm-TSR": "C6", "mha-AR": "C7", "mha-TSR": "C8", "mlp-AR": "C9", "mlp-TSR": "C9",
                    "original": "C10", "stat": "C11"}
CONFIGS = [f"C{i}" for i in range(1, 12)]
CONFIG_ROWCFGS = {c: [r for r in ROWCFGS if ROWCFG_TO_CONFIG[r] == c] for c in CONFIGS}
CONFIG_NAMES = {"C1": "Conv1D-AR", "C2": "Conv1D-TSR", "C3": "Conv2D-AR", "C4": "Conv2D-TSR", "C5": "LSTM-AR",
                "C6": "LSTM-TSR", "C7": "MHA-AR", "C8": "MHA-TSR", "C9": "MLP", "C10": "original (KitNET-style)",
                "C11": "STAT"}
CTX_CONFIG_NAMES = dict(M.CTX_CONFIG_NAMES)
CTX_CONFIGS = list(CTX_CONFIG_NAMES)
DETERMINISTIC_ROWCFGS = ("original", "stat")
NONDET_CONFIGS = [f"C{i}" for i in range(1, 10)]
X4 = ("FUS", "KMD_f", "MVD_f", "RAW")
CONF_KEYS = ("auroc_FUS", "auc_loc_RAW", "auroc_KMD_f", "auroc_MVD_f", "auprc_FUS", "f1_FUS")
AUROC_CONF_KEYS = ("auroc_FUS", "auroc_KMD_f", "auroc_MVD_f")
RAW_KEYS = set(CONF_KEYS)
ROPE_NOISE_FLOOR = 0.01
TAGS = {"conf": "[CONFIRMATORIO plan §5]", "sens": "[SENSIBILIDAD plan §2.9]", "t4": "[SENSIBILIDAD plan §8]",
        "abl": "[SENSIBILIDAD plan §7]", "expl": "[EXPLORATORIO]", "o1": "[ORACULO: as-submitted (O1)]",
        "o2": "[ORACULO: evaluation-region oracle F1 (O2)]", "sec": "[SECUNDARIA plan §3.2]",
        "itt": "[SENSIBILIDAD plan §12.2 S-ITT]", "dupcal": "[EXPLORATORIO] S-DUPCAL (F-01; fuera del plan)",
        "post_freeze": "[EXPLORATORIO] post-freeze (DONE despues de la congelacion; §4.5)"}

AG_DEVIATIONS_PROPOSED = [
    "AG-1 (§4.3, §4.5) C9 (MLP) is the mean of the AR and TSR replicas with the same seed; if either of the two is a "
    "lost cell, C9 is a lost cell (it is not replaced by the remaining replica, which would be another estimator). A "
    "replica that was not run shortens S*_d like any other cell, unless its sibling is a lost cell: C9 is then "
    "already lost and the remaining replica contributes to no test, so it counts as lost for the prefix (ST-m5).",
    "AG-2 (§4.5) 'cell' = row of the manifest per (variant, paradigm) across the seeds. S*_d is the largest k in {3, "
    "2, 1, 0} such that, with lost = cells FAILED/BLOCKED/not executable/removed by the contingency sequence in some "
    "seed of (A, B, C)[:k], all the others have DONE with a valid metrics.json in every seed of the prefix. "
    "original/stat: their single run (seed A) stands for every seed; if it was not run it shortens every prefix.",
    "AG-3 (§4.5, §12.2) a DONE job without a valid metrics.json (absent, different schema or state, different plan "
    "or inputs, non-finite primary) counts as not finished for S*_d (it shortens the prefix) and is listed; "
    "aggregate exits with code 1. A metrics.json with section_errors (sensitivity sections, T4-P, T4-E or S-DUPCAL) "
    "is valid for the primary: only those sections are NA, a warning is issued and aggregate.complete = false "
    "(stats.py then runs on partial inputs) until the metrics are recomputed (F-04).",
    "AG-4 (§12.2, §13.3) escalera: --ladder-json lista job_ids eliminados. En T2-T4 son celdas perdidas (en todas las "
    "semillas); en T1 (escalones 5, 6 y 10: semilla C o B no ejecutada) cuentan como no ejecutadas y acortan S*_d.",
    "AG-5 (§12.2) divergencia (nonfinite.divergent): en los analisis confirmatorios se usan divergence.imputed_itt "
    "(ROC-AUC FUS 0,5; AUC_loc(RAW) 0,5; PR-AUC p; F1 0) y, por la misma regla 'ROC-AUC = 0,5', ROC-AUC de KMD_f y MVD_f "
    "= 0,5 (entran en H3). La sensibilidad 'sin imputar' (s_div_noimp) elimina la celda por comparacion.",
    "AG-6 (§12.2 S-ITT) celdas perdidas imputadas con ROC-AUC = 0,5 (FUS, KMD_f, MVD_f); el resto de metricas de la "
    "celda quedan sin valor (AUC_loc no se imputa: la regla de robustez §5.8 no se aplica a sensibilidades).",
    "AG-7 (§4.5 S-SEED) prefijo comun L = min_d |S*_d| sobre los 9 datasets con el MISMO conjunto de celdas perdidas del "
    "analisis principal; si algun dataset tiene |S*_d| = 0, S-SEED no es calculable.",
    "AG-8 (§7.1) referencia de una ablacion = el job de la rejilla T1 del bloque de referencia (T1-A; T1-B para T2-B) con "
    "el mismo dataset, variante y paradigma (MLP de T3: la replica AR, no C9); en T4-E la referencia c = 0 es "
    "t4e_reference de ese job. Delta = ablacion - referencia (ROC-AUC de FUS; PR-AUC y F1 como secundarias).",
    "AG-9 (§7.1) SD_pool(d)^2 = media de las varianzas entre semillas (ddof = 1, sobre S*_d) de ROC-AUC de FUS de las "
    "configuraciones C1-C9 no perdidas (C9: media de replicas por semilla) con los valores SIN imputar: las ablaciones "
    "son descriptivas y la ITT de §12.2 solo rige en los analisis confirmatorios; las semillas divergidas se excluyen de "
    "la varianza de su configuracion (hacen falta >= 2 semillas); gl = sum_c (n_semillas_c - 1). La SD y el min-max de "
    "cada celda usan esos mismos valores; la version con ITT se reporta aparte (sufijo _itt). Umbral = max(sqrt(2) "
    "SD_pool(d), 0,01); 0,01 en original/stat y si |S*_d| = 1. En T3 la MLP es una sola replica (AR) y su umbral usa "
    "SD_pool calculado con C9 (media de 2 replicas), que infraestima el ruido de una replica: se declara y cada celda "
    "MLP de T3 lleva |Delta_MLP| de la semilla A (ruido entre replicas, §3.2) al lado (ST-m7).",
    "AG-10 (§12.2) solo cuentan los metrics.json con el mismo metrics_rev_sha256 y los mismos helpers_sha256 "
    "(metrics_rev.py, common.py y pipelines.py: la logica de regiones y de F_cal vive en los auxiliares; F-05). Un "
    "cambio exige recalculo completo (§14.2): si hay mezcla, aggregate lo marca como error de consistencia y stats.py "
    "se niega a correr.",
    "AG-11 (§4.5 'at the freeze of the results') a T1-T4 job whose DONE is later than --freeze (default "
    "common.FREEZE = 2026-09-30 23:59:59, local time; time of the DONE = its ISO content written by the runner or, "
    "if it cannot be read, its mtime) had not finished at the freeze: it shortens S*_d like a cell that was not run, "
    "enters no value set and no ablation as confirmatory, and is listed in runs.csv as [EXPLORATORIO] post-freeze "
    "(ST-M4).",
    "AG-12 (§4.3) original/stat: el valor de la celda es el de su UNICA ejecucion (semilla A), no la media de |S*_d| "
    "copias identicas (que puede diferir en 1 ulp y crear o romper empates; ST-m3).",
    "AG-13 (§3.6, §14.1) las cotas O2 van en runs.csv en filas propias (analysis = 'O2', etiqueta [ORACULO]); las filas "
    "'main' no llevan columnas oraculo (F-03). Una fila por intento conservado (analysis = 'attempt', §4.6; ST-m9).",
    "AG-14 (§14.4 hito (i), §6.1) la hora del primer metrics.json (global y por dataset) es el minimo del registro "
    "append-only de metrics_rev (.metrics_rev/, MR-12), de su log y de los 'created' de los metrics.json agregados "
    "(F-10); stats.py la usa en la guarda de D (ST-M3).",
    "AG-15 (outside the frozen plan) the blocks T7-MLPW and T7-MLPDEEP (variants 'mlp_ctx' and 'mlp_deep_ctx', "
    "configurations C12-C15) are [EXPLORATORIO]: they are loaded into a collection of their own (load_ctx_jobs) and "
    "do not enter A.jobs, cells, S*_d, SD_pool, value_sets, the ablations of §7-§8, 'counts' or 'complete' (a T7 "
    "cell without a value blocks nothing confirmatory). They are published in aggregate.json['exploratory_ctx'] and "
    "in rows of runs.csv with in_S_star = false and the label [EXPLORATORIO]. AR and TSR are not averaged (unlike "
    "C9): these variants do read the window, so each paradigm is recorded as a separate configuration, as in C1-C8. "
    "Note: it was measured that in 'mlp_ctx' with L = 800 AR and TSR are almost the same estimator (rank correlation "
    "0.99973 on real data), so C12 and C13 cannot support any claim. The comparison with C9 of the same dataset is "
    "descriptive and is always published with noise_threshold and abs_delta_MLP_seedA beside it: the main reference "
    "is C9 of the same seed (A), not the mean over S*_d.",
]


def fnum(v):
    """float finito o None (JSON estricto)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def rowcfg_of(variant, ar):
    if variant in DETERMINISTIC_ROWCFGS:
        return variant
    return f"{variant}-{'AR' if ar else 'TSR'}"


def csv_cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else ""
    return str(v)


def write_csv_atomic(path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([csv_cell(r.get(h)) for h in header])
    C.write_text_atomic(path, buf.getvalue())


def mean_or_none(vals):
    if not vals or any(v is None for v in vals):
        return None
    return float(sum(vals) / len(vals)) if len(vals) != 1 else float(vals[0])


def sd_ddof1(vals):
    if vals is None or len(vals) < 2 or any(v is None for v in vals):
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def done_time(R, jid):
    """Time of the DONE marker of a job: the ISO timestamp written by the runner or, if unreadable, the file mtime."""
    p = R / jid / "DONE"
    try:
        return _ts(p.read_text(encoding="utf-8").strip()), "DONE (contenido)"
    except Exception:
        return _ts(dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")), "DONE (mtime)"


def job_state(R, jid, ladder, freeze=None):
    """State of a job as the runner left it, without reading its metrics."""
    if (R / jid / "DONE").exists():
        if freeze is not None and done_time(R, jid)[0] > freeze:
            return "DONE_POST_FREEZE"
        return "DONE"
    st = R / ".state" / jid
    if st.exists() and any(st.glob("nonexec_*.json")):
        return "LOST_NONEXEC"
    if C.read_fail_count(R, jid) >= C.MAX_FAILS:
        return "LOST_BLOCKED"
    if jid in ladder:
        return "LADDER"
    if (R / ".locks" / jid).exists():
        return "RUNNING"
    return "PENDING_RETRY" if C.read_fail_count(R, jid) > 0 else "PENDING"


def preserved_attempts(R, jid):
    """§12.2: intentos con scores conservados en .state/<job>/ (se listan; solo cuenta el intento con DONE)."""
    st = R / ".state" / jid
    if not st.exists():
        return []
    return sorted(d.name for d in st.iterdir() if d.is_dir() and d.name.startswith(("attempt_", "duplicate_"))
                  and (d / "scores.npy").exists())


def check_metrics(R, row, plan_sha):
    """Valida <job>/metrics.json frente al manifiesto, la config y el plan congelado. Devuelve (estado, dict|None, sha)."""
    J = R / row["job_id"]
    p = J / "metrics.json"
    if not p.exists():
        return "missing", None, None, None
    sha = C.sha256_file(p)
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return f"invalid: JSON ilegible ({type(e).__name__})", None, sha, None
    probs = []
    if m.get("schema") != M.SCHEMA:
        probs.append(f"schema {m.get('schema')!r}")
    if m.get("status") != "ok":
        probs.append(f"status {m.get('status')!r}")
    if m.get("job_id") != row["job_id"]:
        probs.append("job_id distinto")
    job = m.get("job") or {}
    for k in ("block", "dataset", "variant", "seed", "pipeline", "pipeline_version"):
        if str(job.get(k)) != str(row.get(k)):
            probs.append(f"job.{k} {job.get(k)!r} != manifiesto {row.get(k)!r}")
    if job.get("ar") is not None and bool(job.get("ar")) != C.parse_bool(row.get("ar")):
        probs.append("job.ar distinto del manifiesto")
    prov = m.get("provenance") or {}
    if (prov.get("plan") or {}).get("plan_sha256") != plan_sha:
        probs.append("plan_sha256 distinto del plan congelado")
    inp = prov.get("inputs") or {}
    cfgp = J / "config.json"
    if cfgp.exists() and inp.get("config_sha256") != C.sha256_file(cfgp):
        probs.append("config.json cambio desde el calculo")
    arr = (C.read_json(J / "arrays.json") or {}).get("scores.npy") or {}
    if arr.get("sha256") and inp.get("scores_sha256_arrays_json") != arr.get("sha256"):
        probs.append("scores.npy (arrays.json) cambio desde el calculo")
    prim = m.get("primary") or {}
    if fnum(prim.get("value")) is None:
        probs.append("primary.value no finito")
    if not ((m.get("label_consistency") or {}).get("hard") or {}).get("regions"):
        probs.append("label_consistency.hard.regions falso")
    if probs:
        return "invalid: " + "; ".join(probs), None, sha, prov.get("metrics_rev_sha256")
    return "ok", m, sha, prov.get("metrics_rev_sha256")


def _lvl(c):
    return f"{float(c):g}"


def job_metrics(m):
    """Diccionario plano {clave: valor} con las metricas del job que usan aggregate/stats (valores tal cual)."""
    out = {}
    E = m["E"]
    triv = E["trivial"]
    out["auroc_FUS"] = fnum(m["primary"]["value"])
    out["auc_loc_RAW"] = fnum(m["primary"].get("auc_loc_raw"))
    for k in ("auc_t", "prevalence", "f1_trivial", "ap_trivial", "n", "n_attack", "duration_hours"):
        out[f"E_{k}"] = fnum(triv.get(k))
    sloc = (m.get("sensitivities") or {}).get("S-LOC") or {}
    for X in X4:
        b = E.get(X) or {}
        if X != "FUS":
            out[f"auroc_{X}"] = fnum(b.get("auroc"))
        else:
            out["auroc_FUS_fast"] = fnum(b.get("auroc"))
        for k in ("auprc", "apn", "eer", "threshold", "f1", "precision", "recall", "fpr", "alarm_rate", "f1n"):
            out[f"{k}_{X}"] = fnum(b.get(k))
        out[f"f1_le_trivial_{X}"] = (None if b.get("f1_le_trivial_plus_0.02") is None
                                     else float(bool(b.get("f1_le_trivial_plus_0.02"))))
        wd = b.get("windowdiff") or {}
        out[f"wd_{X}"] = fnum(wd.get("wd"))
        out[f"wd_k_{X}"] = fnum(wd.get("k"))
        if b.get("o2"):
            out[f"o2_f1_max_{X}"] = fnum(b["o2"].get("f1_max"))
            out[f"o2_gap_{X}"] = fnum(b["o2"].get("gap_vs_calibrated_f1"))
        out[f"auc_loc_{X}"] = fnum((sloc.get(X) or {}).get("auc_loc"))
    S = m.get("sensitivities") or {}
    for name, xs in (("S-NF", ("FUS_NF", "KMD_raw", "MVD_raw")), ("S-MAX", ("FUS_MAX",))):
        for X in xs:
            b = (S.get(name) or {}).get(X) or {}
            for k in ("auroc", "auprc", "f1"):
                out[f"{k}_{X}"] = fnum(b.get(k))
    for name, tag in (("S-Q95", "q95"), ("S-Q995", "q995")):
        for X in X4:
            b = (S.get(name) or {}).get(X) or {}
            for k in ("threshold", "f1", "precision", "recall", "fpr"):
                out[f"{k}_{X}_{tag}"] = fnum(b.get(k))
    for name, tag in (("S-PS", "ps"), ("S-DUP", "dup"), ("S-INS", "ins"), ("S-DUPCAL", "dupcal")):
        blk = S.get(name) or {}
        if blk.get("applicable") is False or blk.get("error"):
            continue
        for X in X4:
            b = blk.get(X) or {}
            for k in ("auroc", "auprc", "threshold", "f1", "precision", "recall", "fpr"):
                out[f"{k}_{X}_{tag}"] = fnum(b.get(k))
    for X in X4:
        out[f"wd_ep_{X}"] = fnum(((S.get("S-EP") or {}).get(X) or {}).get("wd"))
    op = m.get("operational") or {}
    for g in ("G10000", "G800"):
        b = op.get(g) or {}
        for k in ("n_episodes", "n_detected", "detection_rate", "median_delay_rows", "median_delay_s",
                  "n_false_alarm_episodes", "false_alarm_episodes_per_1e6_benign", "false_alarm_episodes_per_hour"):
            out[f"op_{g}_{k}"] = fnum(b.get(k))
    t4p = m.get("t4p") or {}
    a = t4p.get("a") or {}
    if a.get("applicable"):
        out["t4pa_natural_contamination"] = fnum(a.get("natural_calib_contamination"))
        out["t4pa_n_clean"] = fnum(a.get("n_clean"))
        out["t4pa_estimable"] = float(bool(a.get("estimable")))
        for X, d in (a.get("diff_natural_minus_clean") or {}).items():
            for k, v in d.items():
                out[f"t4pa_diff_{X}_{k}"] = fnum(v)
    b = t4p.get("b") or {}
    if b.get("applicable"):
        for lv in b.get("levels") or []:
            c = _lvl(lv["c"])
            out[f"t4pb_c{c}_eff"] = fnum(lv.get("effective_contamination"))
            for X in ("FUS", "KMD_f", "MVD_f"):
                for k, v in ((lv.get("metrics") or {}).get(X) or {}).items():
                    out[f"t4pb_c{c}_{X}_{k}"] = fnum(v)
    cc = t4p.get("c") or {}
    for f, blk in (cc.get("fractions") or {}).items():
        out[f"t4pc_f{f}_eff"] = fnum(blk.get("effective_contamination"))
        for X in ("FUS", "KMD_f", "MVD_f"):
            for k, v in ((blk.get("metrics") or {}).get(X) or {}).items():
                out[f"t4pc_f{f}_{X}_{k}"] = fnum(v)
    pdy = m.get("per_day") or {}
    if pdy.get("available"):
        for d, rec in (pdy.get("days") or {}).items():
            out[f"day_{d}_fpr_FUS"] = fnum(rec.get("fpr"))
            if rec.get("eligible_auc_per_day"):
                out[f"day_{d}_auroc_FUS"] = fnum(rec.get("auroc_fus"))
                out[f"day_{d}_auc_t"] = fnum(rec.get("auc_t"))
    pty = m.get("per_attack_type") or {}
    if pty.get("available"):
        for t, rec in (pty.get("types") or {}).items():
            for X in X4:
                out[f"type_{t}_recall_{X}"] = fnum(rec.get(f"recall_{X}"))
    ref = m.get("t4e_reference") or {}
    if ref and not ref.get("error"):
        for X in X4:
            bx = ref.get(X) or {}
            for k in ("auroc", "auprc", "f1"):
                out[f"t4eref_{k}_{X}"] = fnum(bx.get(k))
    return out


def imputed_itt(m, flat):
    """Values that the confirmatory analyses use for a run that diverged (plan §12.2)."""
    it = (m.get("divergence") or {}).get("imputed_itt") or {}
    out = dict(flat)
    out["auroc_FUS"] = fnum(it.get("auroc_fus", 0.5))
    out["auc_loc_RAW"] = fnum(it.get("auc_loc_raw", 0.5))
    out["auprc_FUS"] = fnum(it.get("auprc_fus", flat.get("E_prevalence")))
    out["f1_FUS"] = fnum(it.get("f1_fus", 0.0))
    out["auroc_KMD_f"] = 0.5
    out["auroc_MVD_f"] = 0.5
    return out


class Agg:
    pass


def load_all(R, manifest_rows, plan_sha, ladder, freeze=None):
    """Estado y metricas de cada fila T1-T4 del manifiesto y de los directorios de resultados que no estan en el."""
    A = Agg()
    A.jobs = {}
    A.consistency_errors, A.warnings = [], []
    A.section_error_jobs = []
    A.freeze = freeze
    mr_shas, helper_shas = {}, {}
    firsts, firsts_ds = [], {}
    in_manifest = set()
    for row in manifest_rows:
        fam = C.block_family(row["block"])
        if fam not in ("T1", "T2", "T3", "T4") or row.get("pipeline") != "corrected":
            continue
        jid = row["job_id"]
        in_manifest.add(jid)
        st = job_state(R, jid, ladder, freeze)
        att = preserved_attempts(R, jid)
        rec = {"row": row, "job_id": jid, "family": fam, "state": st, "metrics_state": None, "m": None,
               "sha256": None, "n_attempts_preserved": len(att), "attempts_preserved": att,
               "dataset_id": DS_ID.get(row["dataset"]), "rowcfg": rowcfg_of(row["variant"], C.parse_bool(row["ar"])),
               "seed": row.get("seed")}
        if st in ("DONE", "DONE_POST_FREEZE"):
            rec["done_time"] = done_time(R, jid)[0].isoformat(timespec="seconds")
            ms, m, sha, mr = check_metrics(R, row, plan_sha)
            rec.update(metrics_state=ms, sha256=sha)
            if ms == "ok":
                rec["m"] = m
                rec["flat"] = job_metrics(m)
                rec["divergent"] = bool((m.get("nonfinite") or {}).get("divergent"))
                rec["flat_conf"] = imputed_itt(m, rec["flat"]) if rec["divergent"] else rec["flat"]
                mr_shas.setdefault(mr, []).append(jid)
                hk = json.dumps((m.get("provenance") or {}).get("helpers_sha256"), sort_keys=True)
                helper_shas.setdefault(hk, []).append(jid)
                for t in ((m.get("provenance") or {}).get("first_metrics_json_created"), m.get("created")):
                    if t:
                        firsts.append(str(t))
                if m.get("created") and rec["dataset_id"]:
                    firsts_ds.setdefault(rec["dataset_id"], []).append(str(m["created"]))
                if m.get("section_errors"):
                    rec["section_errors"] = list(m["section_errors"])
                    if st == "DONE":
                        A.section_error_jobs.append(jid)
                    A.warnings.append(f"{jid}: section_errors {m['section_errors']} -> esas secciones NA; el primario "
                                      "cuenta; aggregate incompleto hasta recalcular (F-04, AG-3)")
                if not (m.get("label_consistency") or {}).get("all_soft_match", True):
                    A.warnings.append(f"{jid}: label_consistency.all_soft_match = false (se registra; no bloquea)")
            else:
                A.warnings.append(f"{jid}: DONE sin metrics.json valido ({ms}) -> cuenta como no terminado (AG-3)")
            if st == "DONE_POST_FREEZE":
                A.warnings.append(f"{jid}: DONE {rec['done_time']} posterior a la congelacion -> no cuenta para S*_d "
                                  "ni para los tests; [EXPLORATORIO] (AG-11)")
            o1p = R / jid / "metrics_o1.json"
            if o1p.exists():
                rec["o1"] = C.read_json(o1p)
                rec["o1_sha256"] = C.sha256_file(o1p)
        A.jobs[jid] = rec
    if len(mr_shas) > 1:
        A.consistency_errors.append(
            "metrics.json calculados con varias versiones de metrics_rev.py (AG-10, §14.2): " +
            "; ".join(f"{k}: {len(v)} jobs" for k, v in mr_shas.items()))
    if len(helper_shas) > 1:
        A.consistency_errors.append(
            "metrics.json calculados con varias versiones de los auxiliares (helpers_sha256: metrics_rev/common/"
            "pipelines; AG-10, F-05): " + "; ".join(f"{len(v)} jobs con {k[:120]}" for k, v in helper_shas.items()))
    A.metrics_rev_sha256 = sorted(k for k in mr_shas if k)
    first_rec, log = M.read_chronology(R)

    def _confirmatory_entry(e):
        """Whether an entry of the metrics chronology counts as confirmatory (exploratory families do not)."""
        if e.get("exploratory_arm"):
            return False
        fam = e.get("family") or C.block_family(str(e.get("job_id") or "").split("__")[0])
        return fam not in C.EXPLORATORY_FAMILIES

    n_expl = sum(1 for e in log if not _confirmatory_entry(e))
    A.chronology_record = {"first": first_rec, "n_log_entries": len(log),
                           "n_log_entries_exploratory_excluded": n_expl,
                           "rule_exploratory": "The entries of the family T7 ([EXPLORATORIO]) do not enter "
                                               "first_metrics_json_created or its per-dataset version",
                           "path": str(R / M.CHRONO_DIR)}
    if first_rec and first_rec.get("created") and _confirmatory_entry(first_rec):
        firsts.append(str(first_rec["created"]))
    for e in log:
        if e.get("created") and _confirmatory_entry(e):
            firsts.append(str(e["created"]))
            if e.get("dataset_id"):
                firsts_ds.setdefault(e["dataset_id"], []).append(str(e["created"]))
    A.first_metrics_json_created = min(firsts, key=_ts) if firsts else None
    A.first_metrics_json_created_by_dataset = {d: min(v, key=_ts) for d, v in sorted(firsts_ds.items())}
    A.not_in_manifest = []
    if R.exists():
        for d in sorted(R.iterdir()):
            if not d.is_dir() or d.name.startswith((".", "_")) or d.name in in_manifest:
                continue
            fam = C.block_family(d.name.split("__")[0])
            if fam in ("T1", "T2", "T3", "T4"):
                A.not_in_manifest.append({"job_id": d.name, "done": (d / "DONE").exists(),
                                          "has_metrics": (d / "metrics.json").exists(),
                                          "use": "fuera del manifiesto vigente (superada o ajena): no cuenta (§12.2)"})
    return A


def load_ctx_jobs(R, manifest_rows, plan_sha, ladder, freeze=None):
    """Collect the rows of the exploratory blocks T7-* and T8-* apart from the confirmatory jobs."""
    out = []
    for row in manifest_rows:
        fam = C.block_family(row["block"])
        if fam not in C.EXPLORATORY_FAMILIES or row.get("pipeline") != "corrected":
            continue
        jid = row["job_id"]
        ar = C.parse_bool(row["ar"])
        cid = M.CTX_CONFIG_IDS.get((row["variant"], ar))
        st = job_state(R, jid, ladder, freeze)
        rec = {"job_id": jid, "block": row["block"], "family": fam, "dataset": row["dataset"],
               "dataset_id": DS_ID.get(row["dataset"]), "corpus": CORPUS.get(DS_ID.get(row["dataset"])),
               "config_id": cid, "config_name": CTX_CONFIG_NAMES.get(cid), "variant": row["variant"], "ar": ar,
               "seed": row.get("seed"), "state": st, "metrics_state": None, "m": None, "sha256": None,
               "row": row, "n_attempts_preserved": len(preserved_attempts(R, jid)),
               "attempts_preserved": preserved_attempts(R, jid),
               "reference_block": f"T1-A-{T1_CORPUS_BLOCK.get(DS_ID.get(row['dataset']), 'KNAD')}",
               "tag": TAGS["expl"]}
        if st in ("DONE", "DONE_POST_FREEZE"):
            rec["done_time"] = done_time(R, jid)[0].isoformat(timespec="seconds")
            ms, m, sha, mr = check_metrics(R, row, plan_sha)
            rec.update(metrics_state=ms, sha256=sha, metrics_rev_sha256=mr)
            if ms == "ok":
                rec["m"] = m
                rec["flat"] = job_metrics(m)
                rec["divergent"] = bool((m.get("nonfinite") or {}).get("divergent"))
        out.append(rec)
    out.sort(key=lambda r: (r["block"], r["dataset_id"] or "", r["config_id"] or ""))
    return out


def ctx_summary(A):
    """aggregate.json['exploratory_ctx']: resumen legible de los jobs T7 con su referencia descriptiva de T1-A."""
    items = []
    ctx_jobs = getattr(A, "ctx_jobs", [])
    for rec in ctx_jobs:
        it = {k: rec[k] for k in ("job_id", "block", "config_id", "config_name", "dataset", "dataset_id", "variant",
                                  "ar", "seed", "state", "metrics_state", "reference_block", "tag")}
        it["done_time"] = rec.get("done_time")
        it["divergent"] = rec.get("divergent")
        f = rec.get("flat") or {}
        for k in ("auroc_FUS", "auprc_FUS", "f1_FUS", "auc_loc_RAW", "auroc_KMD_f", "auroc_MVD_f", "E_auc_t",
                  "E_prevalence"):
            it[k] = f.get(k)
        did = rec["dataset_id"]
        cell = ((A.cells.get(did) or {}).get("C9") or {}) if did else {}
        ref_mean = (cell.get("mean") or {}).get("auroc_FUS")
        ref_seed = (cell.get("auroc_FUS_by_seed") or {}).get("A")
        rowcfg = "mlp-AR" if rec["ar"] else "mlp-TSR"
        ref_rep, rep_rec = (None, None)
        if did and getattr(A, "idx", None) is not None:
            _st, rep_rec, _why = cell_status(A.idx, did, rowcfg, "A")
            ref_rep = ((rep_rec or {}).get("flat_conf") or {}).get("auroc_FUS")
        sp = ((A.datasets.get(did) or {}).get("sd_pool") or {}) if did else {}
        noise = sp.get("threshold")
        dmlp = ((A.mlp_noise.get(did) or {}).get("A") or {}).get("delta_AR_minus_TSR") if did else None
        v = it["auroc_FUS"]
        it["reference_config"] = "C9 (MLP publicado, sin contexto)"
        it["reference_rowcfg"] = rowcfg
        it["reference_auroc_FUS_seedA"] = ref_seed
        it["reference_auroc_FUS_replica_seedA"] = ref_rep
        it["reference_auroc_FUS_mean"] = ref_mean
        it["delta_auroc_FUS_vs_C9_seedA"] = ((v - ref_seed) if (v is not None and ref_seed is not None) else None)
        it["delta_auroc_FUS_vs_C9_replica_seedA"] = ((v - ref_rep) if (v is not None and ref_rep is not None) else None)
        it["delta_auroc_FUS_vs_C9_mean"] = ((v - ref_mean) if (v is not None and ref_mean is not None) else None)
        it["noise_threshold"] = noise
        it["abs_delta_MLP_seedA"] = (abs(dmlp) if dmlp is not None else None)
        d0 = it["delta_auroc_FUS_vs_C9_seedA"]
        it["exceeds_noise_threshold"] = ((abs(d0) > noise) if (d0 is not None and noise is not None) else None)
        it["delta_rule"] = ("[EXPLORATORIO] descriptive delta against C9 of the same seed (A). It is not read "
                            "without noise_threshold = max(sqrt(2)*SD_pool, ROPE) or without abs_delta_MLP_seedA "
                            "(|AR-TSR| of the two replicas of C9 on seed A): a delta below either of the two fits "
                            "within the variability between seeds or between replicas. No test, no confidence "
                            "interval and no adjustment for multiplicity")
        items.append(it)
    by_arm = {}
    for it in items:
        by_arm.setdefault(it["config_id"] or "?", []).append(it["dataset_id"])
    inventory = {"n_arms": len(by_arm), "n_comparisons_vs_C9": len(items),
                 "by_arm": {k: {"n_datasets": len(v), "datasets": sorted(x for x in v if x)}
                            for k, v in sorted(by_arm.items())},
                 "adjusted": False,
                 "rule": "§11/§12.1: each comparison of C12-C15 against C9 is a descriptive exploratory analysis. "
                         "All the configurations and all the datasets of the manifest are published (a configuration "
                         "without a value is listed with a state other than DONE; it is not dropped). No test, no "
                         "confidence interval and no adjustment for multiplicity. This count is reported beside any "
                         "figure of C12-C15."}
    return {"decision": "MLP con ventana", "tag": TAGS["expl"],
            "scope": ("variantes de MLP CON ventana de contexto (TEX:444, Tabla 3) que el codigo de 2025 nunca tuvo. "
                      "Fuera del plan congelado: no entran en S*_d, ni en H1-H4, ni en R1.4, ni en el abstract, ni en "
                      "SD_pool, ni en las ablaciones §7-§8. La comparacion con C9 es descriptiva."),
            "disclosure_required": (
                "El tramo de entrenamiento benigno se selecciona usando las etiquetas de verdad-terreno; NO es un "
                "warm-up sin etiquetas. Aplica igual que a T1-T4 (pipeline 'corrected'; en CIC/ACI se eliminan "
                "filas de ataque por etiqueta del intervalo de entrenamiento). Ninguna cifra C12-C15 puede "
                "presentarse como resultado zero-day operativo ni describirse como 'sin etiquetas' sin esta frase "
                "al lado."),
            "headline_rule": ("No pre-registered rule says which of C12-C15 is the figure of the manuscript: the "
                              "four configurations are reported on all the datasets and the best one is not chosen "
                              "a posteriori."),
            "exploratory_inventory": inventory,
            "configs": CTX_CONFIG_NAMES, "n_jobs": len(items),
            "by_state": _count(r["state"] for r in ctx_jobs), "jobs": items}


def _ts(t):
    """ISO -> datetime con zona (las ingenuas se leen como hora local), para comparar con la hora del tag."""
    import datetime as dt
    d = dt.datetime.fromisoformat(str(t))
    return d if d.tzinfo else d.astimezone()


def t1_job_index(A):
    """(dataset_id, rowcfg, seed) -> rec de la rejilla T1."""
    idx = {}
    for rec in A.jobs.values():
        if rec["family"] != "T1":
            continue
        idx[(rec["dataset_id"], rec["rowcfg"], rec["seed"])] = rec
    return idx


def cell_status(idx, did, rowcfg, seed):
    """Status of a (dataset, configuration, seed) cell, with its record and the reason when it has no value."""
    s = "A" if rowcfg in DETERMINISTIC_ROWCFGS else seed
    rec = idx.get((did, rowcfg, s))
    if rec is None:
        return "PENDING", None, "sin fila en el manifiesto (no ejecutada)"
    st = rec["state"]
    if st == "DONE":
        if rec["metrics_state"] == "ok":
            return "OK", rec, None
        return "PENDING", rec, f"DONE sin metrics.json valido: {rec['metrics_state']}"
    if st in ("LOST_NONEXEC", "LOST_BLOCKED"):
        return "LOST", rec, st
    if st == "DONE_POST_FREEZE":
        return "PENDING", rec, "DONE despues de la congelacion: no habia terminado en ella (AG-11)"
    if st == "LADDER":
        return "PENDING", rec, "eliminada por la escalera en T1 (semilla no ejecutada; acorta S*_d, AG-4)"
    return "PENDING", rec, st


def s_star(idx, did):
    """Longest prefix of the seeds (A, B, C) in which every cell that is not lost is complete (plan §4.5)."""
    for k in (3, 2, 1, 0):
        P = SEED_ORDER[:k]
        lost = {}
        for rc in ROWCFGS:
            for s in P:
                st, _rec, why = cell_status(idx, did, rc, s)
                if st == "LOST":
                    lost[rc] = f"{why} en la semilla {s}"
                    break
        for a, b in (("mlp-AR", "mlp-TSR"), ("mlp-TSR", "mlp-AR")):
            if a in lost and b not in lost:
                lost[b] = f"replica hermana de {a} (C9 perdida, AG-1; ST-m5)"
        ok = all(cell_status(idx, did, rc, s)[0] == "OK" for rc in ROWCFGS if rc not in lost for s in P)
        if ok:
            return list(P), lost
    return [], {}


def config_lost(cfg, lost_rowcfgs):
    reasons = [f"{rc}: {lost_rowcfgs[rc]}" for rc in CONFIG_ROWCFGS[cfg] if rc in lost_rowcfgs]
    return ("; ".join(reasons) if cfg != "C9" else "AG-1 replica perdida: " + "; ".join(reasons)) if reasons else None


def seed_values(idx, did, cfg, seed, which="flat_conf"):
    """Values of a configuration in one seed (C9 is the key-by-key mean of its replicas)."""
    recs = [cell_status(idx, did, rc, seed)[1] for rc in CONFIG_ROWCFGS[cfg]]
    if any(r is None or r.get(which) is None for r in recs):
        return None, recs
    if len(recs) == 1:
        return dict(recs[0][which]), recs
    keys = set(recs[0][which]) | set(recs[1][which])
    out = {}
    for k in keys:
        a, b = recs[0][which].get(k), recs[1][which].get(k)
        out[k] = None if (a is None or b is None) else float((a + b) / 2.0)
    return out, recs


def cell_mean(cell, P, which, k):
    """Mean of key k over the seeds P; for original and stat, the value of their single run."""
    src = cell[which]
    if cell["deterministic"]:
        return (src.get(P[0]) or {}).get(k) if P else None
    return mean_or_none([(src.get(s) or {}).get(k) for s in P])


def aggregate(R, manifest_rows, plan_sha, ladder, freeze=None):
    A = load_all(R, manifest_rows, plan_sha, ladder, freeze)
    idx = t1_job_index(A)
    datasets, cells, mlp_noise = {}, {}, {}
    for did, dname in DATASETS:
        S, lost_rc = s_star(idx, did)
        status = {rc: {s: cell_status(idx, did, rc, s)[0] for s in SEED_ORDER} for rc in ROWCFGS}
        pend = [{"rowcfg": rc, "seed": s, "why": cell_status(idx, did, rc, s)[2]}
                for rc in ROWCFGS for s in SEED_ORDER if status[rc][s] == "PENDING"]
        datasets[did] = {"dataset": dname, "corpus": CORPUS[did], "S_star": S, "n_seeds": len(S),
                         "status": status, "lost_rowcfgs": lost_rc, "pending_or_incomplete": pend}
        cells[did] = {}
        for cfg in CONFIGS:
            lr = config_lost(cfg, lost_rc)
            cell = {"config": cfg, "name": CONFIG_NAMES[cfg], "lost": lr is not None, "lost_reason": lr,
                    "deterministic": cfg in ("C10", "C11"), "seeds": {}, "seeds_raw": {}, "job_ids": {},
                    "divergent_seeds": [], "mean": {}, "mean_raw": {}}
            if lr is None and S:
                for s in S:
                    v, recs = seed_values(idx, did, cfg, s, "flat_conf")
                    vr, _ = seed_values(idx, did, cfg, s, "flat")
                    cell["seeds"][s] = v
                    cell["seeds_raw"][s] = {k: x for k, x in vr.items() if k in RAW_KEYS or k.startswith("E_")}
                    cell["job_ids"][s] = [r["job_id"] for r in recs]
                    if any(r.get("divergent") for r in recs):
                        cell["divergent_seeds"].append(s)
                keys = set().union(*[set(v) for v in cell["seeds"].values()])
                for k in sorted(keys):
                    cell["mean"][k] = cell_mean(cell, S, "seeds", k)
                for k in sorted(set().union(*[set(v) for v in cell["seeds_raw"].values()])):
                    cell["mean_raw"][k] = cell_mean(cell, S, "seeds_raw", k)
                vals = [cell["seeds"][s].get("auroc_FUS") for s in S]
                cell["auroc_FUS_by_seed"] = {s: cell["seeds"][s].get("auroc_FUS") for s in S}
                vn = [cell["seeds_raw"][s].get("auroc_FUS") for s in S if s not in cell["divergent_seeds"]]
                okn = bool(vn) and all(v is not None for v in vn)
                cell["auroc_FUS_sd_ddof1"] = sd_ddof1(vn) if (okn and not cell["deterministic"]) else None
                cell["auroc_FUS_min"] = min(vn) if okn else None
                cell["auroc_FUS_max"] = max(vn) if okn else None
                cell["auroc_FUS_n_seeds_noimp"] = len(vn)
                cell["auroc_FUS_sd_ddof1_itt"] = sd_ddof1(vals) if not cell["deterministic"] else None
                cell["auroc_FUS_min_itt"] = min(vals) if all(v is not None for v in vals) else None
                cell["auroc_FUS_max_itt"] = max(vals) if all(v is not None for v in vals) else None
                if cfg == "C9":
                    mlp_noise.setdefault(did, {})
                    for s in S:
                        recs = [cell_status(idx, did, rc, s)[1] for rc in ("mlp-AR", "mlp-TSR")]
                        a, b = (r["flat"].get("auroc_FUS") for r in recs)
                        mlp_noise[did][s] = {"delta_AR_minus_TSR": (a - b) if (a is not None and b is not None) else None,
                                             "divergent": any(r.get("divergent") for r in recs),
                                             "tag": "[SECUNDARIA plan §3.2 referencia de ruido entre replicas]"}
            cells[did][cfg] = cell
        datasets[did]["trivial"] = _trivial(cells[did])
        datasets[did]["sd_pool"] = sd_pool(cells[did], S)
    value_sets = build_value_sets(idx, datasets, cells)
    abl = ablations(A, idx, datasets)
    A.datasets, A.cells, A.mlp_noise, A.value_sets, A.ablations = datasets, cells, mlp_noise, value_sets, abl
    A.idx = idx
    A.ctx_jobs = load_ctx_jobs(R, manifest_rows, plan_sha, ladder, freeze)
    if A.metrics_rev_sha256:
        odd = sorted({r["job_id"] for r in A.ctx_jobs
                      if r.get("metrics_rev_sha256") and r["metrics_rev_sha256"] not in A.metrics_rev_sha256})
        if odd:
            A.warnings.append("AG-15: metrics.json de jobs T7 ([EXPLORATORIO]) calculados con "
                              "una version de metrics_rev.py distinta de la de los "
                              "confirmatorios: " + ", ".join(odd[:8]) +
                              (f" (+{len(odd) - 8})" if len(odd) > 8 else ""))
    return A


def _trivial(dcells):
    for cfg in CONFIGS:
        m = dcells[cfg]["mean_raw"]
        if m.get("E_prevalence") is not None:
            return {k: m.get(f"E_{k}") for k in ("auc_t", "prevalence", "f1_trivial", "ap_trivial", "n", "n_attack",
                                                   "duration_hours")}
    return None


def sd_pool(dcells, S):
    """Noise reference of the ablations (plan §7.1)."""
    out = {"n_seeds": len(S), "configs_used": [], "tag": "[SENSIBILIDAD plan §7.1 referencia de ruido]"}
    if len(S) < 2:
        out.update(sd_pool=None, df=0, sqrt2_sd_pool=None, threshold=ROPE_NOISE_FLOOR,
                   note="no between-seed noise estimate (|S*_d| < 2): umbral 0,01")
        return out
    variances, dfs, var_itt = [], [], []
    for cfg in NONDET_CONFIGS:
        c = dcells[cfg]
        if c["lost"]:
            continue
        vals = [c["seeds_raw"][s].get("auroc_FUS") for s in S if s not in c["divergent_seeds"]]
        vi = [c["seeds"][s].get("auroc_FUS") for s in S]
        if all(v is not None for v in vi):
            var_itt.append(sd_ddof1(vi) ** 2)
        if len(vals) < 2 or any(v is None for v in vals):
            continue
        variances.append(sd_ddof1(vals) ** 2)
        dfs.append(len(vals) - 1)
        out["configs_used"].append(cfg)
    out["sd_pool_itt"] = math.sqrt(sum(var_itt) / len(var_itt)) if var_itt else None
    if not variances:
        out.update(sd_pool=None, df=0, sqrt2_sd_pool=None, threshold=ROPE_NOISE_FLOOR,
                   note="sin configuraciones no deterministas disponibles: umbral 0,01")
        return out
    sp = math.sqrt(sum(variances) / len(variances))
    out.update(sd_pool=sp, df=int(sum(dfs)), sqrt2_sd_pool=math.sqrt(2.0) * sp,
               threshold=max(math.sqrt(2.0) * sp, ROPE_NOISE_FLOOR),
               rule="max(sqrt(2) SD_pool(d), 0,01); SD_pool^2 = media de varianzas ddof 1 sobre S*_d (C1-C9 no perdidas; "
                    "valores sin imputar, semillas divergidas fuera, AG-9)")
    return out


def build_value_sets(idx, datasets, cells):
    """Conjuntos de valores por (dataset, configuracion) para los tests (CONF_KEYS): main (§4, §12.2), s_itt (lost ->
    ROC-AUC 0,5), s_div_noimp (divergidas sin imputar -> eliminadas), s_seed (prefijo comun min |S*_d|)."""
    vs = {"main": {}, "s_itt": {}, "s_div_noimp": {}, "s_seed": {}}
    Ls = [len(datasets[d]["S_star"]) for d, _ in DATASETS]
    L = min(Ls) if Ls else 0
    for did, _ in DATASETS:
        for k in vs:
            vs[k][did] = {}
        S = datasets[did]["S_star"]
        for cfg in CONFIGS:
            c = cells[did][cfg]
            main = {k: (c["mean"].get(k) if (not c["lost"] and S) else None) for k in CONF_KEYS}
            vs["main"][did][cfg] = main
            itt = dict(main)
            if c["lost"]:
                for k in AUROC_CONF_KEYS:
                    itt[k] = 0.5
            vs["s_itt"][did][cfg] = itt
            vs["s_div_noimp"][did][cfg] = ({k: None for k in CONF_KEYS} if c["divergent_seeds"] else
                                          {k: (c["mean_raw"].get(k) if (not c["lost"] and S) else None) for k in CONF_KEYS})
            if L >= 1 and not c["lost"] and S:
                P = S[:L]
                vs["s_seed"][did][cfg] = {k: cell_mean(c, P, "seeds", k) for k in CONF_KEYS}
            else:
                vs["s_seed"][did][cfg] = {k: None for k in CONF_KEYS}
    vs["_meta"] = {"s_seed_prefix_len": L, "s_seed_computable": L >= 1,
                   "rules": {"main": "§4.5 + §12.2 (divergidas imputadas ITT, AG-5)",
                             "s_itt": "§12.2 S-ITT: celdas perdidas con ROC-AUC = 0,5 (AG-6)",
                             "s_div_noimp": "§12.2: divergidas sin imputar = eliminadas por comparacion",
                             "s_seed": "§4.5 S-SEED: prefijo comun min_d |S*_d| (AG-7)"}}
    return vs


def ablations(A, idx, datasets):
    """Cells of blocks T2-T4 with their T1 reference and the noise threshold (plan §7 and §8.5)."""
    out = []
    for rec in A.jobs.values():
        if rec["family"] not in ("T2", "T3", "T4"):
            continue
        row = rec["row"]
        rule = C.ABLATION_BLOCKS.get(row["block"]) or {}
        ref_block = C.ablation_reference_block(row["block"], row["dataset"])
        did = rec["dataset_id"]
        ref_seed = rule.get("seed", row.get("seed"))
        ref = idx.get((did, rec["rowcfg"], "A" if rec["rowcfg"] in DETERMINISTIC_ROWCFGS else ref_seed))
        lost = rec["state"] in ("LOST_NONEXEC", "LOST_BLOCKED", "LADDER")
        post = rec["state"] == "DONE_POST_FREEZE" or (ref is not None and ref["state"] == "DONE_POST_FREEZE")
        factor = rule.get("factor")
        value = {"window": row.get("window"), "k": row.get("k_clusters"),
                 "grouping": f"{row.get('grouping')}-p{row.get('partition_seed')}",
                 "contamination": row.get("contamination")}.get(factor)
        item = {"job_id": rec["job_id"], "block": row["block"], "plan": rule.get("plan"), "factor": factor,
                "value": value, "partition_seed": row.get("partition_seed"), "dataset_id": did, "dataset": row["dataset"],
                "rowcfg": rec["rowcfg"], "variant": row["variant"], "ar": C.parse_bool(row["ar"]),
                "seed": row.get("seed"), "state": rec["state"], "metrics_state": rec["metrics_state"],
                "lost": lost, "deterministic": rec["rowcfg"] in DETERMINISTIC_ROWCFGS,
                "reference_block": ref_block, "reference_job_id": ref["job_id"] if ref else None,
                "reference_state": (ref["state"] if ref else None),
                "reference_metrics_state": (ref["metrics_state"] if ref else None),
                "tag": TAGS["post_freeze"] if post else (TAGS["t4"] if rec["family"] == "T4" else TAGS["abl"]),
                "post_freeze": post}
        if rec["rowcfg"].startswith("mlp-"):
            ra, rt = (idx.get((did, r_, "A")) for r_ in ("mlp-AR", "mlp-TSR"))
            fa, ft = ((r_ or {}).get("flat") or {} for r_ in (ra, rt))
            if fa.get("auroc_FUS") is not None and ft.get("auroc_FUS") is not None:
                item["abs_delta_MLP_seedA"] = abs(fa["auroc_FUS"] - ft["auroc_FUS"])
            item["mlp_single_replica_note"] = ("una sola replica MLP; el umbral usa SD_pool con C9 (media de 2 "
                                               "replicas), que infraestima su ruido (AG-9, declarado)")
        vals = {}
        if rec.get("flat_conf") is not None and ref is not None and ref.get("flat_conf") is not None:
            ab, rf = rec["flat_conf"], ref["flat_conf"]
            if rec["family"] == "T4":
                pairs = {"auroc": (ab.get("auroc_FUS"), rf.get("t4eref_auroc_FUS")),
                         "auprc": (ab.get("auprc_FUS"), rf.get("t4eref_auprc_FUS")),
                         "f1": (ab.get("f1_FUS"), rf.get("t4eref_f1_FUS"))}
                item["reference_field"] = "t4e_reference.FUS (E sin el pool con guardas, §8.5)"
            else:
                pairs = {"auroc": (ab.get("auroc_FUS"), rf.get("auroc_FUS")),
                         "auprc": (ab.get("auprc_FUS"), rf.get("auprc_FUS")),
                         "f1": (ab.get("f1_FUS"), rf.get("f1_FUS"))}
                item["reference_field"] = "primary.value / E.FUS"
            for k, (a, b) in pairs.items():
                vals[f"{k}_ablation"] = a
                vals[f"{k}_reference"] = b
                vals[f"delta_{k}"] = (a - b) if (a is not None and b is not None) else None
        item.update(vals)
        sp = (datasets.get(did) or {}).get("sd_pool") or {}
        if item["deterministic"]:
            thr, src = ROPE_NOISE_FLOOR, "0,01 (configuracion determinista, §7.1)"
        elif sp.get("sd_pool") is None:
            thr, src = ROPE_NOISE_FLOOR, "0,01: no between-seed noise estimate (§7.1)"
        else:
            thr, src = sp["threshold"], f"max(sqrt(2) SD_pool(d), 0,01); SD_pool con gl = {sp['df']}"
        item["noise_threshold"] = thr
        item["noise_threshold_source"] = src
        d = item.get("delta_auroc")
        item["within_noise"] = (abs(d) <= thr) if d is not None else None
        out.append(item)
    out.sort(key=lambda r: (r["block"], r["dataset_id"] or "", r["rowcfg"], str(r["value"]), str(r["seed"])))
    return out


RUNS_HEADER = ["job_id", "block", "family", "dataset", "dataset_id", "corpus", "config_id", "variant", "ar",
               "mlp_replica", "seed", "rep", "window", "k_clusters", "grouping", "partition_seed", "contamination",
               "pipeline_version", "state", "metrics_state", "divergent", "n_attempts_preserved", "use", "in_S_star",
               "analysis", "candidate", "series", "level", "primary_value", "auroc", "auprc", "apn", "eer", "threshold",
               "f1", "precision", "recall", "fpr", "alarm_rate", "f1n", "f1_le_trivial_plus_0.02", "wd", "wd_k",
               "auc_loc", "auc_t", "prevalence", "n_eval", "effective_contamination", "o2_f1_max", "o2_gap",
               "detection_rate", "n_episodes", "median_delay_rows", "false_alarm_episodes_per_1e6_benign", "tag",
               "metrics_sha256", "done_time", "attempt"]


def job_use(rec, A):
    fam = rec["family"]
    st = rec["state"]
    if st in ("LOST_NONEXEC", "LOST_BLOCKED"):
        return "celda perdida (§12.2)", False
    if st == "LADDER":
        return ("no ejecutada por la escalera (acorta S*_d, AG-4)" if fam == "T1"
                else "celda eliminada por la escalera (perdida, §12.2)"), False
    if st == "DONE_POST_FREEZE":
        return TAGS["post_freeze"] + " no cuenta para S*_d ni para los tests", False
    if st != "DONE":
        return "not run", False
    if rec["metrics_state"] != "ok":
        return "DONE sin metrics.json valido (AG-3)", False
    if fam == "T1":
        did = rec["dataset_id"]
        S = A.datasets[did]["S_star"]
        seed = "A" if rec["rowcfg"] in DETERMINISTIC_ROWCFGS else rec["seed"]
        cfg = ROWCFG_TO_CONFIG[rec["rowcfg"]]
        if A.cells[did][cfg]["lost"]:
            return "celda perdida en S*_d (se reporta aparte, §4.5)", False
        if seed in S:
            return "confirmatorio (S*_d)", True
        return "fuera de S*_d (se reporta aparte, §4.6)", False
    return {"T2": "ablacion §7.2 (descriptiva)", "T3": "ablacion §7.3 (descriptiva)",
            "T4": "T4-E §8.5 (descriptiva)"}[fam], False


def _mb_row(base, analysis, series, b, tag, **extra):
    r = dict(base, analysis=analysis, series=series, tag=tag)
    b = b or {}
    for k in ("auroc", "auprc", "apn", "eer", "threshold", "f1", "precision", "recall", "fpr", "alarm_rate", "f1n"):
        r[k] = fnum(b.get(k))
    if b.get("f1_le_trivial_plus_0.02") is not None:
        r["f1_le_trivial_plus_0.02"] = bool(b["f1_le_trivial_plus_0.02"])
    wd = b.get("windowdiff") or {}
    r["wd"], r["wd_k"] = fnum(wd.get("wd")), fnum(wd.get("k"))
    r.update(extra)
    return r


def runs_rows(A):
    rows = []
    for jid in sorted(A.jobs):
        rec = A.jobs[jid]
        row = rec["row"]
        use, in_s = job_use(rec, A)
        base = {"job_id": jid, "block": row["block"], "family": rec["family"], "dataset": row["dataset"],
                "dataset_id": rec["dataset_id"], "corpus": CORPUS.get(rec["dataset_id"]),
                "config_id": ROWCFG_TO_CONFIG.get(rec["rowcfg"]), "variant": row["variant"],
                "ar": C.parse_bool(row["ar"]), "mlp_replica": (("AR" if C.parse_bool(row["ar"]) else "TSR")
                                                               if row["variant"] == "mlp" else None),
                "seed": row.get("seed"), "rep": row.get("rep"), "window": row.get("window"),
                "k_clusters": row.get("k_clusters"), "grouping": row.get("grouping"),
                "partition_seed": row.get("partition_seed"), "contamination": row.get("contamination"),
                "pipeline_version": row.get("pipeline_version"), "state": rec["state"],
                "metrics_state": rec["metrics_state"], "divergent": rec.get("divergent"),
                "n_attempts_preserved": rec["n_attempts_preserved"], "use": use, "in_S_star": in_s,
                "metrics_sha256": rec["sha256"], "done_time": rec.get("done_time")}
        for att in rec["attempts_preserved"]:
            rows.append(dict(base, analysis="attempt", attempt=att, use="intento conservado (no cuenta, §12.2)",
                             in_S_star=False, metrics_sha256=None, tag="[listado §4.6]"))
        m = rec.get("m")
        if m is None:
            rows.append(dict(base, analysis="none"))
            continue
        E = m["E"]
        triv = E["trivial"]
        cand = (m.get("regions") or {}).get("candidate")
        sloc = (m.get("sensitivities") or {}).get("S-LOC") or {}
        common = dict(base, candidate=cand, auc_t=fnum(triv.get("auc_t")), prevalence=fnum(triv.get("prevalence")),
                      n_eval=fnum(triv.get("n")))
        for X in X4:
            r = _mb_row(common, "main", X, E.get(X), TAGS["conf"] if X == "FUS" else TAGS["sec"],
                        auc_loc=fnum((sloc.get(X) or {}).get("auc_loc")))
            if X == "FUS":
                r["primary_value"] = fnum(m["primary"]["value"])
            if X == "RAW":
                r["tag"] = "[SENSIBILIDAD plan §2.9 S-RAW]"
            if rec["state"] == "DONE_POST_FREEZE":
                r["tag"] = TAGS["post_freeze"]
            rows.append(r)
        for X in ("FUS", "KMD_f", "MVD_f"):
            o2 = (E.get(X) or {}).get("o2")
            if o2:
                rows.append(dict(common, analysis="O2", series=X, threshold=fnum(o2.get("threshold")),
                                 precision=fnum(o2.get("precision")), recall=fnum(o2.get("recall")),
                                 o2_f1_max=fnum(o2.get("f1_max")), o2_gap=fnum(o2.get("gap_vs_calibrated_f1")),
                                 tag=TAGS["o2"]))
        if m.get("divergence"):
            it = m["divergence"].get("imputed_itt") or {}
            rows.append(dict(common, analysis="ITT-divergence", series="FUS", auroc=fnum(it.get("auroc_fus")),
                             auprc=fnum(it.get("auprc_fus")), f1=fnum(it.get("f1_fus")),
                             auc_loc=fnum(it.get("auc_loc_raw")), tag="[CONFIRMATORIO plan §12.2 imputacion ITT]"))
        S = m.get("sensitivities") or {}
        for name, xs in (("S-NF", ("FUS_NF", "KMD_raw", "MVD_raw")), ("S-MAX", ("FUS_MAX",)),
                         ("S-Q95", X4), ("S-Q995", X4), ("S-PS", X4), ("S-DUP", X4), ("S-INS", X4),
                         ("S-DUPCAL", X4)):
            blk = S.get(name) or {}
            if blk.get("applicable") is False or blk.get("error"):
                rows.append(dict(common, analysis=name, series=None,
                                 tag=(TAGS["dupcal"] if name == "S-DUPCAL" else TAGS["sens"]) + " no aplicable/NA"))
                continue
            cand_s = ((blk.get("regions") or {}).get("candidate") if name == "S-INS" else cand)
            for X in xs:
                if X in blk:
                    rows.append(_mb_row(dict(common, candidate=cand_s), name, X, blk[X], blk.get("tag") or TAGS["sens"]))
        for X in X4:
            b = (S.get("S-EP") or {}).get(X) or {}
            rows.append(dict(common, analysis="S-EP", series=X, wd=fnum(b.get("wd")), wd_k=fnum(b.get("k")),
                             tag=TAGS["sens"]))
        for g, tg in (("G10000", TAGS["sec"]), ("G800", TAGS["sens"] + " S-G800")):
            b = (m.get("operational") or {}).get(g) or {}
            rows.append(dict(common, analysis=f"operational-{g}", series="FUS", detection_rate=fnum(b.get("detection_rate")),
                             n_episodes=fnum(b.get("n_episodes")), median_delay_rows=fnum(b.get("median_delay_rows")),
                             **{"false_alarm_episodes_per_1e6_benign": fnum(b.get("false_alarm_episodes_per_1e6_benign"))},
                             tag=tg))
        t4p = m.get("t4p") or {}
        a = t4p.get("a") or {}
        if a.get("applicable") and a.get("estimable"):
            for arm in ("natural", "clean"):
                for X in ("FUS", "KMD_f", "MVD_f"):
                    rows.append(_mb_row(common, f"T4P-a-{arm}", X, (a.get(arm) or {}).get(X), TAGS["t4"] + " T4-P-a",
                                        effective_contamination=fnum(a.get("natural_calib_contamination"))
                                        if arm == "natural" else 0.0))
        elif a.get("applicable"):
            rows.append(dict(common, analysis="T4P-a", series=None, tag=TAGS["t4"] + " T4-P-a no estimable -> S-INS"))
        b = t4p.get("b") or {}
        if b.get("applicable"):
            for lv in b.get("levels") or []:
                for X in ("FUS", "KMD_f", "MVD_f"):
                    rows.append(_mb_row(dict(common, n_eval=fnum(b.get("E_T4_rows")),
                                             prevalence=fnum((b.get("trivial_E_T4") or {}).get("prevalence")),
                                             auc_t=fnum((b.get("trivial_E_T4") or {}).get("auc_t"))),
                                        "T4P-b", X, (lv.get("metrics") or {}).get(X), TAGS["t4"] + " T4-P-b",
                                        level=_lvl(lv["c"]), effective_contamination=fnum(lv.get("effective_contamination"))))
        cc = t4p.get("c") or {}
        for f, blk in sorted((cc.get("fractions") or {}).items(), key=lambda kv: int(kv[0])):
            tc = cc.get("trivial_common_eval") or {}
            for X in ("FUS", "KMD_f", "MVD_f"):
                rows.append(_mb_row(dict(common, candidate=blk.get("candidate"), n_eval=fnum(tc.get("n")),
                                         prevalence=fnum(tc.get("prevalence")), auc_t=fnum(tc.get("auc_t"))),
                                    "T4P-c", X, (blk.get("metrics") or {}).get(X), TAGS["t4"] + " T4-P-c", level=f,
                                    effective_contamination=fnum(blk.get("effective_contamination"))))
        ref = m.get("t4e_reference")
        if ref and not ref.get("error"):
            for X in X4:
                rows.append(_mb_row(dict(common, n_eval=fnum((ref.get("trivial") or {}).get("n")),
                                         prevalence=fnum((ref.get("trivial") or {}).get("prevalence")),
                                         auc_t=fnum((ref.get("trivial") or {}).get("auc_t"))),
                                    "T4E-reference-c0", X, ref.get(X), TAGS["t4"] + " T4-E c = 0"))
        o1 = rec.get("o1")
        if o1:
            for det in ("MVD", "KMD"):
                d = o1.get(det) or {}
                rows.append(dict(base, analysis="O1", series=det, f1=fnum(d.get("F1-Score Smooth")),
                                 wd=fnum(d.get("Window Diff Smooth")), precision=fnum(d.get("Precision Smooth")),
                                 recall=fnum(d.get("Recall Smooth")), threshold=fnum(d.get("Threshold Smooth")),
                                 tag=TAGS["o1"]))
            sel = o1.get("selected") or {}
            rows.append(dict(base, analysis="O1", series=f"selected:{sel.get('detector')}",
                             f1=fnum(sel.get("F1-Score Smooth")), wd=fnum(sel.get("Window Diff Smooth")),
                             tag=TAGS["o1"], metrics_sha256=rec.get("o1_sha256")))
    for extra in A.not_in_manifest:
        rows.append({"job_id": extra["job_id"], "block": extra["job_id"].split("__")[0],
                     "family": C.block_family(extra["job_id"].split("__")[0]),
                     "state": "DONE" if extra["done"] else "sin DONE", "use": extra["use"], "analysis": "none"})
    rows += ctx_runs_rows(A)
    return rows


def ctx_runs_rows(A):
    """Rows of runs.csv for the jobs of blocks T7 and T8, always labelled [EXPLORATORIO] and outside S*_d."""
    rows = []
    use = TAGS["expl"] + (" windowed MLP and retraining experiment: descriptive, outside S*_d and H1-H4")
    for rec in getattr(A, "ctx_jobs", []):
        row = rec["row"]
        base = {"job_id": rec["job_id"], "block": rec["block"], "family": rec["family"], "dataset": rec["dataset"],
                "dataset_id": rec["dataset_id"], "corpus": rec["corpus"], "config_id": rec["config_id"],
                "variant": rec["variant"], "ar": rec["ar"], "mlp_replica": None, "seed": rec["seed"],
                "rep": row.get("rep"), "window": row.get("window"), "k_clusters": row.get("k_clusters"),
                "grouping": row.get("grouping"), "partition_seed": row.get("partition_seed"),
                "contamination": row.get("contamination"), "pipeline_version": row.get("pipeline_version"),
                "state": rec["state"], "metrics_state": rec["metrics_state"], "divergent": rec.get("divergent"),
                "n_attempts_preserved": rec["n_attempts_preserved"], "use": use, "in_S_star": False,
                "metrics_sha256": rec["sha256"], "done_time": rec.get("done_time")}
        for att in rec["attempts_preserved"]:
            rows.append(dict(base, analysis="attempt", attempt=att, use="intento conservado (no cuenta, §12.2)",
                             metrics_sha256=None, tag=TAGS["expl"]))
        m = rec.get("m")
        if m is None:
            rows.append(dict(base, analysis="none", tag=TAGS["expl"]))
            continue
        E = m["E"]
        triv = E["trivial"]
        sloc = (m.get("sensitivities") or {}).get("S-LOC") or {}
        common = dict(base, candidate=(m.get("regions") or {}).get("candidate"), auc_t=fnum(triv.get("auc_t")),
                      prevalence=fnum(triv.get("prevalence")), n_eval=fnum(triv.get("n")))
        for X in X4:
            r = _mb_row(common, "main", X, E.get(X), TAGS["expl"],
                        auc_loc=fnum((sloc.get(X) or {}).get("auc_loc")))
            if X == "FUS":
                r["primary_value"] = fnum(m["primary"]["value"])
            rows.append(r)
    return rows


CELLS_HEADER = ["dataset_id", "dataset", "corpus", "config_id", "config_name", "deterministic", "S_star", "n_seeds",
                "lost", "lost_reason", "auroc_FUS_A", "auroc_FUS_B", "auroc_FUS_C", "auroc_FUS_mean",
                "auroc_FUS_sd_ddof1", "auroc_FUS_min", "auroc_FUS_max", "auroc_FUS_sd_ddof1_itt", "auc_loc_RAW_mean",
                "auroc_KMD_f_mean",
                "auroc_MVD_f_mean", "auroc_RAW_mean", "auprc_FUS_mean", "f1_FUS_mean", "auroc_FUS_NF_mean",
                "auroc_KMD_raw_mean", "auroc_MVD_raw_mean", "delta_MLP_A", "delta_MLP_B", "delta_MLP_C",
                "divergent_seeds", "auc_t", "prevalence", "f1_trivial", "ap_trivial", "sd_pool", "noise_threshold",
                "tag"]


def cells_rows(A):
    rows = []
    for did, dname in DATASETS:
        D = A.datasets[did]
        for cfg in CONFIGS:
            c = A.cells[did][cfg]
            mn = c["mean"]
            r = {"dataset_id": did, "dataset": dname, "corpus": CORPUS[did], "config_id": cfg,
                 "config_name": CONFIG_NAMES[cfg], "deterministic": c["deterministic"], "S_star": "".join(D["S_star"]),
                 "n_seeds": len(D["S_star"]), "lost": c["lost"], "lost_reason": c["lost_reason"],
                 "auroc_FUS_mean": mn.get("auroc_FUS"), "auroc_FUS_sd_ddof1": c.get("auroc_FUS_sd_ddof1"),
                 "auroc_FUS_min": c.get("auroc_FUS_min"), "auroc_FUS_max": c.get("auroc_FUS_max"),
                 "auroc_FUS_sd_ddof1_itt": c.get("auroc_FUS_sd_ddof1_itt"),
                 "divergent_seeds": "".join(c["divergent_seeds"]),
                 "tag": "[CONFIRMATORIO plan §3.1/§4] (auroc_FUS); resto secundarias/sensibilidades"}
            for s in SEED_ORDER:
                r[f"auroc_FUS_{s}"] = (c.get("auroc_FUS_by_seed") or {}).get(s)
                if cfg == "C9":
                    r[f"delta_MLP_{s}"] = ((A.mlp_noise.get(did) or {}).get(s) or {}).get("delta_AR_minus_TSR")
            for k in ("auc_loc_RAW", "auroc_KMD_f", "auroc_MVD_f", "auroc_RAW", "auprc_FUS", "f1_FUS", "auroc_FUS_NF",
                      "auroc_KMD_raw", "auroc_MVD_raw"):
                r[f"{k}_mean"] = mn.get(k)
            tv = D.get("trivial") or {}
            for k in ("auc_t", "prevalence", "f1_trivial", "ap_trivial"):
                r[k] = tv.get(k)
            r["sd_pool"] = D["sd_pool"].get("sd_pool")
            r["noise_threshold"] = (D["sd_pool"].get("threshold") if not c["deterministic"] else ROPE_NOISE_FLOOR)
            rows.append(r)
    return rows


def aggregate_json(A, prov, inputs):
    jobs_summary = {jid: {"state": r["state"], "metrics_state": r["metrics_state"], "divergent": r.get("divergent"),
                          "dataset_id": r["dataset_id"], "rowcfg": r["rowcfg"], "seed": r["seed"],
                          "block": r["row"]["block"], "n_attempts_preserved": r["n_attempts_preserved"],
                          "attempts_preserved": r["attempts_preserved"]}
                    for jid, r in sorted(A.jobs.items())}
    o1 = []
    for jid, r in sorted(A.jobs.items()):
        if r.get("o1"):
            sel = r["o1"].get("selected") or {}
            o1.append({"job_id": jid, "dataset_id": r["dataset_id"], "config_id": ROWCFG_TO_CONFIG[r["rowcfg"]],
                       "rowcfg": r["rowcfg"], "seed": r["seed"], "selected": sel.get("detector"),
                       "F1-Score Smooth": fnum(sel.get("F1-Score Smooth")),
                       "Window Diff Smooth": fnum(sel.get("Window Diff Smooth")), "tag": TAGS["o1"]})
    complete = not A.consistency_errors and not A.section_error_jobs and not any(
        r["state"] == "DONE" and r["metrics_state"] != "ok" for r in A.jobs.values())
    tag_t = (prov.get("plan") or {}).get("tag_commit_time")
    first = A.first_metrics_json_created
    chrono = {"tag_commit_time": tag_t, "first_metrics_json_created": first,
              "first_metrics_json_created_by_dataset": A.first_metrics_json_created_by_dataset,
              "record": A.chronology_record,
              "tag_before_first_metrics_json": (_ts(tag_t) < _ts(first)) if (tag_t and first) else None,
              "rule": "R-7 / §14.4 hito (i): hora del commit del tag frente al primer metrics.json (registro append-only "
                      "de metrics_rev + created de los metrics.json agregados; AG-14)"}
    fz = A.freeze
    freeze = {"at": fz.isoformat(timespec="seconds") if fz else None,
              "is_plan_default": bool(fz is not None and fz == _ts(C.FREEZE.isoformat())),
              "n_done_post_freeze": sum(1 for r in A.jobs.values() if r["state"] == "DONE_POST_FREEZE"),
              "rule": "AG-11 (§4.5)"}
    return {"schema": SCHEMA_AGG, "created": C.now_iso(), "complete": complete, "chronology_R7": chrono,
            "freeze": freeze, "section_error_jobs": A.section_error_jobs,
            "consistency_errors": A.consistency_errors, "warnings": A.warnings,
            "metrics_rev_sha256": A.metrics_rev_sha256,
            "counts": {"jobs": len(A.jobs),
                       "by_state": _count(r["state"] for r in A.jobs.values()),
                       "by_metrics_state": _count((r["metrics_state"] or "-").split(":")[0] for r in A.jobs.values())},
            "datasets": A.datasets, "cells": A.cells, "value_sets": A.value_sets, "mlp_noise": A.mlp_noise,
            "ablations": A.ablations, "exploratory_ctx": ctx_summary(A),
            "o1": o1, "not_in_manifest": A.not_in_manifest, "jobs": jobs_summary,
            "rules": {"S_star": "§4.5 (AG-2)", "C9": "media de replicas (AG-1)", "C10_C11": "una ejecucion (semilla A)",
                      "lost": "§12.2", "divergent": "AG-5", "ladder": "AG-4"},
            "deviations_proposed": AG_DEVIATIONS_PROPOSED, "provenance": dict(prov, inputs=inputs)}


def _count(it):
    out = {}
    for v in it:
        out[v] = out.get(v, 0) + 1
    return out


def base_provenance():
    return {"aggregate_version": AGGREGATE_VERSION, "aggregate_sha256": C.sha256_file(Path(__file__).resolve()),
            "aggregate_path": str(Path(__file__).resolve()), "plan": M.plan_guard(), "versions": M.versions(),
            "env": M.env_guard(), "host": C.hostname(),
            "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=str(C.RESULTS_ROOT))
    ap.add_argument("--manifest", default=str(C.MANIFEST))
    ap.add_argument("--out-dir", default=None, help="raiz de salida (runs.csv y aggregate/); por defecto --results-root")
    ap.add_argument("--ladder-json", default=None, help="JSON {'removed_job_ids': [...], 'source': '...'}")
    ap.add_argument("--compute-missing", action="store_true",
                    help="antes de agregar, metrics_rev.py --all sobre --results-root (>5 min: desacoplado)")
    ap.add_argument("--blocks", default=None, help="(--compute-missing) filtro de bloques de metrics_rev")
    ap.add_argument("--freeze", default=C.FREEZE.isoformat(timespec="seconds"),
                    help="congelacion de resultados (AG-11; defecto common.FREEZE); 'none' la desactiva (solo pruebas)")
    a = ap.parse_args(argv)
    for pth in (a.results_root, a.manifest, a.out_dir, a.ladder_json):
        if pth is not None and not os.path.isabs(pth):
            print(f"ERROR: rutas absolutas: {pth}")
            return 2
    R = Path(a.results_root)
    out_root = Path(a.out_dir) if a.out_dir else R
    try:
        C.check_not_forbidden(out_root, "--out-dir")
    except Exception as e:
        print(f"ERROR: {e}")
        return 2
    try:
        prov = base_provenance()
    except (M.PlanGuardError, M.EnvGuardError) as e:
        print(f"ABORT: {e}")
        return 3
    if a.compute_missing:
        mr_args = ["--all", "--results-root", str(R)] + (["--blocks", a.blocks] if a.blocks else [])
        rc = M.main(mr_args)
        print(f"[{C.now_iso()}] metrics_rev --all terminado con codigo {rc}", flush=True)
        if rc == 3:
            return 3
    ladder, ladder_sha = set(), None
    if a.ladder_json:
        lj = C.read_json(a.ladder_json)
        if not isinstance(lj, dict) or not isinstance(lj.get("removed_job_ids"), list):
            print("ERROR: --ladder-json sin la lista 'removed_job_ids'")
            return 2
        ladder = set(lj["removed_job_ids"])
        ladder_sha = C.sha256_file(a.ladder_json)
    try:
        freeze = None if str(a.freeze).lower() == "none" else _ts(a.freeze)
    except ValueError:
        print(f"ERROR: --freeze {a.freeze!r} no es ISO")
        return 2
    t0 = time.time()
    rows = C.read_manifest(a.manifest)
    A = aggregate(R, rows, prov["plan"]["plan_sha256"], ladder, freeze)
    agg_dir = out_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    runs_p, cells_p, json_p = out_root / "runs.csv", agg_dir / "cells.csv", agg_dir / "aggregate.json"
    write_csv_atomic(runs_p, RUNS_HEADER, runs_rows(A))
    write_csv_atomic(cells_p, CELLS_HEADER, cells_rows(A))
    inputs = {"results_root": str(R), "manifest": a.manifest, "manifest_sha256": C.sha256_file(a.manifest),
              "ladder_json": a.ladder_json, "ladder_json_sha256": ladder_sha,
              "metrics_json_sha256": {jid: r["sha256"] for jid, r in sorted(A.jobs.items()) if r["sha256"]},
              "metrics_o1_json_sha256": {jid: r["o1_sha256"] for jid, r in sorted(A.jobs.items()) if r.get("o1_sha256")},
              "metrics_json_sha256_exploratory": {r["job_id"]: r["sha256"]
                                                  for r in sorted(getattr(A, "ctx_jobs", []) or [],
                                                                  key=lambda x: x["job_id"]) if r.get("sha256")},
              "label_counts_sha256": C.sha256_file(M.REPO / M.LABEL_COUNTS_REL)}
    outputs = {"runs.csv": C.sha256_file(runs_p), "aggregate/cells.csv": C.sha256_file(cells_p)}
    J = aggregate_json(A, prov, inputs)
    J["outputs_sha256"] = outputs
    J["seconds"] = round(time.time() - t0, 2)
    C.write_json_atomic(json_p, M._clean(J))
    outputs["aggregate/aggregate.json"] = C.sha256_file(json_p)
    C.write_text_atomic(agg_dir / "SHA256SUMS", "".join(f"{h}  {p}\n" for p, h in sorted(outputs.items())))
    n_ok = sum(1 for r in A.jobs.values() if r["metrics_state"] == "ok")
    sizes = ", ".join(f"{d}: {len(A.datasets[d]['S_star'])}" for d, _ in DATASETS)
    print(f"[{C.now_iso()}] aggregate: {len(A.jobs)} filas T1-T4 del manifiesto, {n_ok} con metrics.json valido; "
          f"|S*_d| = {{{sizes}}}; "
          f"errores de consistencia {len(A.consistency_errors)}, avisos {len(A.warnings)} -> {runs_p}, {agg_dir}",
          flush=True)
    for e in A.consistency_errors:
        print(f"  ERROR: {e}")
    return 0 if J["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
