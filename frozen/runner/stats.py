#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py - estadistica del plan congelado (revision JISAS-D-26-02300; analysis_plan.md, tag analysis-plan-frozen).

Implementa §1.2 (dependencia entre fuentes), §4 (S*_d via aggregate.py), §5 (H1-H4: Wilcoxon exacto con enumeracion
propia cruzada con scipy <= 1e-12, Holm de bajada, Hodges-Lehmann con IC exacto por inversion, rango-biserial, k/n,
ROPE, Friedman + Iman-Davenport con p por permutacion, reglas de redaccion §5.6 y de robustez §5.8), §6 (R1.4:
descriptor D y Spearman exacto por permutacion completa; exploratorios E4.x contados), §7.1 (ablaciones descriptivas
con la referencia de ruido), §8 (T4-P, descriptivo), §10 (casos de fallo), §11 (cifras del abstract, R1.10, X13),
§12 (celdas perdidas: S-ITT, divergidas sin imputar, S-SEED) y §14 (R-5, R-7, sha256, filas de numbers.md).
No redacta conclusiones: solo aplica de forma mecanica las reglas pre-registradas y etiqueta cada salida con
[CONFIRMATORIO plan §x] / [SENSIBILIDAD plan §x] / [EXPLORATORIO] / [ORACULO: ...].

Subcomandos (entorno del plan, RF-2: envs/postproc; R-7 como metrics_rev: sin tag o con plan distinto -> codigo 3):
    R=<raiz del repositorio>; PY=$R/envs/postproc/bin/python
    # 1) descriptor D (§6.1; solo features, ni etiquetas ni scores). SIEMPRE antes de cualquier Delta de H2/R1.4.
    #    Lee la cache corrected (KNAD) o relee los CSV crudos (CIC/ACI: antes de normalizar y de sustituir +-inf) y
    #    comprueba la alineacion con la cache. CIC2018 tarda minutos y ~6 GB de RAM: desacoplado.
    nohup nice -n 19 env OMP_NUM_THREADS=4 $PY $R/revision/experiments/stats.py descriptor-d [--datasets D1,D3] \
        [--with-e41] [--out ABS] > $R/logs/stats_descriptor_d.log 2>&1 &
    # 2) estadistica (lee results/revision/aggregate/aggregate.json de aggregate.py; ~1-2 min por las permutaciones)
    nice -n 19 env OMP_NUM_THREADS=4 $PY $R/revision/experiments/stats.py run [--aggregate ABS] [--descriptor ABS] \
        [--out-dir ABS] [--friedman-b 100000]
    # 3) tablas de alcanzabilidad (R-5; sin datos): stats.py attainability --out ABS
Salida: 0 OK; 1 fallo (aggregate con errores de consistencia, cruce con scipy > 1e-12, entrada invalida); 2 uso;
3 guarda R-7 o de entorno (no se escribe nada).
Salidas de run en <out-dir> (defecto results/revision/stats/): stats.json, tests.csv, numbers_proposed.md, env.json,
SHA256SUMS. Lecturas literales de huecos del plan: ST_DEVIATIONS_PROPOSED (tambien en stats.json y en
AGG_STATS.md).
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
import metrics_rev as M

SCHEMA_STATS = "jisa-stats/1"
SCHEMA_D = "jisa-descriptor-d/1"
STATS_VERSION = "1.1.0"
ALPHA = 0.05
ROPE = 0.01
N_TESTABLE = 6
HL_TAIL = Fraction(1, 40)
CROSS_TOL = 1e-12
FRIEDMAN_B = 100_000
PERM_SEED = 20260917
N_EXPL_R14 = 11
DETECT_AUC_LOC = 0.51
FAILURE_MARGIN = 0.01
DS = [f"D{i}" for i in range(1, 10)]
KNAD = DS[:6]
FLOW = ["D7", "D8", "D9"]
CORPUS = {"D1": "KNAD", "D2": "KNAD", "D3": "KNAD", "D4": "KNAD", "D5": "KNAD", "D6": "KNAD", "D7": "ACI",
          "D8": "CIC2017", "D9": "CIC2018"}
DS_NAME = {"D1": "KNAD:Active_Wiretap", "D2": "KNAD:Fuzzing", "D3": "KNAD:OS_Scan", "D4": "KNAD:SSDP_Flood",
           "D5": "KNAD:SSL_Renegotiation", "D6": "KNAD:Video_Injection", "D7": "ACI-IOT-2023", "D8": "CIC2017",
           "D9": "CIC2018"}
DS_KEY = {d: C.label_counts_key(n) for d, n in DS_NAME.items()}
CONFIGS = [f"C{i}" for i in range(1, 12)]
EXCLUDED_CONFIGS = ["C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22"]
CTX8 = [f"C{i}" for i in range(1, 9)]
SIMPLE = ["C9", "C10", "C11"]
ARCH = {"Conv1D": ("C1", "C2"), "Conv2D": ("C3", "C4"), "LSTM": ("C5", "C6"), "MHA": ("C7", "C8")}
ARCH5 = ["Conv1D", "Conv2D", "LSTM", "MHA", "MLP"]
CONTROL = "C10"
T = {"H1": "[CONFIRMATORIO plan §5.3]", "H2": "[CONFIRMATORIO plan §5.4]",
     "H2s": "[CONFIRMATORIO plan §5.4 secundaria con guardian]", "H3": "[CONFIRMATORIO plan §5.5]",
     "H4": "[CONFIRMATORIO plan §5.7]", "R14": "[CONFIRMATORIO plan §6.2]", "dep": "[SENSIBILIDAD plan §1.2]",
     "rob": "[SENSIBILIDAD plan §5.8]", "desc": "[DESCRIPTIVO plan §5.3]", "omni": "[DESCRIPTIVO plan §5.3 omnibus]",
     "expl": "[EXPLORATORIO]", "sens_itt": "[SENSIBILIDAD plan §12.2 S-ITT]",
     "sens_div": "[SENSIBILIDAD plan §12.2 divergidas sin imputar]", "sens_seed": "[SENSIBILIDAD plan §4.5 S-SEED]",
     "abl": "[SENSIBILIDAD plan §7]", "t3a": "[SENSIBILIDAD plan §7.3 T3-min-a]", "t4": "[SENSIBILIDAD plan §8]",
     "oracle_best2": "[ORACULO: best of the two detectors selected with evaluation labels (§3.6, §5.5)]",
     "oracle_o2": "[ORACULO: evaluation-region oracle F1 (O2)]", "o1": "[ORACULO: as-submitted (O1)]",
     "abs": "[CONFIRMATORIO plan §11]", "fail": "[EXPLORATORIO plan §10]", "r110": "[DESCRIPTIVO plan §11 R1.10]",
     "r110_best": "[ORACULO: selected on evaluation results (R1.10; descriptive, not an estimate of expected "
                  "performance)]",
     "r14_late": "[EXPLORATORIO: D calculado despues del primer metrics.json de su dataset, con codigo distinto o sin "
                 "cronologia verificable (§6.1, ST-M3)]",
     "interim": "[PARTIAL INPUTS]"}
T_NUM = {"abs": "[CONFIRMATORIO plan §11 — valor descriptivo, sin test]",
         "t1": "[CONFIRMATORIO plan §3.1 — valor por celda, descriptivo, sin test]",
         "rho_knad": "[CONFIRMATORIO plan §6.2 — rho_KNAD descriptivo, sin test confirmatorio]",
         "omni": "[CONFIRMATORIO plan §5.3 — omnibus descriptivo, no decide ninguna afirmacion]"}
TREND_MIN_K, TREND_N = 7, 9

ST_DEVIATIONS_PROPOSED = [
    "ST-1 (§5.1, §5.3) en una familia de Holm con m fijo, una comparacion 'not testable' (n_eff < 6) no tiene p: se "
    "ordena la ultima, no rechaza y no recibe p ajustado (equivale a p = 1; no cambia los p ajustados del resto).",
    "ST-2 (§5.1) el p que se reporta es el de scipy (exact sin empates ni ceros; PermutationMethod(n_resamples=inf) con "
    "empates o ceros); la enumeracion propia usa rangos medios x 2 en enteros y p = min(1, 2 min(P(T+ <= t), "
    "P(T+ >= t))), la misma definicion bilateral de scipy.stats.permutation_test. |diff| > 1e-12 aborta.",
    "ST-3 (§5.2) el IC de HL usa la distribucion de T+ sin empates con el n de todas las d (ceros incluidos), como "
    "attainability.py; con n <= 5 se da [A_(1), A_(M)] con su cobertura (no alcanza el 95 %).",
    "ST-4 (§5.3) Friedman con rangos medios (1 = menor ROC-AUC) y chi2 de Demsar sin correccion por empates; "
    "F_ID = (n-1) chi2 / (n(k-1) - chi2) (+inf si el denominador es 0). Permutacion: rng = default_rng(20260917), una "
    "llamada rng.permuted sobre un tensor (B, n, k) de rangos x 2 a lo largo de k; se compara la suma de cuadrados de las "
    "sumas de rangos (entera, monotona con F_ID). El chi2 con correccion de empates de scipy se registra aparte.",
    "ST-5 (§6.2) Spearman: p bilateral = 2 min(P(rho* <= rho), P(rho* >= rho)) de scipy.stats.permutation_test "
    "(pairings, n_resamples = inf), cruzado con enumeracion entera de los n! emparejamientos; P(|rho*| >= |rho|) se "
    "registra aparte (coinciden sin empates).",
    "ST-6 (§5.8, §1.2) si una regla de robustez no es evaluable (HL del subconjunto indefinido: sin datasets, o "
    "AUC_loc(RAW) ausente), la comparacion se trata como NO robusta (conservador: no va al abstract).",
    "ST-7 (§1.2) 'KNAD colapsado' = una unidad con la media de las d de los KNAD presentes en la comparacion + D7-D9.",
    "ST-8 (§6.1, E-M4) D must exist before any Delta: without descriptor_d.json (or without the D of some dataset "
    "with the 8 cells of H2), H2, R1.4 and the E4.x that use Delta are not computed and 'bloqueado' is reported. "
    "Delta_MLP (§3.2, noise between replicas) is not the Delta of §6 and is computed in aggregate. Verified order "
    "(ST-M3/F-02): R1.4 is confirmatory only if, in every dataset, the D in use was computed before the first "
    "metrics.json of that dataset (append-only chronology of aggregate, AG-14) and all the D share d_code_sha256; "
    "otherwise R1.4 is labelled [EXPLORATORIO: D calculado despues...]. descriptor-d is append-only (recomputing "
    "requires --force and is kept in history) and stores stats_sha256 and d_code_sha256 per entry. The time of D and "
    "the sha256 of the script are recorded with the numbers of the manuscript (§6.1).",
    "ST-9 (§6.1) D en CIC/ACI: features crudas releidas con pipelines.corrected_raw_features (NaN -> 0 como fillna(0) "
    "del pipeline corrected, +-inf conservados y ordenados como +-inf; categoricas excluidas), comprobando que, "
    "transformadas como el pipeline, reproducen x.npy de la cache bit a bit en la calibracion. KNAD: x.npy de la cache "
    "(features AfterImage crudas).",
    "ST-10 (§5.7, §11 X13) estado X13 por comparacion para cada signo reclamado posible; afirmacion H4 'MLP highest': "
    "mantenida si las 4 rechazan con HL > 0 y son robustas; debilitada si las 4 tienen HL > 0 sin cumplir lo anterior; "
    "retirada en otro caso.",
    "ST-11 (§10) family means over the configurations that are not lost; if none exceeds the margin, the dataset "
    "with the smallest difference (with context - simple). Analyses 2, 4 and 5 need scores.npy and are not computed "
    "by stats.py.",
    "ST-12 (§11) 'supera a F1_trivial' = F1 calibrado > F1_trivial (estricto); 'promedio de las 8 configuraciones con "
    "contexto' exige las 8 celdas C1-C8 del dataset (misma definicion que H1 descriptiva); un dataset sin alguna queda "
    "fuera y se reporta n (ST-m9).",
    "ST-13 (§7.3 T3-min-a) HL e IC exacto sobre los datasets disponibles por (configuracion, k), sin p; la MLP de T3 es "
    "la replica AR (AG-8).",
    "ST-14 (§5.6 'trend ... con k/n >= 7/9') lectura literal mas conservadora: 'trend' solo en analisis descriptivos "
    "o exploratorios con n = 9 y k >= 7 datasets del mismo signo; nunca en secciones de H1-H4 (H2.a/H2.b con la "
    "primaria sin rechazar, Conv1D/Conv2D de H2 y C9 frente a original, que es el contraste de H1.e) (ST-M2).",
    "ST-15 (§6.3 LODO) empate 4/4 de la regla del signo mayoritario en un pliegue -> predice 'negativo' (Delta <= 0); se "
    "reporta el numero de pliegues empatados.",
    "ST-16 (§5.6, §12.2, §4.5) las familias de sensibilidad (S-ITT, divergidas sin imputar, S-SEED) reutilizan H1-H4 con "
    "Holm dentro de su propia familia, pero su redaccion es 'sensitivity_consistent' (mismo signo de HL y misma decision "
    "que el analisis principal) o 'sensitivity_not_consistent'; nunca 'significant'/'outperforms', no van al abstract "
    "ni asignan X13 (ST-M1).",
    "ST-17 (§4.5 'at the freeze of the results', §13.3 'without looking at results') a run of stats.py before "
    "common.FREEZE, with an incomplete aggregate or without the frozen plan, is a run on partial inputs: every label "
    "carries [PARTIAL INPUTS], numbers_proposed.md is marked in the same way and every run is appended to "
    "runs_log.jsonl (append-only) (ST-M4).",
    "ST-18 (§11 X13; F-08) el estado X13 por afirmacion enviada sale de una tabla fijada en git antes de la ejecucion "
    "(--x13-claims: TEX:linea -> familia, comparacion y signo); sin ella no se asigna ningun estado por afirmacion. Si "
    "la tabla no esta en git o tiene cambios sin commit, los estados se calculan pero x13_summary queda 'no fijada' y "
    "no se citan (citable = false); se registra si su ultimo commit es anterior al primer metrics.json.",
    "ST-19 (§5.2, §5.6) 'equivalent' exige el IC exacto de §5.2 (cobertura >= 95 %, n >= 6) entero dentro de +-0,01; "
    "con n <= 5 se reporta 'equivalence not assessable (coverage x%)' (ST-m1). Comparaciones 'not testable' y familias "
    "'uninformative by construction' tienen su propia regla de redaccion y no proponen filas .p/.W/.pHolm para "
    "numbers.md; w5 solo si la familia H1 es informativa (ST-m2).",
    "ST-20 (§14.1) en numbers_proposed.md solo se usan las etiquetas del vocabulario (CONFIRMATORIO / SENSIBILIDAD / "
    "EXPLORATORIO; ORACULO al final): los valores pre-registrados sin test (abstract, celdas de T1, rho_KNAD, omnibus) "
    "van como '[CONFIRMATORIO plan §x — descriptivo, sin test]' (ST-m4). best_per_dataset de R1.10 lleva [ORACULO: "
    "selected on evaluation results] (F-09).",
    "ST-21 (§4.2 S-s0-KNAD; ST-m6) the pre-registered sensitivity S-s0-KNAD (protocol of §2 on the scores stored "
    "from the original runs on the 6 KNAD captures) is not implemented in this version of "
    "metrics_rev/aggregate/stats: its input is not a job of the runner, and the alignment 'label of row i + 1 with "
    "header=None' is ambiguous because *_labels.csv has a header ('\"\",\"x\"') and an index column. stats.json "
    "declares it in S_s0_KNAD as 'no implementada', and no other analysis takes its place.",
]


class StatsError(RuntimeError):
    """Entrada invalida o cruce con scipy fuera de tolerancia (R-5)."""


def doubled_ranks(a, axis=-1):
    """Rangos medios x 2 como enteros exactos (int64)."""
    from scipy.stats import rankdata
    r = rankdata(np.asarray(a, dtype=np.float64), method="average", axis=axis)
    r2 = np.rint(r * 2.0).astype(np.int64)
    if not np.array_equal(r2.astype(np.float64), r * 2.0):
        raise StatsError("rangos medios no semienteros")
    return r2


def _sum_counts(weights):
    """counts[t] = n de asignaciones de signo con suma de los pesos positivos = t (programacion dinamica exacta)."""
    total = int(sum(weights))
    c = [0] * (total + 1)
    c[0] = 1
    for w in weights:
        w = int(w)
        for s in range(total, w - 1, -1):
            c[s] += c[s - w]
    return c


def null_counts_no_ties(n):
    """counts[t] = n de asignaciones con T+ = t para rangos 1..n (sin empates)."""
    return _sum_counts(range(1, n + 1))


def p_two_sided_no_ties(n, w):
    c = null_counts_no_ties(n)
    return min(Fraction(1), Fraction(2 * sum(c[:w + 1]), 2 ** n))


def wilcoxon_enumeration(d):
    """Enumeracion propia (§5.1): ceros fuera (wilcox), rangos medios de |d| sobre los n_eff no nulos."""
    d = np.asarray(d, dtype=np.float64)
    nz = d[d != 0.0]
    n_eff = int(len(nz))
    out = {"n": int(len(d)), "n_eff": n_eff, "n_zeros": int(len(d) - n_eff),
           "ties": bool(np.unique(np.abs(nz)).size < n_eff)}
    if n_eff == 0:
        out.update(t_plus=0.0, t_minus=0.0, W=0.0, p_enum=None, p_enum_fraction=None, r_rank_biserial=None)
        return out
    r2 = doubled_ranks(np.abs(nz))
    t2p = int(r2[nz > 0].sum())
    t2m = int(r2[nz < 0].sum())
    counts = _sum_counts(r2.tolist())
    tot = 2 ** n_eff
    le = sum(counts[:t2p + 1])
    ge = sum(counts[t2p:])
    p = min(Fraction(1), Fraction(2 * min(le, ge), tot))
    out.update(t_plus=t2p / 2.0, t_minus=t2m / 2.0, W=min(t2p, t2m) / 2.0, p_enum=float(p), p_enum_fraction=str(p),
               r_rank_biserial=(t2p - t2m) / (t2p + t2m))
    return out


def wilcoxon_bruteforce(d):
    """Fuerza bruta (tests): 2^n_eff signos sobre los rangos medios."""
    d = np.asarray(d, dtype=np.float64)
    nz = d[d != 0.0]
    if not len(nz):
        return None
    r2 = doubled_ranks(np.abs(nz))
    t2 = int(r2[nz > 0].sum())
    le = ge = 0
    for signs in itertools.product((0, 1), repeat=len(nz)):
        s = int(sum(r for r, g in zip(r2, signs) if g))
        le += s <= t2
        ge += s >= t2
    return float(min(Fraction(1), Fraction(2 * min(le, ge), 2 ** len(nz))))


def wilcoxon_test(d):
    """§5.1: Wilcoxon bilateral exacto (scipy) + enumeracion propia; exige |diff| <= 1e-12."""
    from scipy import stats as ss
    d = np.asarray(d, dtype=np.float64)
    out = wilcoxon_enumeration(d)
    out["testable"] = out["n_eff"] >= N_TESTABLE
    if out["n_eff"] == 0:
        out.update(method=None, p=None, p_scipy=None, abs_diff_enum_scipy=None,
                   note="todas las d son 0: sin test")
        return out
    if out["ties"] or out["n_zeros"]:
        method = "scipy.stats.wilcoxon(zero_method='wilcox', method=PermutationMethod(n_resamples=inf))"
        r = ss.wilcoxon(d, zero_method="wilcox", method=ss.PermutationMethod(n_resamples=np.inf))
    else:
        method = "scipy.stats.wilcoxon(zero_method='wilcox', method='exact')"
        r = ss.wilcoxon(d, zero_method="wilcox", method="exact")
    ps = float(r.pvalue)
    diff = abs(ps - out["p_enum"])
    if not diff <= CROSS_TOL:
        raise StatsError(f"Wilcoxon: p scipy {ps} != enumeracion {out['p_enum']} (|d| = {diff}; R-5)")
    if abs(float(r.statistic) - out["W"]) > 1e-9:
        raise StatsError(f"Wilcoxon: W scipy {r.statistic} != {out['W']}")
    out.update(method=method, p=ps, p_scipy=ps, abs_diff_enum_scipy=diff)
    if not out["testable"]:
        out["note"] = "not testable (n_eff < 6, §5.1): el p no entra en Holm"
    return out


def walsh_averages(d):
    d = np.asarray(d, dtype=np.float64)
    i, j = np.triu_indices(len(d))
    return np.sort((d[i] + d[j]) / 2.0)


def hodges_lehmann(d):
    d = np.asarray(d, dtype=np.float64)
    return float(np.median(walsh_averages(d))) if len(d) else None


def hl_ci_orders(n):
    """§5.2: ordenes (1-based) y cobertura exacta del IC de HL por inversion (como attainability.py)."""
    M_ = n * (n + 1) // 2
    c = null_counts_no_ties(n)
    tot = 2 ** n
    cum, k = 0, -1
    for t in range(0, M_ + 1):
        if Fraction(cum + c[t], tot) <= HL_TAIL:
            cum += c[t]
            k = t
        else:
            break
    if k < 0:
        return {"n": n, "M_walsh": M_, "lower_order": 1, "upper_order": M_,
                "coverage": float(1 - Fraction(2 * c[0], tot)), "reaches_95": False}
    return {"n": n, "M_walsh": M_, "lower_order": k + 1, "upper_order": M_ - k, "critical_k": k,
            "coverage": float(1 - Fraction(2 * sum(c[:k + 1]), tot)), "reaches_95": True}


def hl_ci(d):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n == 0:
        return None
    o = hl_ci_orders(n)
    A = walsh_averages(d)
    return dict(o, lower=float(A[o["lower_order"] - 1]), upper=float(A[o["upper_order"] - 1]),
                note="IC exacto por inversion de Wilcoxon, sin ajustar por multiplicidad (§5.2)")


def sign_test_p(n, k):
    k = max(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def n_min_holm(m, alpha=ALPHA):
    """§5.1: n minimo para alcanzar el paso 1 de Holm (m * 2 / 2^n <= alpha)."""
    n = 1
    while Fraction(2 * m, 2 ** n) > Fraction(alpha).limit_denominator(10 ** 6):
        n += 1
    return n


def holm(pvals, m=None, alpha=ALPHA):
    """Holm step-down adjusted p-values (plan §5.1); None marks a comparison that is not testable."""
    m = m or len(pvals)
    order = sorted([(p, i) for i, p in enumerate(pvals) if p is not None], key=lambda t: (t[0], t[1]))
    adj = [None] * len(pvals)
    run = 0.0
    for j, (p, i) in enumerate(order, start=1):
        run = max(run, min(1.0, (m - j + 1) * p))
        adj[i] = run
    return adj, [a is not None and a <= alpha for a in adj]


def holm_attainability(n, m):
    """Tabla de attainability.json: W maximo por paso de Holm (sin empates)."""
    rows = []
    maxs = n * (n + 1) // 2
    for j in range(1, m + 1):
        thr = ALPHA / (m - j + 1)
        best = None
        for w in range(0, maxs // 2 + 1):
            if float(p_two_sided_no_ties(n, w)) <= thr + 1e-15:
                best = w
            else:
                break
        if best is None:
            rows.append({"step": j, "threshold": thr, "max_W": None, "p_at_max_W": None, "rank_biserial_min_abs": None,
                         "min_sign_agreement": None, "attainable": False, "min_p_n": float(p_two_sided_no_ties(n, 0))})
            continue
        s = k = 0
        for r in range(1, n + 1):
            if s + r <= best:
                s += r
                k += 1
            else:
                break
        rows.append({"step": j, "threshold": thr, "max_W": best, "p_at_max_W": float(p_two_sided_no_ties(n, best)),
                     "rank_biserial_min_abs": (maxs - 2 * best) / maxs, "min_sign_agreement": n - k, "attainable": True})
    return rows


def _perms(n):
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


def spearman_exact(x, y, scipy_check=True):
    """§6.2: rho de Spearman y p exacto bilateral por permutacion completa (enumeracion entera + scipy)."""
    from scipy import stats as ss
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    out = {"n": n}
    if n < 2:
        out.update(rho=None, p=None, note="n < 2")
        return out
    cx = doubled_ranks(x) - (n + 1)
    cy = doubled_ranks(y) - (n + 1)
    sxx, syy = int((cx * cx).sum()), int((cy * cy).sum())
    if sxx == 0 or syy == 0:
        out.update(rho=None, p=None, note="variable constante: rho no definido")
        return out
    den = math.sqrt(sxx * syy)
    num = int((cx * cy).sum())
    rho = num / den
    P = _perms(n)
    nums = (cx[P] * cy[None, :]).sum(axis=1)
    tot = len(P)
    le, ge = int((nums <= num).sum()), int((nums >= num).sum())
    p = float(min(Fraction(1), Fraction(2 * min(le, ge), tot)))
    p_abs = int((np.abs(nums) >= abs(num)).sum()) / tot
    out.update(rho=rho, p=p, p_abs=p_abs, n_permutations=tot, method="enumeracion entera de n! emparejamientos")
    if scipy_check:
        rs = float(ss.spearmanr(x, y).statistic)

        def stat(a, axis=-1):
            ca = doubled_ranks(a, axis=axis) - (n + 1)
            return (ca * cy).sum(axis=axis) / den

        pt = ss.permutation_test((x,), stat, permutation_type="pairings", n_resamples=np.inf,
                                 alternative="two-sided", vectorized=True)
        d1, d2 = abs(float(pt.pvalue) - p), abs(rs - rho)
        if not (d1 <= CROSS_TOL and d2 <= CROSS_TOL):
            raise StatsError(f"Spearman: scipy p {pt.pvalue} / rho {rs} != enumeracion {p} / {rho} (R-5)")
        out.update(p_scipy=float(pt.pvalue), rho_scipy=rs, abs_diff_p=d1, abs_diff_rho=d2,
                   scipy="permutation_test((x,), pairings, n_resamples=inf) + spearmanr")
    return out


def spearman_null_table(n):
    """attainability.json 'spearman': p minimo y |rho| critico (sin empates)."""
    P = _perms(n)
    d = np.arange(n)[None, :] - P
    ss_ = (d * d).sum(axis=1)
    vals, counts = np.unique(ss_, return_counts=True)
    rho = 1 - 6 * vals / (n * (n * n - 1))
    tot = len(P)
    absr = np.round(np.abs(rho), 12)
    table = {float(r): float(counts[absr >= r - 1e-12].sum() / tot) for r in np.unique(absr)}
    crit = min((r for r, p in table.items() if p <= ALPHA), default=None)
    return {"n": n, "n_perm": tot, "min_p_two_sided": float(min(table.values())), "critical_abs_rho_alpha05": crit,
            "p_at_critical": table.get(crit)}


def friedman_iman_davenport(X, B=FRIEDMAN_B, seed=PERM_SEED):
    """Friedman test with the Iman-Davenport statistic (plan §5.3): asymptotic p of F and permutation p within each dataset."""
    from scipy import stats as ss
    X = np.asarray(X, dtype=np.float64)
    n, k = X.shape
    R2 = doubled_ranks(X, axis=1)
    S_obs = int((R2.sum(axis=0) ** 2).sum())
    chi2 = 12.0 / (n * k * (k + 1)) * (S_obs / 4.0) - 3.0 * n * (k + 1)
    den = n * (k - 1) - chi2
    F = (n - 1) * chi2 / den if den > 0 else math.inf
    df1, df2 = k - 1, (k - 1) * (n - 1)
    rng = np.random.default_rng(seed)
    perm = rng.permuted(np.broadcast_to(R2.astype(np.int16), (B, n, k)).copy(), axis=2)
    S = (perm.sum(axis=1, dtype=np.int64) ** 2).sum(axis=1)
    cnt = int((S >= S_obs).sum())
    try:
        chi2_ties = float(ss.friedmanchisquare(*X.T).statistic)
    except Exception:
        chi2_ties = None
    return {"n": n, "k": k, "chi2_demsar": chi2, "chi2_scipy_tie_corrected": chi2_ties, "F": F, "df": [df1, df2],
            "p_F_asymptotic": float(ss.f.sf(F, df1, df2)) if math.isfinite(F) else 0.0,
            "F_crit_0.05": float(ss.f.ppf(1 - ALPHA, df1, df2)), "B": B, "seed": seed, "count_ge": cnt,
            "p_permutation": (1 + cnt) / (B + 1), "mean_ranks": (R2.sum(axis=0) / (2.0 * n)).tolist(),
            "rank_direction": "1 = menor ROC-AUC, k = mayor", "decides": "p_permutation (§5.3, m1)"}


def median_iqr(v):
    v = np.asarray([x for x in v if x is not None], dtype=np.float64)
    if not len(v):
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {"n": int(len(v)), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)), "min": float(v.min()), "max": float(v.max())}


def describe(vals, order=None, with_ci=True):
    """HL (todas las d, ceros incluidos), IC exacto, k/n (§5.2)."""
    order = order or list(vals)
    keys = [k for k in order if vals.get(k) is not None]
    d = np.array([vals[k] for k in keys], dtype=np.float64)
    n = len(d)
    out = {"units": keys, "n": n, "d": {k: float(vals[k]) for k in keys},
           "k_pos": int((d > 0).sum()), "k_neg": int((d < 0).sum()), "k_zero": int((d == 0).sum()),
           "hl": hodges_lehmann(d) if n else None,
           "min_abs_d": float(np.min(np.abs(d))) if n else None}
    if with_ci:
        out["hl_ci"] = hl_ci(d) if n else None
        ci = out["hl_ci"]
        inside = bool(ci is not None and ci["lower"] >= -ROPE and ci["upper"] <= ROPE)
        out["ci_within_rope"] = inside
        out["equivalent_rope"] = bool(inside and ci["reaches_95"])
        if inside and not ci["reaches_95"]:
            out["equivalence"] = f"equivalence not assessable (coverage {100 * ci['coverage']:.2f}%, n = {n} <= 5; §5.2)"
    return out


def compare(d_by_ds, test=True, order=DS):
    out = describe(d_by_ds, order)
    out["datasets"] = out.pop("units")
    if test and out["n"]:
        out["wilcoxon"] = wilcoxon_test([out["d"][k] for k in out["datasets"]])
    return out


def _sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def _preserved(h, h_full):
    """§1.2 / §5.8: False si el signo es opuesto o h == 0 con h_full != 0; None si no evaluable."""
    if h is None or h_full is None:
        return None
    if _sign(h) * _sign(h_full) < 0:
        return False
    if h == 0 and h_full != 0:
        return False
    return True


def dependence(d_by_ds):
    """§1.2 (descriptivo, sin p): HL y k/n por subconjuntos; regla de redaccion con KNAD colapsado y sin KNAD."""
    subsets = {"full": DS, "without_KNAD": FLOW, "without_ACI": [d for d in DS if d != "D7"],
               "without_CIC2017": [d for d in DS if d != "D8"], "without_CIC2018": [d for d in DS if d != "D9"],
               "only_KNAD": KNAD}
    res = {}
    for name, members in subsets.items():
        blk = describe({d: d_by_ds.get(d) for d in members}, members, with_ci=(name == "only_KNAD"))
        res[name] = blk
    kv = [d_by_ds[d] for d in KNAD if d_by_ds.get(d) is not None]
    units = {}
    if kv:
        units["KNAD"] = float(sum(kv) / len(kv))
    for d in FLOW:
        if d_by_ds.get(d) is not None:
            units[d] = d_by_ds[d]
    res["KNAD_collapsed"] = dict(describe(units, ["KNAD"] + FLOW, with_ci=False), n_knad_averaged=len(kv),
                                 rule="ST-7: unidad KNAD = media de las d de los KNAD presentes")
    hf = res["full"]["hl"]
    pc = _preserved(res["KNAD_collapsed"]["hl"], hf)
    pw = _preserved(res["without_KNAD"]["hl"], hf)
    res["rule"] = {"direction_preserved_KNAD_collapsed": pc, "direction_preserved_without_KNAD": pw,
                   "dependence_ok": (pc is True and pw is True),
                   "if_not_ok": "; the direction was not preserved [when the six KNAD captures were collapsed into "
                                "one unit / outside the KNAD testbed] -> X13 como maximo 'debilitada'; fuera del "
                                "abstract, highlights y contribuciones (§1.2)",
                   "not_evaluable_is_not_ok": "ST-6"}
    res["tag"] = T["dep"]
    return res


def drift(dloc_by_ds, datasets, hl_fus):
    """§5.8.1: HL con AUC_loc(RAW) en los mismos datasets; signo opuesto o 0 -> 'not robust to within-block evaluation'."""
    vals = {d: dloc_by_ds.get(d) for d in datasets}
    if not datasets or any(v is None for v in vals.values()):
        return {"evaluable": False, "robust": False, "hl_auc_loc_raw": None, "n": len(datasets),
                "note": "AUC_loc(RAW) no disponible en algun dataset: no robusta (ST-6)", "tag": T["rob"]}
    h = hodges_lehmann([vals[d] for d in datasets])
    ok = _preserved(h, hl_fus)
    return {"evaluable": True, "robust": bool(ok), "hl_auc_loc_raw": h, "n": len(datasets),
            "d_auc_loc_raw": vals, "if_not_robust": "not robust to within-block evaluation (§5.8)",
            "note": "cambia a la vez la ventana de evaluacion y el score (sin detector), declarado (§5.8)",
            "tag": T["rob"]}


def family(comps, m, tag):
    """Holm de bajada con m fijo; 'uninformative by construction' si ninguna comparacion alcanza n_min(m) (§5.1)."""
    nmin = n_min_holm(m)
    effs = [((c.get("wilcoxon") or {}).get("n_eff") or 0) for c in comps.values()]
    informative = max(effs, default=0) >= nmin
    pv = [(c["wilcoxon"]["p"] if (c.get("wilcoxon") and c["wilcoxon"]["testable"]) else None) for c in comps.values()]
    if informative:
        adj, rej = holm(pv, m)
        status = "Holm step-down"
    else:
        adj, rej = [None] * len(pv), [False] * len(pv)
        ne = max(effs, default=0)
        status = (f"uninformative by construction (minimum Holm-adjusted p = m*2/2^n_eff = "
                  f"{(m * 2 / 2 ** ne) if ne else 'NA'} > 0.05)")
    for (name, c), a, r in zip(comps.items(), adj, rej):
        c["p_holm"] = a
        c["reject"] = bool(r)
        if not c.get("wilcoxon"):
            c["status"] = "not testable (sin datasets)"
        elif not c["wilcoxon"]["testable"]:
            c["status"] = "not testable (n_eff < 6)"
        else:
            c["status"] = "tested" if informative else "uninformative family"
    return {"m": m, "n_min": nmin, "informative": informative, "status": status, "alpha": ALPHA,
            "comparisons": comps, "tag": tag}


def trend_ok(c):
    """Rule 'k/n >= 7/9' of plan §5.6, read literally and conservatively: n = 9 and at least 7 datasets with the same sign."""
    k = max(c.get("k_pos", 0), c.get("k_neg", 0))
    return c.get("n", 0) == TREND_N and k >= TREND_MIN_K


def wording(c, confirmatory=True, trend_forbidden=False, trend_note=None):
    """§5.6 (mecanico; no redacta): regla de redaccion aplicable a una comparacion."""
    ci = c.get("hl_ci") or {}
    hl = c.get("hl")
    eq = bool(c.get("equivalent_rope"))
    status = str(c.get("status") or "")
    if confirmatory and c.get("reject"):
        rule = "outperforms" if (hl is not None and abs(hl) > ROPE) else "significant_below_margin"
        n_eff = (c.get("wilcoxon") or {}).get("n_eff")
        cons = ((c["k_pos"] == n_eff == c["n"] or c["k_neg"] == n_eff == c["n"]) and
                (ci.get("lower", 0) > 0 or ci.get("upper", 0) < 0))
        return {"rule": rule, "direction": "first > second" if (hl or 0) > 0 else "first < second",
                "consistently": bool(cons), "trend_allowed": False}
    if eq:
        rule = "equivalent"
    elif confirmatory and status.startswith("uninformative"):
        rule = "uninformative_by_construction"
    elif confirmatory and status.startswith("not testable"):
        rule = "not_testable"
    else:
        rule = "no_significant_difference" if confirmatory else "descriptive"
    out = {"rule": rule, "consistently": False,
           "trend_allowed": bool((not confirmatory) and (not trend_forbidden) and trend_ok(c)),
           "note": ("trend prohibido en H1-H4 y R1.4 (§5.6)" if confirmatory else
                    (trend_note or "trend prohibido en esta seccion (§5.6)") if trend_forbidden else
                    "descriptivo: sin 'significant'; trend solo con n = 9 y k >= 7 (§5.6, ST-14)")}
    if c.get("equivalence"):
        out["equivalence"] = c["equivalence"]
    return out


def sensitivity_wording(c, main_c):
    """Wording of a sensitivity family (plan §5.6): whether it agrees with the main analysis, never 'significant' or 'outperforms'."""
    hl, hm = c.get("hl"), (main_c or {}).get("hl")
    same = bool(hl is not None and hm is not None and _sign(hl) == _sign(hm) and _sign(hl) != 0)
    same_dec = bool(c.get("reject")) == bool((main_c or {}).get("reject"))
    return {"rule": "sensitivity_consistent" if (same and same_dec) else "sensitivity_not_consistent",
            "same_direction_as_main": same, "same_decision_as_main": same_dec,
            "rejects_within_sensitivity_family": bool(c.get("reject")),
            "main_rejects": bool((main_c or {}).get("reject")), "consistently": False, "trend_allowed": False,
            "note": "sensibilidad (§5.6): sin 'significant' ni 'outperforms'; p y Holm solo dentro de la familia de "
                    "sensibilidad y etiquetados como tales"}


def x13_status(sig, hl, robust, claimed_sign):
    same = hl is not None and _sign(hl) == claimed_sign
    if sig and same and robust:
        return "mantenida"
    if same:
        return "debilitada"
    return "retirada"


def annotate(c, confirmatory=True, robust_rules=True, trend_forbidden=False, trend_note=None):
    c["wording"] = wording(c, confirmatory, trend_forbidden, trend_note)
    if robust_rules:
        rob = c.get("robustness") or {}
        dr = (rob.get("drift") or {}).get("robust", True) if "drift" in rob else True
        dep = ((rob.get("dependence") or {}).get("rule") or {}).get("dependence_ok", False)
        c["robust"] = bool(dr and dep)
        c["abstract_eligible"] = bool(confirmatory and c.get("reject") and c["robust"])
        sig = bool(c.get("reject"))
        c["x13_if_claimed_first_higher"] = x13_status(sig, c.get("hl"), c["robust"], +1)
        c["x13_if_claimed_first_lower"] = x13_status(sig, c.get("hl"), c["robust"], -1)
    return c


def val(V, d, c, key="auroc_FUS"):
    return ((V.get(d) or {}).get(c) or {}).get(key)


def arch_val(V, d, a, key="auroc_FUS"):
    if a == "MLP":
        return val(V, d, "C9", key)
    x, y = (val(V, d, c, key) for c in ARCH[a])
    return None if (x is None or y is None) else (x + y) / 2.0


def diff(a, b):
    return None if (a is None or b is None) else a - b


def delta_h2(V, d, key="auroc_FUS"):
    """§5.4: (1/4) sum_A [AUC(A-AR) - AUC(A-TSR)]; None si falta alguna de las 8 celdas."""
    parts = [diff(val(V, d, ar, key), val(V, d, tsr, key)) for ar, tsr in ARCH.values()]
    return None if any(p is None for p in parts) else sum(parts) / 4.0


def h1(V, robust=True, tag=T["H1"], confirmatory=True):
    comps = {}
    for a in ARCH5:
        d = {x: diff(arch_val(V, x, a), val(V, x, CONTROL)) for x in DS}
        c = compare(d)
        c["contrast"] = f"AUC({a}) - AUC(original)"
        if robust:
            dl = {x: diff(arch_val(V, x, a, "auc_loc_RAW"), val(V, x, CONTROL, "auc_loc_RAW")) for x in DS}
            c["robustness"] = {"drift": drift(dl, c["datasets"], c["hl"]), "dependence": dependence(d)}
        comps[f"{a}_vs_original"] = c
    fam = family(comps, 5, tag)
    for c in comps.values():
        annotate(c, confirmatory, robust)
    return fam


def h1_descriptive(V, friedman_b=FRIEDMAN_B):
    out = {"tag": T["desc"], "vs_control": {}}
    for c in CONFIGS:
        if c == CONTROL:
            continue
        blk = compare({x: diff(val(V, x, c), val(V, x, CONTROL)) for x in DS}, test=False)
        blk["wording"] = wording(blk, confirmatory=False, trend_forbidden=(c == "C9"),
                                 trend_note="mismo contraste que H1.e (MLP frente a original): trend prohibido (§5.6)")
        out["vs_control"][c] = blk

    def ctx(x, key="auroc_FUS"):
        v = [val(V, x, c, key) for c in CTX8]
        return None if any(z is None for z in v) else sum(v) / 8.0
    blk = compare({x: diff(ctx(x), val(V, x, CONTROL)) for x in DS}, test=False)
    blk["contrast"] = "media de las 8 configuraciones con ventana - original"
    out["windowed8_mean_vs_control"] = blk
    full = [x for x in DS if all(val(V, x, c) is not None for c in CONFIGS)]
    out["omnibus"] = {"datasets": full, "tag": T["omni"],
                      "note": "ranks among Conv1D/Conv2D variants are within noise (§5.3, m10); sin diagrama CD"}
    if len(full) >= 2:
        X = np.array([[val(V, x, c) for c in CONFIGS] for x in full])
        out["omnibus"].update(friedman_iman_davenport(X, B=friedman_b))
        out["omnibus"]["mean_ranks_by_config"] = dict(zip(CONFIGS, out["omnibus"]["mean_ranks"]))
    else:
        out["omnibus"]["status"] = "no calculable (< 2 datasets con las 11 configuraciones)"
    return out


def h2(V, robust=True, tag=T["H2"], tag_s=T["H2s"], confirmatory=True):
    d = {x: delta_h2(V, x) for x in DS}
    c = compare(d)
    c["contrast"] = "(1/4) sum_{Conv1D,Conv2D,LSTM,MHA} [AUC(AR) - AUC(TSR)]"
    if robust:
        dl = {x: delta_h2(V, x, "auc_loc_RAW") for x in DS}
        c["robustness"] = {"drift": drift(dl, c["datasets"], c["hl"]), "dependence": dependence(d)}
    prim = family({"H2_primary": c}, 1, tag)
    annotate(c, confirmatory, robust)
    c["template_fields"] = {"hl": c["hl"], "ci": [((c.get("hl_ci") or {}).get("lower")),
                                                  ((c.get("hl_ci") or {}).get("upper"))],
                            "coverage": (c.get("hl_ci") or {}).get("coverage"),
                            "p_exact": (c.get("wilcoxon") or {}).get("p"), "n": c["n"], "k_AR_higher": c["k_pos"],
                            "note": "Conv1D and Conv2D predictions are nearly independent of the window by construction, "
                                    "which dilutes this average (§5.4)"}
    gate = bool(c.get("reject"))
    sec = {}
    for a in ("LSTM", "MHA"):
        ar, tsr = ARCH[a]
        dd = {x: diff(val(V, x, ar), val(V, x, tsr)) for x in DS}
        cs = compare(dd, test=gate)
        cs["contrast"] = f"AUC({a}-AR) - AUC({a}-TSR)"
        if robust:
            dl = {x: diff(val(V, x, ar, "auc_loc_RAW"), val(V, x, tsr, "auc_loc_RAW")) for x in DS}
            cs["robustness"] = {"drift": drift(dl, cs["datasets"], cs["hl"]), "dependence": dependence(dd)}
        sec[f"H2{'a' if a == 'LSTM' else 'b'}_{a}"] = cs
    h2note = "seccion de H2 (§5.4): trend prohibido en H1-H4 cuando no son significativas (§5.6; ST-M2)"
    if gate:
        secf = family(sec, 2, tag_s)
        for cs in sec.values():
            annotate(cs, confirmatory, robust)
            if confirmatory:
                cs["wording"]["qualifier"] = ("in the pre-specified per-architecture secondary analysis (Holm-adjusted, "
                                              "tested only because the primary test was significant)")
    else:
        secf = {"m": 2, "status": "descriptive (H2 primaria no rechaza: sin p, §5.4)", "comparisons": sec,
                "tag": "[DESCRIPTIVO plan §5.4]"}
        for cs in sec.values():
            annotate(cs, False, robust, trend_forbidden=True, trend_note=h2note)
    conv = {}
    for a in ("Conv1D", "Conv2D"):
        ar, tsr = ARCH[a]
        cc = compare({x: diff(val(V, x, ar), val(V, x, tsr)) for x in DS}, test=False)
        cc["contrast"] = f"AUC({a}-AR) - AUC({a}-TSR)"
        cc["wording"] = wording(cc, confirmatory=False, trend_forbidden=True, trend_note=h2note)
        conv[a] = cc
    return {"primary": prim, "gate_open": gate, "secondary": secf,
            "conv_descriptive": {"comparisons": conv, "tag": "[DESCRIPTIVO plan §5.4]",
                                 "note": "Conv1D/Conv2D: HL, IC, k/9 y regla de equivalencia, sin p"}}


def h3(V, cells, robust=True, tag=T["H3"], confirmatory=True):
    comps = {}
    for det in ("KMD_f", "MVD_f"):
        d, nconf = {}, {}
        for x in DS:
            parts = [diff(val(V, x, c), val(V, x, c, f"auroc_{det}")) for c in CONFIGS]
            parts = [p for p in parts if p is not None]
            nconf[x] = len(parts)
            d[x] = (sum(parts) / len(parts)) if parts else None
        c = compare(d)
        c["contrast"] = f"media sobre configuraciones no perdidas de [AUC(FUS) - AUC({det})]"
        c["n_configs_per_dataset"] = nconf
        if robust:
            c["robustness"] = {"dependence": dependence(d),
                               "drift": {"applies": False, "note": "§5.5: la regla por bloques no aplica a H3"}}
        comps[f"FUS_vs_{det}"] = c
    fam = family(comps, 2, tag)
    for c in comps.values():
        annotate(c, confirmatory, robust)
    auc_t = {x: _dataset_trivial(cells, x).get("auc_t") for x in DS}
    snf = {}
    for det, raw in (("KMD_f", "KMD_raw"), ("MVD_f", "MVD_raw")):
        d = {}
        for x in DS:
            parts = [diff(_cm(cells, x, c, "auroc_FUS_NF"), _cm(cells, x, c, f"auroc_{raw}")) for c in CONFIGS
                     if not _lost(cells, x, c)]
            parts = [p for p in parts if p is not None]
            d[x] = (sum(parts) / len(parts)) if parts else None
        blk = compare(d, test=False)
        blk["contrast"] = f"media de [AUC(FUS_NF) - AUC({raw})] (S-NF)"
        snf[f"FUS_NF_vs_{raw}"] = blk
    kvm = {"by_config": {}, "total": {"pos": 0, "neg": 0, "zero": 0}}
    gaps = []
    for c in CONFIGS:
        s = {"pos": 0, "neg": 0, "zero": 0, "n": 0}
        for x in DS:
            k_, m_, f_ = val(V, x, c, "auroc_KMD_f"), val(V, x, c, "auroc_MVD_f"), val(V, x, c)
            if k_ is None or m_ is None:
                continue
            s["n"] += 1
            key = "pos" if k_ > m_ else ("neg" if k_ < m_ else "zero")
            s[key] += 1
            kvm["total"][key] += 1
            if f_ is not None:
                gaps.append(max(k_, m_) - f_)
        kvm["by_config"][c] = s
    return {"family": fam, "auc_t_by_dataset": auc_t, "S_NF": {"comparisons": snf, "tag": "[SENSIBILIDAD plan §2.9 S-NF]"},
            "KMD_minus_MVD_sign_counts": dict(kvm, tag=T["desc"]),
            "best_of_two_minus_FUS": dict(median_iqr(gaps), tag=T["oracle_best2"],
                                          rule="max(AUC(KMD_f), AUC(MVD_f)) - AUC(FUS) por (dataset, configuracion)")}


def h4(V, robust=True, tag=T["H4"], h1fam=None, confirmatory=True):
    comps = {}
    for a in ("Conv1D", "Conv2D", "LSTM", "MHA"):
        d = {x: diff(arch_val(V, x, "MLP"), arch_val(V, x, a)) for x in DS}
        c = compare(d)
        c["contrast"] = f"AUC(MLP) - AUC({a})"
        if robust:
            dl = {x: diff(arch_val(V, x, "MLP", "auc_loc_RAW"), arch_val(V, x, a, "auc_loc_RAW")) for x in DS}
            c["robustness"] = {"drift": drift(dl, c["datasets"], c["hl"]), "dependence": dependence(d)}
        comps[f"MLP_vs_{a}"] = c
    fam = family(comps, 4, tag)
    for c in comps.values():
        annotate(c, confirmatory, robust)
    cs = list(comps.values())
    all_rej_pos = all(c.get("reject") and (c.get("hl") or 0) > 0 for c in cs)
    all_pos = all((c.get("hl") or 0) > 0 for c in cs)
    rob = all(c.get("robust") for c in cs) if robust else None
    claim = ("mantenida" if (all_rej_pos and rob) else ("debilitada" if all_pos else "retirada"))
    margin = any((c.get("hl") or 0) <= ROPE for c in cs)
    ext = None
    if h1fam is not None:
        e = h1fam["comparisons"].get("MLP_vs_original") or {}
        ext = bool(claim == "mantenida" and e.get("reject") and (e.get("hl") or 0) > 0 and e.get("robust"))
    if not confirmatory:
        fam["claim_mlp_highest"] = {"x13": None, "all_four_rejected_with_hl_positive_within_sensitivity": all_rej_pos,
                                    "note": "sensibilidad: no asigna estado X13 (§11, §5.6)", "tag": tag}
        return fam
    fam["claim_mlp_highest"] = {"x13": claim, "all_four_rejected_with_hl_positive": all_rej_pos,
                                "all_robust": rob, "margin_clause_needed": bool(all_rej_pos and margin),
                                "extension_to_kitnet_style_allowed": ext,
                                "rule": "§5.7 / ST-10", "tag": tag}
    return fam


def _lost(cells, x, c):
    return bool(((cells.get(x) or {}).get(c) or {}).get("lost", True))


def _cm(cells, x, c, key):
    cc = (cells.get(x) or {}).get(c) or {}
    if cc.get("lost"):
        return None
    return (cc.get("mean") or {}).get(key)


def _dataset_trivial(cells, x):
    for c in CONFIGS:
        m = ((cells.get(x) or {}).get(c) or {}).get("mean_raw") or {}
        if m.get("E_prevalence") is not None:
            return {k: m.get(f"E_{k}") for k in ("auc_t", "prevalence", "f1_trivial", "ap_trivial")}
    return {}


def load_descriptor(path):
    if path is None or not Path(path).exists():
        return None
    Dj = C.read_json(path)
    if not isinstance(Dj, dict) or Dj.get("schema") != SCHEMA_D:
        raise StatsError(f"{path}: schema distinto de {SCHEMA_D}")
    return Dj


def _aware(t):
    """ISO -> datetime con zona (las ingenuas son hora local)."""
    import datetime as dt
    d = dt.datetime.fromisoformat(str(t))
    return d if d.tzinfo else d.astimezone()


def d_guard(Dj, needed, chrono=None):
    """Guard of the descriptor D (plan §6.1): blocks H2 and R1.4 without D; R1.4 is confirmatory only if D predates the metrics."""
    now = C.now_iso()
    if Dj is None:
        return False, {"status": "bloqueado", "reason": "falta descriptor_d.json: D debe calcularse antes de cualquier "
                                                       "Delta (§6.1, E-M4). Ejecuta 'stats.py descriptor-d'",
                       "needed": needed, "r14_confirmatory": False}
    have = {d: e for d, e in (Dj.get("datasets") or {}).items() if (e or {}).get("D") is not None}
    missing = [d for d in needed if d not in have]
    if missing:
        return False, {"status": "bloqueado", "reason": "D ausente en datasets con Delta calculable (§6.1)",
                       "missing": missing, "needed": needed, "r14_confirmatory": False}
    chrono = chrono or {}
    by_ds = chrono.get("first_metrics_json_created_by_dataset") or {}
    glob = chrono.get("first_metrics_json_created")
    order, late, unverifiable = {}, [], []
    for d in needed:
        e = have[d]
        ref = by_ds.get(d) or glob
        rec = {"D_created": e.get("created"), "first_metrics_json_created": ref,
               "reference": "dataset" if by_ds.get(d) else ("global" if glob else None)}
        try:
            rec["D_before_first_metrics_json"] = (_aware(e["created"]) < _aware(ref)) if (ref and e.get("created")) \
                else None
        except (TypeError, ValueError):
            rec["D_before_first_metrics_json"] = None
        if rec["D_before_first_metrics_json"] is None:
            unverifiable.append(d)
        elif not rec["D_before_first_metrics_json"]:
            late.append(d)
        order[d] = rec
    codes = {d: have[d].get("d_code_sha256") for d in needed}
    code_ok = bool(needed) and None not in codes.values() and len(set(codes.values())) == 1
    conf = not late and not unverifiable and code_ok
    return True, {"status": "ok", "r14_confirmatory": conf, "order_by_dataset": order, "D_after_metrics": late,
                  "not_verifiable": unverifiable, "d_code_sha256_by_dataset": codes, "d_code_consistent": code_ok,
                  "delta_computed_at": now,
                  "rule": "ST-M3: D anterior al primer metrics.json de su dataset y mismo codigo de D en todos; si no, "
                          "R1.4 [EXPLORATORIO]"}


def r14(V, Dj, h2res, label_counts, cells, abl):
    delta = {x: delta_h2(V, x) for x in DS}
    Dd = {x: ((Dj.get("datasets") or {}).get(x) or {}).get("D") for x in DS}
    both = [x for x in DS if delta.get(x) is not None and Dd.get(x) is not None]
    n = len(both)
    out = {"tag": T["R14"], "datasets": both, "n": n, "D": {x: Dd[x] for x in both}, "delta": {x: delta[x] for x in both},
           "descriptor_created": {x: (Dj["datasets"][x] or {}).get("created") for x in both}}
    if n < 5:
        out.update(status="not testable (n < 5, §6.2)")
    else:
        sp = spearman_exact([Dd[x] for x in both], [delta[x] for x in both])
        out["spearman"] = sp
        out["reject"] = bool(sp.get("p") is not None and sp["p"] <= ALPHA)
        out["critical_abs_rho"] = spearman_null_table(n)["critical_abs_rho_alpha05"]
        out["status"] = "tested (familia m = 1)"
    kn = [x for x in both if x in KNAD]
    if len(kn) >= 2:
        sk = spearman_exact([Dd[x] for x in kn], [delta[x] for x in kn])
        out["rho_KNAD"] = dict(sk, datasets=kn, tag="[DESCRIPTIVO plan §6.2 rho_KNAD]")
    if n:
        order = sorted(both, key=lambda x: Dd[x])
        k = len(kn)
        extreme = k > 0 and (set(order[:k]) == set(kn) or set(order[-k:]) == set(kn))
        out["knad_occupy_extreme_D_ranks"] = bool(extreme)
        if extreme:
            out["interpretation_rule"] = "contraste entre corpus (§6.2, E-M4)"
    rho, rk = (out.get("spearman") or {}).get("rho"), (out.get("rho_KNAD") or {}).get("rho")
    out["sequential_dependence_sentence_allowed"] = bool(rho is not None and rk is not None and _sign(rho) == _sign(rk)
                                                         and _sign(rho) != 0)
    if not out["sequential_dependence_sentence_allowed"]:
        out["required_sentence_if_discussed"] = ("the association reflects the difference between the packet-level KNAD "
                                                 "captures and the flow-level datasets and could not be separated from it")
    out["dependence_delta"] = (h2res["primary"]["comparisons"]["H2_primary"].get("robustness") or {}).get("dependence")
    out["exploratory"] = r14_exploratory(V, Dj, delta, label_counts, cells, abl)
    return out


def _expl_corr(x_by, y_by, label):
    ks = [k for k in DS if x_by.get(k) is not None and y_by.get(k) is not None]
    blk = {"label": label, "datasets": ks, "n": len(ks),
           "tag": f"[EXPLORATORIO] (exploratory; unadjusted p; one of N = {N_EXPL_R14} exploratory correlations)"}
    if len(ks) >= 3:
        blk.update(spearman_exact([x_by[k] for k in ks], [y_by[k] for k in ks]))
    else:
        blk["status"] = "no calculable (n < 3)"
    return blk


def _count_expl(o):
    """Number of exploratory correlations with a computed rho."""
    if isinstance(o, dict):
        own = 1 if (str(o.get("tag", "")).startswith("[EXPLORATORIO] (exploratory") and o.get("rho") is not None) else 0
        return own + sum(_count_expl(v) for v in o.values() if isinstance(v, (dict, list)))
    if isinstance(o, list):
        return sum(_count_expl(v) for v in o)
    return 0


def r14_exploratory(V, Dj, delta, lc, cells, abl):
    D_ = {x: ((Dj.get("datasets") or {}).get(x) or {}).get("D") for x in DS}
    out = {"N_expl": N_EXPL_R14, "tag": T["expl"]}
    e41 = {}
    for seg in ("train", "eval"):
        dseg = {x: (((Dj.get("datasets") or {}).get(x) or {}).get("E41") or {}).get(f"D_{seg}") for x in DS}
        e41[f"D_{seg}_vs_delta"] = _expl_corr(dseg, delta, f"E4.1 D sobre {seg} frente a Delta")
    out["E4.1"] = e41
    for key, lab in (("continuity_5tuple", "E4.2"), ("continuity_dstport", "E4.3")):
        vals = {x: ((Dj.get("datasets") or {}).get(x) or {}).get(key) for x in DS}
        out[lab] = (_expl_corr(vals, delta, f"{lab} {key} frente a Delta") if any(v is not None for v in vals.values())
                    else {"status": "no calculado: insumo no disponible en descriptor_d.json", "tag": T["expl"]})
    out["E4.4"] = {a: _expl_corr(D_, {x: diff(val(V, x, ARCH[a][0]), val(V, x, ARCH[a][1])) for x in DS},
                                 f"E4.4 D frente a Delta de {a}") for a in ARCH}
    out["E4.5"] = {"KNAD": describe({x: delta.get(x) for x in KNAD}, KNAD, with_ci=False),
                   "flow": describe({x: delta.get(x) for x in FLOW}, FLOW, with_ci=False),
                   "tag": "[EXPLORATORIO] mediana y k/n, sin p"}
    out["E4.5"]["KNAD"]["median"] = _median([delta.get(x) for x in KNAD])
    out["E4.5"]["flow"]["median"] = _median([delta.get(x) for x in FLOW])
    prev = {x: ((lc.get(DS_KEY[x]) or {}).get("eval") or {}).get("prevalence") for x in DS}
    nep = {x: ((((lc.get(DS_KEY[x]) or {}).get("eval") or {}).get("episodes") or {}).get("G10000") or {}).get("n")
           for x in DS}
    lep = {x: ((((lc.get(DS_KEY[x]) or {}).get("eval") or {}).get("episodes") or {}).get("G10000") or {}).get("mean")
           for x in DS}
    out["E4.6"] = {"prevalence": _expl_corr(prev, delta, "E4.6 prevalencia de E frente a Delta"),
                   "n_episodes_G10000": _expl_corr(nep, delta, "E4.6 n de episodios frente a Delta"),
                   "mean_episode_len_G10000": _expl_corr(lep, delta, "E4.6 longitud media de episodio frente a Delta")}
    out["E4.7"] = e47(abl)
    out["E4.8"] = e48(cells)
    out["LODO_threshold_rule"] = lodo_rule(D_, delta)
    out["N_computed"] = _count_expl(out)
    out["N_note"] = ("N_expl = 11 fijado por el plan (§6.3) y se declara siempre; N_computed = correlaciones con rho "
                     "calculado (E4.2/E4.3 no se calculan si falta el insumo). Los demas exploratorios (por dia, por "
                     "tipo, FPR frente al tiempo, IC bootstrap) se cuentan por tabla en exploratory_counts (§12.1)")
    return out


def _median(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def e47(abl):
    """E4.7: interaccion paradigma x L con T2 (semilla A; descriptivo)."""
    idx = {}
    for it in abl or []:
        if it["block"] in ("T2-min", "T2-ext") and it.get("auroc_ablation") is not None:
            idx[(it["dataset_id"], it["rowcfg"], str(it["value"]))] = it
    rows = []
    for x in ("D1", "D8"):
        for a in ("lstm", "mha"):
            for L in ("200", "2000"):
                ar, ts = idx.get((x, f"{a}-AR", L)), idx.get((x, f"{a}-TSR", L))
                rows.append({"dataset_id": x, "arch": a, "L": int(L),
                             "delta_AR_minus_TSR": diff((ar or {}).get("auroc_ablation"), (ts or {}).get("auroc_ablation"))})
            ref_ar = next((idx[k] for k in idx if k[:2] == (x, f"{a}-AR")), {})
            ref_ts = next((idx[k] for k in idx if k[:2] == (x, f"{a}-TSR")), {})
            rows.append({"dataset_id": x, "arch": a, "L": 800, "delta_AR_minus_TSR":
                         diff(ref_ar.get("auroc_reference"), ref_ts.get("auroc_reference")),
                         "source": "referencia T1-A semilla A de las celdas T2 (una semilla, como las ablaciones)"})
    return {"rows": rows, "tag": "[EXPLORATORIO] E4.7 (2 datasets, una semilla; sin p)"}


def e48(cells):
    """E4.8: cortes por dia (ROC-AUC por dia elegible) y por tipo de ataque (recall de FUS) del Delta."""
    days, types = {}, {}
    for x in DS:
        keys = set()
        for c in CTX8:
            keys |= set(((cells.get(x) or {}).get(c) or {}).get("mean") or {})
        for k in sorted(keys):
            if k.startswith("day_") and k.endswith("_auroc_FUS"):
                dname = k[len("day_"):-len("_auroc_FUS")]
                parts = [diff(_cm(cells, x, ar, k), _cm(cells, x, ts, k)) for ar, ts in ARCH.values()]
                days.setdefault(x, {})[dname] = None if any(p is None for p in parts) else sum(parts) / 4.0
            if k.startswith("type_") and k.endswith("_recall_FUS"):
                tname = k[len("type_"):-len("_recall_FUS")]
                parts = [diff(_cm(cells, x, ar, k), _cm(cells, x, ts, k)) for ar, ts in ARCH.values()]
                types.setdefault(x, {})[tname] = None if any(p is None for p in parts) else sum(parts) / 4.0
    return {"delta_auroc_by_day": days, "delta_recall_by_attack_type": types,
            "tag": "[EXPLORATORIO] E4.8 (sin p con dias o tipos como unidades, §3.5)"}


def lodo_rule(D_, delta):
    """§6.3: regla 'usar AR si D > x' con validacion dejando un dataset fuera (exploratoria)."""
    ks = [x for x in DS if D_.get(x) is not None and delta.get(x) is not None]
    if len(ks) < 3:
        return {"status": "no calculable", "tag": T["expl"]}
    folds, hits, hits_major, ties = [], 0, 0, 0
    for out_ in ks:
        tr = [x for x in ks if x != out_]
        vals = sorted(set(D_[x] for x in tr))
        cands = [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:])]
        best = None
        for direction in ("gt", "lt"):
            for thr in cands:
                acc = sum(((D_[x] > thr) if direction == "gt" else (D_[x] < thr)) == (delta[x] > 0) for x in tr)
                if best is None or acc > best[0]:
                    best = (acc, direction, thr)
        pos = sum(delta[x] > 0 for x in tr)
        major = pos > len(tr) - pos
        ties += int(pos == len(tr) - pos)
        if best is None:
            pred = major
        else:
            _a, direction, thr = best
            pred = (D_[out_] > thr) if direction == "gt" else (D_[out_] < thr)
        truth = delta[out_] > 0
        hits += pred == truth
        hits_major += major == truth
        folds.append({"left_out": out_, "threshold": None if best is None else best[2],
                      "direction": None if best is None else best[1], "train_hits": None if best is None else best[0],
                      "predicted_positive": bool(pred), "actual_positive": bool(truth)})
    return {"n": len(ks), "hits": int(hits), "hits_majority_rule": int(hits_major), "folds": folds,
            "majority_ties": int(ties),
            "rule": "§6.3: umbrales = puntos medios de D en los 8 de entrenamiento; dos direcciones; empate -> primera "
                    "direccion ('gt') y umbral menor; se compara con la regla del signo mayoritario (LODO); empate de "
                    "la regla mayoritaria (4/4) -> predice 'negativo' (Delta <= 0), declarado (ST-15)",
            "tag": T["expl"]}


def ablations_summary(abl, datasets):
    out = {"tag": T["abl"], "single_seed": "una sola semilla (T2-B: semilla B), declarado (§7.1)", "blocks": {}}
    groups = {}
    for it in abl or []:
        groups.setdefault(it["block"], []).append(it)
    for b, items in sorted(groups.items()):
        per_ds = {}
        for it in items:
            per_ds.setdefault((it["dataset_id"], str(it["value"])), []).append(it)
        summ = []
        for (x, v), its in sorted(per_ds.items()):
            post = [i["job_id"] for i in its if i.get("post_freeze")]
            its = [i for i in its if not i.get("post_freeze")]
            ds_ = [i["delta_auroc"] for i in its if i.get("delta_auroc") is not None]
            summ.append({"dataset_id": x, "value": v, "n_configs": len(its), "n_with_delta": len(ds_),
                         "post_freeze_excluded": post,
                         "delta_auroc_median": _median(ds_), "delta_auroc_min": min(ds_) if ds_ else None,
                         "delta_auroc_max": max(ds_) if ds_ else None,
                         "n_within_noise": sum(1 for i in its if i.get("within_noise")),
                         "lost_or_pending": [i["job_id"] for i in its if i.get("delta_auroc") is None]})
        out["blocks"][b] = {"cells": items, "summary_by_dataset_and_value": summ,
                            "note": "sin IC entre datasets cuando n = 2 (§7.1); |Delta| <= umbral de ruido no se "
                                    "interpreta como efecto"}
    t3a = {}
    for it in groups.get("T3-min-a", []):
        if it.get("post_freeze"):
            continue
        t3a.setdefault((it["rowcfg"], str(it["value"])), {})[it["dataset_id"]] = it.get("delta_auroc")
    out["T3-min-a_hl"] = {f"{rc}|k={k}": dict(describe(v, DS, with_ci=True), tag=T["t3a"] + " HL e IC exacto, sin p")
                          for (rc, k), v in sorted(t3a.items())}
    return out


def t4p_summary(cells):
    out = {"tag": T["t4"], "a": {}, "a_substitute_S_INS": {}, "b": {}, "c": {}}
    ks = ("auroc", "auprc", "threshold", "f1", "precision", "recall", "fpr")
    for x in DS:
        dc = cells.get(x) or {}
        est = [c for c in CONFIGS if _cm(cells, x, c, "t4pa_estimable") == 1.0]
        if est:
            blk = {"configs": {}, "summary": {}, "tag": T["t4"] + " T4-P-a [NO DESPLEGABLE]"}
            for c in est:
                blk["configs"][c] = {f"{X}_{k}": _cm(cells, x, c, f"t4pa_diff_{X}_{k}")
                                     for X in ("FUS", "KMD_f", "MVD_f") for k in ks}
            for X in ("FUS", "KMD_f", "MVD_f"):
                for k in ks:
                    blk["summary"][f"{X}_{k}"] = median_iqr([blk["configs"][c][f"{X}_{k}"] for c in est])
            out["a"][x] = blk
        nest = [c for c in CONFIGS if _cm(cells, x, c, "t4pa_estimable") == 0.0]
        if nest:
            blk = {"configs": {}, "summary": {}, "tag": T["t4"] + " T4-P-a no estimable -> S-INS [IN-SAMPLE]"}
            for c in nest:
                rec = {}
                for X in ("FUS", "KMD_f", "MVD_f"):
                    for k in ks:
                        main_k = {"auroc": f"auroc_{X}" if X != "FUS" else "auroc_FUS"}.get(k, f"{k}_{X}")
                        rec[f"{X}_{k}"] = diff(_cm(cells, x, c, main_k), _cm(cells, x, c, f"{k}_{X}_ins"))
                blk["configs"][c] = rec
            for key in next(iter(blk["configs"].values())):
                blk["summary"][key] = median_iqr([blk["configs"][c][key] for c in nest])
            blk["rule"] = "natural - S-INS (calibracion limpia dentro de muestra), §8.2"
            out["a_substitute_S_INS"][x] = blk
        for part, prefix in (("b", "t4pb_c"), ("c", "t4pc_f")):
            levels = sorted({k[len(prefix):].split("_")[0] for c in CONFIGS if not _lost(cells, x, c)
                             for k in ((dc.get(c) or {}).get("mean") or {}) if k.startswith(prefix)},
                            key=lambda s: float(s))
            if not levels:
                continue
            rows = []
            for lv in levels:
                for c in CONFIGS:
                    if _lost(cells, x, c):
                        continue
                    r = {"config": c, "level": lv, "effective_contamination": _cm(cells, x, c, f"{prefix}{lv}_eff")}
                    for k in ("auroc", "auprc", "threshold", "f1", "precision", "recall", "fpr"):
                        r[f"FUS_{k}"] = _cm(cells, x, c, f"{prefix}{lv}_FUS_{k}")
                    rows.append(r)
            summ = {lv: {k: median_iqr([r[k] for r in rows if r["level"] == lv])
                         for k in ("effective_contamination", "FUS_auroc", "FUS_auprc", "FUS_threshold", "FUS_f1",
                                   "FUS_precision", "FUS_recall", "FUS_fpr")} for lv in levels}
            out[part][x] = {"rows": rows, "summary_over_configs": summ,
                            "tag": T["t4"] + (" T4-P-b (E_T4)" if part == "b" else " T4-P-c (evaluacion comun)")}
    return out


def failure_cases(V, cells, Dj):
    per = {}
    for x in DS:
        s = [val(V, x, c) for c in SIMPLE if val(V, x, c) is not None]
        k = [val(V, x, c) for c in CTX8 if val(V, x, c) is not None]
        if s and k:
            ms, mk = sum(s) / len(s), sum(k) / len(k)
            per[x] = {"simple_mean": ms, "context_mean": mk, "simple_minus_context": ms - mk, "n_simple": len(s),
                      "n_context": len(k), "is_failure_case": (ms - mk) > FAILURE_MARGIN}
    sel = [x for x, r in per.items() if r["is_failure_case"]]
    rule = "simple - context > 0,01 (§10)"
    if not sel and per:
        sel = [min(per, key=lambda x: per[x]["context_mean"] - per[x]["simple_mean"])]
        rule = "ningun dataset supera el margen: el de menor (con contexto - simple) (§10)"
    analyses = {}
    for x in sel:
        rec = {"recall_by_attack_type": {}, "D": (((Dj or {}).get("datasets") or {}).get(x) or {}).get("D")}
        for c, lab in (("C10", "original"), ("C9", "MLP"), ("C5", "LSTM-AR")):
            mm = ((cells.get(x) or {}).get(c) or {}).get("mean") or {}
            rec["recall_by_attack_type"][lab] = {k[len("type_"):-len("_recall_FUS")]: v for k, v in mm.items()
                                                 if k.startswith("type_") and k.endswith("_recall_FUS")}
        rec["pending_need_scores"] = ["2: series RAW/FUS en +-20.000 filas alrededor de cada episodio",
                                      "4: origen de los falsos positivos (1 % benigno con mayor FUS)",
                                      "5: solapamiento de los 1.000 FP/FN mas extremos con problemas de etiqueta"]
        rec["not_computed"] = "6: mitigaciones (exploratorias, sin afirmaciones de eficacia)"
        analyses[x] = rec
    return {"per_dataset": per, "selected": sel, "selection_rule": rule, "analyses": analyses, "tag": T["fail"],
            "definition": "simple = C9, C10, C11 (sin contexto); con contexto = C1-C8; medias de familia (ST-11)"}


def abstract_numbers(V, h1fam, h2res, trivial):
    def rng(v):
        v = [z for z in v if z is not None]
        return {"n": len(v), "median": _median(v), "min": min(v) if v else None, "max": max(v) if v else None}

    def ctx(x, key):
        v = [val(V, x, c, key) for c in CTX8]
        return None if any(z is None for z in v) else sum(v) / 8.0
    out = {"tag": T["abs"], "config": "FUS, umbral P99 de calibracion; ROC-AUC primaria y F1 calibrado; S*_d",
           "context8_rule": "media de las 8 configuraciones con contexto; un dataset sin alguna de ellas queda fuera "
                            "y se reporta n (ST-12, ST-m9)"}
    out["1_control_auroc"] = rng([val(V, x, CONTROL) for x in DS])
    out["1_control_auc_loc_raw_median"] = _median([val(V, x, CONTROL, "auc_loc_RAW") for x in DS])
    out["2_context8_auroc"] = rng([ctx(x, "auroc_FUS") for x in DS])
    out["2_context8_auc_loc_raw_median"] = _median([ctx(x, "auc_loc_RAW") for x in DS])
    out["2_context8_n_configs_per_dataset"] = {x: sum(1 for c in CTX8 if val(V, x, c) is not None) for x in DS}
    out["2_context8_n_datasets"] = sum(1 for x in DS if ctx(x, "auroc_FUS") is not None)
    elig = {k: {"hl": c["hl"], "ci": [(c.get("hl_ci") or {}).get("lower"), (c.get("hl_ci") or {}).get("upper")],
                "coverage": (c.get("hl_ci") or {}).get("coverage")}
            for k, c in h1fam["comparisons"].items() if c.get("abstract_eligible")}
    hls = [c["hl"] for c in h1fam["comparisons"].values() if c.get("hl") is not None]
    any_sig = any(c.get("reject") for c in h1fam["comparisons"].values())
    informative = bool(h1fam.get("informative"))
    ns = {k: c.get("n") for k, c in h1fam["comparisons"].items()}
    w5 = (not any_sig) and informative
    out["3_H1"] = {"eligible": elig, "any_significant": any_sig, "family_informative": informative,
                   "w5_applies": w5, "w5_hl_range": ([min(hls), max(hls)] if (hls and w5) else None),
                   "n_by_comparison": ns,
                   "w5_n_note": (None if (not w5 or all(v == 9 for v in ns.values())) else
                                 "la frase w5 dice 'n = 9 datasets': con otro n se escribe el n real de cada comparacion"),
                   "uninformative_note": (None if informative else
                                          "familia H1 'uninformative by construction' (§5.1): w5 no aplica; se "
                                          "reportan HL, IC y k/n con el n real, sin p ajustados"),
                   "rule": "significativas y robustas (§5.8) con HL e IC; si ninguna es significativa y la familia es "
                           "informativa, frase w5 con el rango de HL; significativas no robustas no van al abstract"}
    p = h2res["primary"]["comparisons"]["H2_primary"]
    out["4_H2_primary"] = {"reject": p.get("reject"), "robust": p.get("robust"),
                           "abstract_eligible": p.get("abstract_eligible"), "hl": p.get("hl")}
    f1c = {x: val(V, x, CONTROL, "f1_FUS") for x in DS}
    f1k = {x: ctx(x, "f1_FUS") for x in DS}
    ft = {x: (trivial.get(x) or {}).get("f1_trivial") for x in DS}
    out["5_f1"] = {"control_median": _median(list(f1c.values())), "context8_median": _median(list(f1k.values())),
                   "control_n_above_f1_trivial": sum(1 for x in DS if f1c[x] is not None and ft.get(x) is not None
                                                     and f1c[x] > ft[x]),
                   "context8_n_above_f1_trivial": sum(1 for x in DS if f1k[x] is not None and ft.get(x) is not None
                                                      and f1k[x] > ft[x]),
                   "n_datasets": sum(1 for x in DS if f1c[x] is not None),
                   "n_datasets_context8": sum(1 for x in DS if f1k[x] is not None), "rule": "ST-12"}
    out["forbidden"] = ["mejor modelo por dataset", "cifras O1/O2", "ablaciones con 2 datasets como generales",
                        "resultados que no cumplen §5.8"]
    return out


def r110(V):
    heat = {x: {c: val(V, x, c) for c in CONFIGS} for x in DS}
    best = {}
    for x in DS:
        vs = {c: v for c, v in heat[x].items() if v is not None}
        if vs:
            mx = max(vs.values())
            best[x] = {"configs": [c for c, v in vs.items() if v == mx], "auroc": mx, "tag": T["r110_best"]}
    detects = {x: {c: (None if val(V, x, c, "auc_loc_RAW") is None else bool(val(V, x, c, "auc_loc_RAW") > DETECT_AUC_LOC))
                   for c in CONFIGS} for x in DS}
    return {"heatmap_auroc_fus": heat, "best_per_dataset": best, "best_per_dataset_tag": T["r110_best"],
            "note": "selected on evaluation results; descriptive, not an estimate of expected performance",
            "detects_allowed_auc_loc_raw_gt_0.51": detects, "tag": T["r110"]}


def numbers_rows(S, cmd, script_ref, prefix=""):
    """Rows of the numbers table; `prefix` is put in front of the label of every row."""
    rows = []

    def add(idd, v, ptr, tag):
        if v is None:
            return
        rows.append((idd, repr(float(v)) if isinstance(v, (float, int)) and not isinstance(v, bool) else str(v),
                     f"{prefix}{tag} results/revision/stats/stats.json#{ptr}"))

    def comp_rows(idp, fam, ptr, tag, names):
        for key, short in names.items():
            c = fam["comparisons"].get(key)
            if not c:
                continue
            p = f"{ptr}/comparisons/{key}"
            add(f"{idp}.{short}.HL", c.get("hl"), f"{p}/hl", tag)
            ci = c.get("hl_ci") or {}
            add(f"{idp}.{short}.CIlo", ci.get("lower"), f"{p}/hl_ci/lower", tag)
            add(f"{idp}.{short}.CIhi", ci.get("upper"), f"{p}/hl_ci/upper", tag)
            add(f"{idp}.{short}.CIcov", ci.get("coverage"), f"{p}/hl_ci/coverage", tag)
            w = c.get("wilcoxon") or {}
            if c.get("status") == "tested":
                add(f"{idp}.{short}.p", w.get("p"), f"{p}/wilcoxon/p", tag)
                add(f"{idp}.{short}.W", w.get("W"), f"{p}/wilcoxon/W", tag)
                add(f"{idp}.{short}.pHolm", c.get("p_holm"), f"{p}/p_holm", tag)
            add(f"{idp}.{short}.neff", w.get("n_eff"), f"{p}/wilcoxon/n_eff", tag)
            add(f"{idp}.{short}.n", c.get("n"), f"{p}/n", tag)
            add(f"{idp}.{short}.kpos", c.get("k_pos"), f"{p}/k_pos", tag)
    H = S.get("H1")
    if H:
        comp_rows("R.H1.AGG", H, "/H1", T["H1"], {f"{a}_vs_original": f"{a.upper()}vsORIG" for a in ARCH5})
    H2 = S.get("H2") or {}
    if H2.get("primary"):
        comp_rows("R.H2.AGG", H2["primary"], "/H2/primary", T["H2"], {"H2_primary": "ARvsTSR"})
        if H2.get("gate_open"):
            comp_rows("R.H2s.AGG", H2["secondary"], "/H2/secondary", T["H2s"],
                      {"H2a_LSTM": "LSTM_ARvsTSR", "H2b_MHA": "MHA_ARvsTSR"})
    H3 = S.get("H3") or {}
    if H3.get("family"):
        comp_rows("R.H3.AGG", H3["family"], "/H3/family", T["H3"], {"FUS_vs_KMD_f": "FUSvsKMD", "FUS_vs_MVD_f": "FUSvsMVD"})
    H4 = S.get("H4")
    if H4:
        comp_rows("R.H4.AGG", H4, "/H4", T["H4"], {f"MLP_vs_{a}": f"MLPvs{a.upper()}" for a in ARCH})
    R = S.get("R14") or {}
    sp = R.get("spearman") or {}
    tr14 = T["r14_late"] if R.get("confirmatory") is False else T["R14"]
    add("R.R14.AGG.D_DELTA.rho", sp.get("rho"), "/R14/spearman/rho", tr14)
    add("R.R14.AGG.D_DELTA.p", sp.get("p"), "/R14/spearman/p", tr14)
    add("R.R14.AGG.D_DELTA.n", R.get("n"), "/R14/n", tr14)
    add("R.R14.KNAD.D_DELTA.rho", (R.get("rho_KNAD") or {}).get("rho"), "/R14/rho_KNAD/rho",
        T_NUM["rho_knad"] if tr14 == T["R14"] else tr14)
    for x, e in (((S.get("descriptor_D") or {}).get("by_dataset")) or {}).items():
        add(f"R.R14.{x}.DESC.D", e.get("D"), f"/descriptor_D/by_dataset/{x}/D",
            f"[CONFIRMATORIO plan §6.1] (calculado {e.get('created')})")
    om = ((S.get("H1_descriptive") or {}).get("omnibus") or {})
    add("R.OMNI.AGG.ID.F", om.get("F"), "/H1_descriptive/omnibus/F", T_NUM["omni"])
    add("R.OMNI.AGG.ID.pPerm", om.get("p_permutation"), "/H1_descriptive/omnibus/p_permutation", T_NUM["omni"])
    A = S.get("abstract") or {}
    for k, idd in (("1_control_auroc", "R.ABS.AGG.C10.AUROC"), ("2_context8_auroc", "R.ABS.AGG.CTX8.AUROC")):
        for s in ("median", "min", "max"):
            add(f"{idd}_{s}", (A.get(k) or {}).get(s), f"/abstract/{k}/{s}", T_NUM["abs"])
    add("R.ABS.AGG.C10.AUCLOCRAW_median", A.get("1_control_auc_loc_raw_median"), "/abstract/1_control_auc_loc_raw_median",
        T_NUM["abs"])
    add("R.ABS.AGG.CTX8.AUCLOCRAW_median", A.get("2_context8_auc_loc_raw_median"),
        "/abstract/2_context8_auc_loc_raw_median", T_NUM["abs"])
    for x, row in ((S.get("R110") or {}).get("heatmap_auroc_fus") or {}).items():
        for c, v in row.items():
            add(f"R.T1.{x}.{c}.AUROC_FUS", v, f"/R110/heatmap_auroc_fus/{x}/{c}", T_NUM["t1"])
    lines = ["| ID | Valor | Ubicación | Fuente (ruta o filtro de runs.csv) | Script + commit | Comando para regenerar |",
             "|---|---|---|---|---|---|"]
    for idd, v, src in rows:
        tail = " [ORACULO: ...]" if "ORACULO" in src else ""
        lines.append(f"| {idd} | {v} | - | {src}{tail} | {script_ref} | `{cmd}` |")
    return lines


def attainability_tables():
    from scipy import stats as ss
    ns = [3, 4, 5, 6, 7, 8, 9]
    res = {"wilcoxon_p": {}, "min_p_two_sided": {}, "holm": {}, "hl_ci": {}, "sign_test": {}, "spearman": {},
           "iman_davenport": []}
    for n in ns:
        maxs = n * (n + 1) // 2
        res["wilcoxon_p"][str(n)] = {str(w): float(p_two_sided_no_ties(n, w)) for w in range(0, min(maxs // 2, 12) + 1)}
        res["min_p_two_sided"][str(n)] = float(p_two_sided_no_ties(n, 0))
        res["hl_ci"][str(n)] = hl_ci_orders(n)
    for n in [9, 8, 7, 6, 5, 4, 3]:
        for m in [1, 2, 4, 5]:
            res["holm"][f"n{n}_m{m}"] = holm_attainability(n, m)
    for n in [6, 8, 9]:
        res["sign_test"][str(n)] = {str(k): sign_test_p(n, k) for k in range((n + 1) // 2, n + 1)}
        res["spearman"][str(n)] = spearman_null_table(n)
    for k in [11, 5]:
        for n in [9, 8, 6]:
            df1, df2 = k - 1, (k - 1) * (n - 1)
            res["iman_davenport"].append({"k": k, "n": n, "df1": df1, "df2": df2,
                                          "F_crit_0.05": float(ss.f.ppf(1 - ALPHA, df1, df2))})
    res["n_min_holm"] = {str(m): n_min_holm(m) for m in (1, 2, 4, 5)}
    return res


def rank_autocorr_lag1(col):
    """rho1 de una feature: Pearson entre rangos medios consecutivos. None si constante o indefinida."""
    from scipy.stats import rankdata
    col = np.asarray(col, dtype=np.float64)
    if np.isnan(col).any():
        raise StatsError("NaN en las features de D (el pipeline debe asignarles un valor antes)")
    if len(col) < 3 or np.all(col == col[0]):
        return None, "constante"
    r = rankdata(col, method="average")
    a, b = r[:-1], r[1:]
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if den == 0.0:
        return None, "indefinida (rangos constantes en un extremo)"
    return float((a * b).sum() / den), None


def descriptor_from_matrix(X, names):
    rhos, const, undef = {}, [], []
    for j, nm in enumerate(names):
        r, why = rank_autocorr_lag1(X[:, j])
        if r is None:
            (const if why == "constante" else undef).append(nm)
        else:
            rhos[nm] = r
    Dv = float(np.median(list(rhos.values()))) if rhos else None
    return {"D": Dv, "n_features_total": len(names), "n_features_used": len(rhos), "n_constant_excluded": len(const),
            "constant_features": const, "undefined_features": undef, "rho1_by_feature": rhos}


def _knad_cache_features(did, lc_entry):
    cache = C.CACHE_ROOT / f"{C.ds_slug(DS_NAME[did])}__corrected__v1"
    if not (cache / "DONE").exists():
        raise StatsError(f"{did}: falta la cache corrected {cache} (R-6): D se calcula cuando exista")
    meta = C.read_json(cache / "meta.json") or {}
    if meta.get("test_mode") or meta.get("pipeline") != "corrected":
        raise StatsError(f"{did}: cache {cache} no es de produccion corrected")
    x = np.load(cache / "x.npy", mmap_mode="r")
    return x, cache, meta


def descriptor_dataset(did, lc, with_e41=False, log=print):
    """Descriptor D of one dataset (plan §6.1)."""
    import pipelines as PL
    t0 = time.time()
    e = lc.get(DS_KEY[did]) or {}
    calib = [int(v) for v in e["calibration"]]
    T_, N_ = int(e["T_train_end"]), int(e["N_model_stream"])
    rec = {"dataset": DS_NAME[did], "rows": calib, "created": None, "rule": "mediana_j rho1_j (rangos medios de la "
           "calibracion en orden del stream; features constantes excluidas), §6.1"}
    if did in KNAD:
        x, cache, meta = _knad_cache_features(did, e)
        if x.shape[0] != N_:
            raise StatsError(f"{did}: filas de la cache {x.shape[0]} != label_counts {N_}")
        names = [f"f{j}" for j in range(x.shape[1])]
        Xc = np.array(x[calib[0]:calib[1], :], dtype=np.float64)
        rec.update(source=f"cache {cache.name}/x.npy (AfterImage crudas; sin normalizar)",
                   cache_x_sha256_meta=(meta.get("x") or {}).get("sha256"), n_nan=int(np.isnan(Xc).sum()),
                   n_posinf=int(np.isposinf(Xc).sum()), n_neginf=int(np.isneginf(Xc).sum()),
                   nan_policy="KNAD: el pipeline corrected no asigna valor a NaN; con NaN se aborta")
        segs = {"train": (0, T_), "eval": (calib[1], N_)}
        full = x
        num_names = names
    else:
        spec = PL.cli_spec(DS_NAME[did])
        cache = C.CACHE_ROOT / f"{C.ds_slug(DS_NAME[did])}__corrected__v1"
        if not (cache / "DONE").exists():
            raise StatsError(f"{did}: falta la cache corrected {cache} (R-6)")
        meta = C.read_json(cache / "meta.json") or {}
        plan = PL.corrected_plan(spec, C.DATA_ROOT, C.DEFAULT_CODE_ROOT, C.CORRECTED_TRAIN_ROWS[DS_NAME[did]])
        xr, raw = PL.corrected_raw_features(spec, plan)
        if xr.shape[0] != N_ or int(plan["T"]) != T_:
            raise StatsError(f"{did}: stream releido N={xr.shape[0]}, T={plan['T']} != label_counts {N_}, {T_}")
        ycache = np.load(cache / "y.npy", mmap_mode="r")
        if not np.array_equal(np.asarray(ycache), plan["y"]):
            raise StatsError(f"{did}: etiquetas del stream releido != y.npy de la cache (desalineado)")
        cats = set(raw["categorical"])
        num_idx = [j for j, c in enumerate(raw["columns"]) if c not in cats]
        num_names = [raw["columns"][j] for j in num_idx]
        prep = ((meta.get("extra") or {}).get("preprocessing") or {})
        if prep.get("numeric_columns") != num_names:
            raise StatsError(f"{did}: columnas numericas releidas != meta de la cache")
        Xc = np.ascontiguousarray(xr[calib[0]:calib[1], :][:, num_idx])
        n_nan = int(np.isnan(Xc).sum())
        Xc[np.isnan(Xc)] = 0.0
        rec.update(source="CSV crudos releidos con pipelines.corrected_raw_features (antes de normalizar y de sustituir "
                          "+-inf)", n_nan_replaced_by_0=n_nan, n_posinf=int(np.isposinf(Xc).sum()),
                   n_neginf=int(np.isneginf(Xc).sum()), nan_policy="NaN -> 0 (fillna(0) del pipeline corrected)",
                   categorical_excluded=sorted(cats))
        z = Xc.copy()
        fill = ((prep.get("inf_to_train_max") or {}).get("fill") or {})
        for j, nm in enumerate(num_names):
            mi = np.isinf(z[:, j])
            if mi.any():
                z[mi, j] = float(fill[nm])
        mu = np.array(prep["zscore"]["mean"], dtype=np.float64)
        sd = np.array(prep["zscore"]["std"], dtype=np.float64)
        z -= mu
        z /= (sd + PL.ZSCORE_EPS)
        xc = np.load(cache / "x.npy", mmap_mode="r")
        ref = np.array(xc[calib[0]:calib[1], :len(num_idx)], dtype=np.float64)
        ok = bool(np.array_equal(z, ref))
        rec["alignment_check"] = {"bitwise_equal_to_cache_x": ok,
                                  "max_abs_diff": float(np.max(np.abs(z - ref))) if z.size else 0.0}
        if not ok:
            raise StatsError(f"{did}: las features releidas no reproducen x.npy de la cache ({rec['alignment_check']})")
        segs = {"train": (0, T_), "eval": (calib[1], N_)}
        full = None
        del z, ref
    rec.update(descriptor_from_matrix(Xc, num_names))
    if with_e41:
        rec["E41"] = {}
        for seg, (a, b) in segs.items():
            if full is not None:
                Xs = np.array(full[a:b, :], dtype=np.float64)
            else:
                Xs = np.ascontiguousarray(xr[a:b, :][:, num_idx])
                Xs[np.isnan(Xs)] = 0.0
            rec["E41"][f"D_{seg}"] = descriptor_from_matrix(Xs, num_names)["D"]
            rec["E41"][f"rows_{seg}"] = [a, b]
            del Xs
        rec["E41"]["tag"] = "[EXPLORATORIO] E4.1"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["created"] = C.now_iso()
    log(f"[{C.now_iso()}] D {did}: {rec['n_features_used']} features, {rec['seconds']} s")
    return rec


def d_code_sha256():
    """Fingerprint of the code that computes D: functions of stats.py plus pipelines.py and common.py."""
    import inspect
    import pipelines as PL
    src = "".join(inspect.getsource(f) for f in (rank_autocorr_lag1, descriptor_from_matrix, _knad_cache_features,
                                                  descriptor_dataset))
    blob = json.dumps({"src": hashlib.sha256(src.encode()).hexdigest(),
                       "pipelines": C.sha256_file(Path(PL.__file__).resolve()),
                       "common": C.sha256_file(Path(C.__file__).resolve())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def main_descriptor(a):
    out = Path(a.out) if a.out else C.RESULTS_ROOT / "stats" / "descriptor_d.json"
    C.check_not_forbidden(out.parent, "--out")
    try:
        prov = base_provenance()
    except (M.PlanGuardError, M.EnvGuardError) as e:
        print(f"ABORT: {e}")
        return 3
    want = [d.strip() for d in a.datasets.split(",")] if a.datasets else DS
    bad = [d for d in want if d not in DS]
    if bad:
        print(f"ERROR: datasets desconocidos {bad}")
        return 2
    lc = C.read_json(M.REPO / M.LABEL_COUNTS_REL)
    old = C.read_json(out) if out.exists() else None
    if out.exists() and not (isinstance(old, dict) and old.get("schema") == SCHEMA_D):
        print(f"ERROR: {out} existe y no es un descriptor_d.json valido: no se sobrescribe")
        return 1
    J = old if isinstance(old, dict) else {"schema": SCHEMA_D, "datasets": {}}
    dcode = d_code_sha256()
    rc = 0
    for did in want:
        prev = (J["datasets"].get(did) or {})
        if prev.get("D") is not None and not a.force:
            print(f"[{C.now_iso()}] SKIP D {did}: ya existe (calculado {prev.get('created')}); append-only, usa --force "
                  "para recalcular (queda en history y la guarda de R1.4 usa la hora del D en uso)", flush=True)
            continue
        try:
            rec = descriptor_dataset(did, lc, with_e41=a.with_e41)
        except Exception as e:
            rc = 1
            print(f"[{C.now_iso()}] FAIL D {did}: {type(e).__name__}: {e}", flush=True)
            continue
        rec.update(stats_sha256=prov["stats_sha256"], d_code_sha256=dcode, git_head=prov.get("git_head"))
        if prev:
            rec["history"] = list(prev.get("history") or []) + [{k: v for k, v in prev.items() if k != "history"}]
        J["datasets"][did] = rec
    J.update(tag="[descriptor R1.4 plan §6.1: solo features, sin etiquetas ni scores]",
             rule="D = mediana sobre features numericas de la autocorrelacion de rangos de retardo 1 en [T, T + C)",
             deviations_proposed=[s for s in ST_DEVIATIONS_PROPOSED if s.startswith(("ST-8", "ST-9"))],
             updated=C.now_iso(), provenance=prov,
             complete=all(((J["datasets"].get(d) or {}).get("D") is not None) for d in DS))
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(out, M._clean(J))
    print(f"[{C.now_iso()}] descriptor_d -> {out} (sha256 {C.sha256_file(out)}; completo: {J['complete']})")
    return rc


def base_provenance():
    return {"stats_version": STATS_VERSION, "stats_sha256": C.sha256_file(Path(__file__).resolve()),
            "stats_path": str(Path(__file__).resolve()), "plan": M.plan_guard(), "versions": M.versions(),
            "env": M.env_guard(), "host": C.hostname(), "git_head": C.head_commit(M.REPO),
            "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def _tests_rows(S):
    rows = []

    def walk(fam_key, fam, confirm):
        for name, c in (fam.get("comparisons") or {}).items():
            w = c.get("wilcoxon") or {}
            ci = c.get("hl_ci") or {}
            rob = c.get("robustness") or {}
            rows.append({"family": fam_key, "comparison": name, "contrast": c.get("contrast"), "n": c.get("n"),
                         "n_eff": w.get("n_eff"), "ties": w.get("ties"), "W": w.get("W"), "T_plus": w.get("t_plus"),
                         "T_minus": w.get("t_minus"), "p_exact": w.get("p"), "p_enum": w.get("p_enum"),
                         "method": w.get("method"), "p_holm": c.get("p_holm"), "reject": c.get("reject"),
                         "status": c.get("status"), "hl": c.get("hl"), "ci_lo": ci.get("lower"),
                         "ci_hi": ci.get("upper"), "ci_coverage": ci.get("coverage"),
                         "r_rank_biserial": w.get("r_rank_biserial"), "k_pos": c.get("k_pos"),
                         "k_zero": c.get("k_zero"), "k_neg": c.get("k_neg"), "equivalent_rope": c.get("equivalent_rope"),
                         "drift_robust": (rob.get("drift") or {}).get("robust"),
                         "dependence_ok": ((rob.get("dependence") or {}).get("rule") or {}).get("dependence_ok"),
                         "wording": (c.get("wording") or {}).get("rule"), "robust": c.get("robust"),
                         "abstract_eligible": c.get("abstract_eligible"), "tag": fam.get("tag"),
                         "confirmatory": confirm})
    for key, confirm in (("H1", True), ("H4", True)):
        if S.get(key):
            walk(key, S[key], confirm)
    if (S.get("H2") or {}).get("primary"):
        walk("H2", S["H2"]["primary"], True)
        walk("H2-secondary", S["H2"]["secondary"], bool(S["H2"].get("gate_open")))
        walk("H2-conv", S["H2"]["conv_descriptive"], False)
    if (S.get("H3") or {}).get("family"):
        walk("H3", S["H3"]["family"], True)
    for sname, blk in (S.get("sensitivities") or {}).items():
        for hk in ("H1", "H3", "H4"):
            if blk.get(hk):
                walk(f"{sname}:{hk}", blk[hk] if hk != "H3" else blk[hk]["family"], False)
        if (blk.get("H2") or {}).get("primary"):
            walk(f"{sname}:H2", blk["H2"]["primary"], False)
    return rows


TESTS_HEADER = ["family", "comparison", "contrast", "n", "n_eff", "ties", "W", "T_plus", "T_minus", "p_exact", "p_enum",
                "method", "p_holm", "reject", "status", "hl", "ci_lo", "ci_hi", "ci_coverage", "r_rank_biserial",
                "k_pos", "k_zero", "k_neg", "equivalent_rope", "drift_robust", "dependence_ok", "wording", "robust",
                "abstract_eligible", "tag", "confirmatory"]


SCHEMA_X13 = "jisa-x13-claims/1"
X13_FAMILIES = ("H1", "H2", "H2s", "H3", "H4", "H4_claim", "R14")


def load_x13_claims(path):
    """F-08: tabla fijada ANTES de stats.py run que asigna cada afirmacion enviada (TEX:linea) a una comparacion y a
    un signo. JSON {"schema": "jisa-x13-claims/1", "claims": [{"id", "tex", "text", "family", "comparison",
    "claimed_sign": +1|-1}]}; family en X13_FAMILIES (H4_claim = 'MLP achieves the highest performance', sin
    comparison; H2s = H2.a/H2.b)."""
    if path is None:
        return None, None
    J = C.read_json(path)
    if not isinstance(J, dict) or J.get("schema") != SCHEMA_X13 or not isinstance(J.get("claims"), list):
        raise StatsError(f"{path}: no es una tabla {SCHEMA_X13}")
    for cl in J["claims"]:
        if cl.get("family") not in X13_FAMILIES or cl.get("claimed_sign") not in (1, -1) or not cl.get("id") or \
                not cl.get("tex") or (cl.get("family") not in ("H4_claim", "R14") and not cl.get("comparison")):
            raise StatsError(f"{path}: afirmacion invalida {cl}")
    rel = None
    try:
        rel = str(Path(path).resolve().relative_to(M.REPO))
    except ValueError:
        pass
    git = None
    if rel:
        rc, out, _ = M._git_bytes(M.REPO, "log", "-1", "--format=%H %cI", "--", rel)
        txt = out.decode().strip() if rc == 0 else ""
        rc2, _o, _e = M._git_bytes(M.REPO, "diff", "--quiet", "HEAD", "--", rel)
        git = {"last_commit": txt or None, "clean_vs_HEAD": rc2 == 0}
    fixed = bool(git and git.get("last_commit") and git.get("clean_vs_HEAD"))
    return J, {"path": str(path), "sha256": C.sha256_file(path), "git": git, "fixed_in_git": fixed,
               "rule": "F-08: la tabla debe estar en git y sin cambios antes de stats.py run; si no, los estados X13 "
                       "no se citan"}


def x13_summary_block(S, claims, claims_prov=None, chrono=None):
    """F-08 (§11 X13): resumen X13. Un estado por afirmacion solo si la tabla esta fijada en git (commit y sin cambios);
    si no, se calculan pero quedan 'no fijada' y citable = false."""
    h4c = ((S.get("H4") or {}).get("claim_mlp_highest") or {}).get("x13")
    if claims is None:
        return {"status": "not assigned: the table of claims (--x13-claims, F-08) was not given, and without it no "
                          "X13 state is assigned per claim",
                "claims": None, "H4_claim_mlp_highest": h4c,
                "rule": "§11 X13; los campos x13_if_claimed_* de cada comparacion son intermedios y no se citan"}
    fixed = bool((claims_prov or {}).get("fixed_in_git"))
    rows = x13_from_claims(S, claims.get("claims") or [])
    for r in rows:
        r["citable"] = fixed
    before = None
    lc = (((claims_prov or {}).get("git") or {}).get("last_commit") or "").split(" ")
    first = (chrono or {}).get("first_metrics_json_created")
    if len(lc) == 2 and first:
        try:
            before = _aware(lc[1]) < _aware(first)
        except (TypeError, ValueError):
            before = None
    return {"status": ("ok" if fixed else "no fijada en git (F-08): la tabla no esta en un commit o tiene cambios; "
                                          "estados calculados pero no citables"),
            "fixed_in_git": fixed, "committed_before_first_metrics_json": before, "claims": rows,
            "H4_claim_mlp_highest": h4c,
            "rule": "§11 X13 con la tabla de afirmaciones fijada antes de la ejecucion (F-08)"}


def x13_from_claims(S, claims):
    """F-08 (§11 X13): un unico estado por afirmacion enviada, con la comparacion y el signo fijados en la tabla."""
    out = []
    for cl in claims:
        fam, comp, sign = cl["family"], cl.get("comparison"), int(cl["claimed_sign"])
        rec = {k: cl.get(k) for k in ("id", "tex", "text", "family", "comparison", "claimed_sign")}
        c = None
        if fam == "H4_claim":
            rec["x13"] = ((S.get("H4") or {}).get("claim_mlp_highest") or {}).get("x13")
            out.append(rec)
            continue
        if fam == "R14":
            R = S.get("R14") or {}
            if R.get("spearman") and R.get("confirmatory"):
                rho = R["spearman"].get("rho")
                rec["x13"] = x13_status(bool(R.get("reject")), rho, bool(R.get("sequential_dependence_sentence_allowed")),
                                        sign)
            else:
                rec["x13"] = None
                rec["note"] = f"R1.4 {R.get('status') or 'no confirmatorio'}"
            out.append(rec)
            continue
        fam_d = {"H1": S.get("H1"), "H2": (S.get("H2") or {}).get("primary"), "H2s": (S.get("H2") or {}).get("secondary"),
                 "H3": (S.get("H3") or {}).get("family"), "H4": S.get("H4")}.get(fam) or {}
        c = (fam_d.get("comparisons") or {}).get(comp)
        if c is None:
            rec["x13"] = None
            rec["note"] = "comparacion no disponible (bloqueada o nombre desconocido)"
        else:
            sig = bool(c.get("reject")) and not (fam == "H2s" and not (S.get("H2") or {}).get("gate_open"))
            rec["x13"] = x13_status(sig, c.get("hl"), bool(c.get("robust")), sign)
            rec["hl"], rec["reject"], rec["robust"] = c.get("hl"), c.get("reject"), c.get("robust")
        out.append(rec)
    return out


def _sens_rewrite(blk, S):
    """Apply the sensitivity wording to the sensitivity families (never 'significant' or 'outperforms')."""
    pairs = []
    for hk in ("H1", "H4"):
        if blk.get(hk) and S.get(hk):
            pairs.append((S[hk]["comparisons"], blk[hk]["comparisons"]))
    if (blk.get("H3") or {}).get("family") and (S.get("H3") or {}).get("family"):
        pairs.append((S["H3"]["family"]["comparisons"], blk["H3"]["family"]["comparisons"]))
    if (blk.get("H2") or {}).get("primary") and (S.get("H2") or {}).get("primary"):
        pairs.append((S["H2"]["primary"]["comparisons"], blk["H2"]["primary"]["comparisons"]))
        pairs.append((S["H2"]["secondary"]["comparisons"], blk["H2"]["secondary"]["comparisons"]))
    for mc, sc in pairs:
        for k, c in sc.items():
            c["wording"] = sensitivity_wording(c, mc.get(k))
            c["abstract_eligible"] = False


def run_stats(agg, Dj, lc, friedman_b=FRIEDMAN_B, claims=None, claims_prov=None):
    """Devuelve el dict de stats.json (sin procedencia)."""
    VS = agg["value_sets"]
    cells = agg["cells"]
    V = VS["main"]
    trivial = {x: (agg["datasets"].get(x) or {}).get("trivial") or {} for x in DS}
    S = {"schema": SCHEMA_STATS, "created": C.now_iso(), "alpha": ALPHA, "rope": ROPE,
         "S_star": {x: (agg["datasets"].get(x) or {}).get("S_star") for x in DS},
         "S_star_sizes": {x: len((agg["datasets"].get(x) or {}).get("S_star") or []) for x in DS}}
    ss_ = [S["S_star_sizes"][x] for x in DS]
    S["seeds_sentence_rule"] = {"min_S_star": min(ss_), "n_datasets_with_3": sum(1 for v in ss_ if v == 3),
                                "rule": "§4 w2: 'three' solo si |S*_d| = 3 en los 9 datasets"}
    S["rope_vs_noise"] = {"rope": ROPE, "median_sqrt2_sd_pool": _median(
        [((agg["datasets"].get(x) or {}).get("sd_pool") or {}).get("sqrt2_sd_pool") for x in DS]),
        "note": "ROPE fijado por convencion; se reporta junto a la mediana de sqrt(2) SD_pool(d) (§5.2, m4)"}
    S["H1"] = h1(V)
    S["H1_descriptive"] = h1_descriptive(V, int(friedman_b))
    S["H3"] = h3(V, cells)
    S["H4"] = h4(V, h1fam=S["H1"])
    needed = [x for x in DS if any(delta_h2(VS[k], x) is not None for k in ("main", "s_itt", "s_div_noimp", "s_seed")
                                   if isinstance(VS.get(k), dict))]
    ok, guard = d_guard(Dj, needed, agg.get("chronology_R7"))
    S["D_guard"] = guard
    if Dj:
        S["descriptor_D"] = {"by_dataset": {x: {"D": ((Dj.get("datasets") or {}).get(x) or {}).get("D"),
                                                "created": ((Dj.get("datasets") or {}).get(x) or {}).get("created")}
                                            for x in DS},
                             "stats_sha256_at_computation": ((Dj.get("provenance") or {}).get("stats_sha256")),
                             "tag": "[CONFIRMATORIO plan §6.1] descriptor de R1.4 (solo features)"}
    if ok:
        S["H2"] = h2(V)
        S["R14"] = r14(V, Dj, S["H2"], lc, cells, agg.get("ablations"))
        if not guard.get("r14_confirmatory"):
            S["R14"]["tag"] = T["r14_late"]
            S["R14"]["confirmatory"] = False
            S["R14"]["guard"] = {k: guard.get(k) for k in ("D_after_metrics", "not_verifiable", "d_code_consistent")}
        else:
            S["R14"]["confirmatory"] = True
    else:
        S["H2"] = {"status": "bloqueado (ST-8)", "guard": guard}
        S["R14"] = {"status": "bloqueado (ST-8)", "guard": guard}
    sens = {}
    for sname, tag in (("s_itt", T["sens_itt"]), ("s_div_noimp", T["sens_div"]), ("s_seed", T["sens_seed"])):
        Vs = VS.get(sname)
        if sname == "s_seed" and not (VS.get("_meta") or {}).get("s_seed_computable"):
            sens[sname] = {"status": "no calculable (algun |S*_d| = 0, AG-7)", "tag": tag}
            continue
        blk = {"tag": tag, "H1": h1(Vs, robust=False, tag=tag, confirmatory=False),
               "H3": h3(Vs, cells, robust=False, tag=tag, confirmatory=False),
               "H4": h4(Vs, robust=False, tag=tag, confirmatory=False)}
        if ok:
            blk["H2"] = h2(Vs, robust=False, tag=tag, tag_s=tag, confirmatory=False)
        _sens_rewrite(blk, S)
        sens[sname] = blk
    S["sensitivities"] = sens
    S["ablations"] = ablations_summary(agg.get("ablations"), agg["datasets"])
    S["T4P"] = t4p_summary(cells)
    S["failure_cases"] = failure_cases(V, cells, Dj)
    if ok:
        S["abstract"] = abstract_numbers(V, S["H1"], S["H2"], trivial)
    else:
        S["abstract"] = {"status": "incompleto: H2 bloqueado (ST-8)", "partial": abstract_numbers(
            V, S["H1"], {"primary": {"comparisons": {"H2_primary": {}}}}, trivial)}
    S["R110"] = r110(V)
    S["oracles"] = {"O2_gap_vs_calibrated_f1": {X: dict(median_iqr(
        [_cm(cells, x, c, f"o2_gap_{X}") for x in DS for c in CONFIGS]), tag=T["oracle_o2"],
        rule="max F1 sobre todos los umbrales en E - F1 calibrado, por (dataset, configuracion); mediana e IQR (§3.6)")
        for X in ("FUS", "KMD_f", "MVD_f")},
        "O1": {"rows": agg.get("o1") or [], "tag": T["o1"],
               "note": "tabla X13 'Changes to reported results' (solo semilla A)"}}
    S["mlp_noise_reference"] = {"by_dataset": agg.get("mlp_noise"), "tag": "[SECUNDARIA plan §3.2]"}
    S["x13_summary"] = x13_summary_block(S, claims, claims_prov, agg.get("chronology_R7"))
    S["S_s0_KNAD"] = {"status": "no implementada en esta version (ST-21); no se sustituye por otro analisis",
                      "plan": "§4.2", "tag": "[SENSIBILIDAD plan §4.2 S-s0-KNAD] no ejecutada"}
    S["exploratory_counts"] = exploratory_counts(S, cells)
    return S


def exploratory_counts(S, cells):
    """Number of exploratory analyses of each kind (plan §12.1), declared in every table."""
    days = sum(1 for x in DS for c in CONFIGS if any(k.startswith("day_") and k.endswith("_auroc_FUS")
                                                     for k in (((cells.get(x) or {}).get(c) or {}).get("mean") or {})))
    types = sum(1 for x in DS for c in CONFIGS if any(k.startswith("type_") for k in
                                                      (((cells.get(x) or {}).get(c) or {}).get("mean") or {})))
    dupcal = sum(1 for x in DS for c in CONFIGS if (((cells.get(x) or {}).get(c) or {}).get("mean") or {})
                 .get("auroc_FUS_dupcal") is not None)
    r14 = (S.get("R14") or {}).get("exploratory") or {}
    return {"R14_correlations": {"N_expl_fixed": N_EXPL_R14, "N_computed": r14.get("N_computed")},
            "per_day_cells_with_auc": days, "per_attack_type_cells": types, "S_DUPCAL_cells": dupcal,
            "tag": T["expl"], "rule": "§12.1: cada tabla indica cuantos analisis exploratorios de su tipo se hicieron"}


def _prefix_tags(o, pre):
    """Put the run-type mark in front of every 'tag' label of stats.json."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "tag" and isinstance(v, str) and not v.startswith(pre):
                o[k] = f"{pre} {v}"
            else:
                _prefix_tags(v, pre)
    elif isinstance(o, list):
        for v in o:
            _prefix_tags(v, pre)
    return o


def run_type(agg):
    """Run type (plan §4.5): FINAL only after common.FREEZE, with a complete aggregate and the default freeze of the plan."""
    why = []
    if not (C.now() > C.FREEZE):
        why.append(f"antes de la congelacion ({C.FREEZE.isoformat(timespec='seconds')})")
    if not agg.get("complete"):
        why.append("aggregate.complete = false")
    if not (agg.get("freeze") or {}).get("is_plan_default"):
        why.append("aggregate sin la congelacion del plan (--freeze distinto o ausente)")
    return ("INTERIM" if why else "FINAL"), why


def main_run(a):
    out = Path(a.out_dir) if a.out_dir else C.RESULTS_ROOT / "stats"
    C.check_not_forbidden(out, "--out-dir")
    try:
        prov = base_provenance()
    except (M.PlanGuardError, M.EnvGuardError) as e:
        print(f"ABORT: {e}")
        return 3
    aggp = Path(a.aggregate) if a.aggregate else C.RESULTS_ROOT / "aggregate" / "aggregate.json"
    if not aggp.exists():
        print(f"ERROR: falta {aggp} (ejecuta aggregate.py)")
        return 1
    agg = C.read_json(aggp)
    if not isinstance(agg, dict) or agg.get("schema") != "jisa-aggregate/1":
        print(f"ERROR: {aggp} no es un aggregate.json valido")
        return 1
    if agg.get("consistency_errors"):
        print(f"ERROR: aggregate.json con errores de consistencia (AG-10): {agg['consistency_errors']}")
        return 1
    if ((agg.get("provenance") or {}).get("plan") or {}).get("plan_sha256") != prov["plan"]["plan_sha256"]:
        print("ERROR: aggregate.json calculado con otro plan")
        return 1
    leaked = sorted({c for d in (agg.get("cells") or {}).values() for c in d if c in EXCLUDED_CONFIGS}
                    | {c for s in (agg.get("value_sets") or {}).values() if isinstance(s, dict)
                       for d in s.values() if isinstance(d, dict) for c in d if c in EXCLUDED_CONFIGS})
    if leaked:
        print(f"ERROR: aggregate.json mezcla configuraciones exploratorias en las celdas del plan: {leaked}")
        return 1
    root = aggp.parent.parent
    runs_p = root / "runs.csv"
    outs = agg.get("outputs_sha256") or {}
    ins = {"aggregate_json": str(aggp), "aggregate_json_sha256": C.sha256_file(aggp),
           "runs_csv": str(runs_p), "runs_csv_sha256": C.sha256_file(runs_p) if runs_p.exists() else None,
           "cells_csv_sha256": C.sha256_file(aggp.parent / "cells.csv") if (aggp.parent / "cells.csv").exists() else None}
    if outs.get("runs.csv") and ins["runs_csv_sha256"] != outs["runs.csv"]:
        print("ERROR: runs.csv no coincide con el sha256 registrado en aggregate.json")
        return 1
    dpath = Path(a.descriptor) if a.descriptor else out / "descriptor_d.json"
    try:
        Dj = load_descriptor(dpath)
    except StatsError as e:
        print(f"ERROR: {e}")
        return 1
    ins["descriptor_d"] = str(dpath) if Dj else None
    ins["descriptor_d_sha256"] = C.sha256_file(dpath) if Dj else None
    ins["label_counts_sha256"] = C.sha256_file(M.REPO / M.LABEL_COUNTS_REL)
    lc = C.read_json(M.REPO / M.LABEL_COUNTS_REL)
    try:
        claims, claims_prov = load_x13_claims(a.x13_claims)
    except StatsError as e:
        print(f"ERROR: {e}")
        return 1
    ins["x13_claims"] = claims_prov
    rtype, rwhy = run_type(agg)
    t0 = time.time()
    try:
        S = run_stats(agg, Dj, lc, friedman_b=a.friedman_b, claims=claims, claims_prov=claims_prov)
    except StatsError as e:
        print(f"ERROR (R-5): {e}")
        return 1
    S["run_type"] = rtype
    S["run_type_reasons"] = rwhy
    if rtype == "INTERIM":
        _prefix_tags(S, T["interim"])
        S["interim_note"] = ("run on partial inputs (ST-M4): a look at the results that the plan does not schedule; "
                             "no figure of it is reported or conditions the contingency sequence or the extension "
                             "(§13.3 'sin mirar resultados')")
    S["seconds"] = round(time.time() - t0, 2)
    S["aggregate_complete"] = agg.get("complete")
    S["aggregate_warnings"] = agg.get("warnings")
    S["deviations_proposed"] = ST_DEVIATIONS_PROPOSED
    S["provenance"] = dict(prov, inputs=ins, options={"friedman_b": a.friedman_b},
                           chronology_R7=agg.get("chronology_R7"))
    out.mkdir(parents=True, exist_ok=True)
    sp, tp, np_, ep = out / "stats.json", out / "tests.csv", out / "numbers_proposed.md", out / "env.json"
    C.write_json_atomic(sp, M._clean(S))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(TESTS_HEADER)
    import aggregate as AG
    for r in _tests_rows(S):
        w.writerow([AG.csv_cell(r.get(h)) for h in TESTS_HEADER])
    C.write_text_atomic(tp, buf.getvalue())
    cmd = ("nice -n 19 env OMP_NUM_THREADS=4 envs/postproc/bin/python revision/experiments/stats.py run"
           + (f" --aggregate {aggp}" if a.aggregate else "") + (f" --out-dir {out}" if a.out_dir else ""))
    ref = f"revision/experiments/stats.py sha256:{prov['stats_sha256'][:12]} @ {str(prov.get('git_head') or '')[:12]}"
    head = ("# Rows of the table of numbers (stats.py)"
            if rtype == "FINAL" else
            "# PARTIAL INPUTS: run before the freeze of the results or with an incomplete aggregate (ST-M4). None of "
            "these rows goes to the table of numbers")
    lines = [head, "",
             f"- Generado: {S['created']}; tipo {rtype} {rwhy}; stats.json sha256 {C.sha256_file(sp)}; runs.csv sha256 "
             f"{ins['runs_csv_sha256']}.", "- Ubicación: no se registra en este fichero.", ""]
    lines += numbers_rows(S, cmd, ref, prefix=(T["interim"] + " ") if rtype == "INTERIM" else "")
    C.write_text_atomic(np_, "\n".join(lines) + "\n")
    C.write_json_atomic(ep, {"versions": prov["versions"], "env": prov["env"], "host": prov["host"],
                             "threads_env": prov["threads_env"], "created": C.now_iso()})
    sums = {p.name: C.sha256_file(p) for p in (sp, tp, np_, ep)}
    sums.update({f"input:{k}": v for k, v in ins.items() if k.endswith("_sha256") and v})
    C.write_text_atomic(out / "SHA256SUMS", "".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
    entry = {"created": S["created"], "run_type": rtype, "reasons": rwhy, "stats_json_sha256": sums["stats.json"],
             "aggregate_json_sha256": ins["aggregate_json_sha256"], "S_star": S["S_star"],
             "stats_sha256": prov["stats_sha256"], "D_guard": S["D_guard"].get("status")}
    fd = os.open(str(out / "runs_log.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (json.dumps(M._clean(entry), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    print(f"[{C.now_iso()}] stats -> {out} ({S['seconds']} s; tipo {rtype}; D_guard {S['D_guard']['status']}; "
          f"aggregate completo {agg.get('complete')})", flush=True)
    return 0


def main_attain(a):
    out = Path(a.out) if a.out else None
    res = attainability_tables()
    txt = json.dumps(res, indent=1, ensure_ascii=False)
    if out:
        C.check_not_forbidden(out.parent, "--out")
        C.write_text_atomic(out, txt + "\n")
        print(f"-> {out}")
    else:
        print(txt[:2000])
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--aggregate", default=None)
    r.add_argument("--descriptor", default=None)
    r.add_argument("--out-dir", default=None)
    r.add_argument("--friedman-b", type=int, default=FRIEDMAN_B)
    r.add_argument("--x13-claims", default=None, help="tabla de afirmaciones enviadas (F-08; JSON jisa-x13-claims/1)")
    d = sub.add_parser("descriptor-d")
    d.add_argument("--datasets", default=None)
    d.add_argument("--with-e41", action="store_true")
    d.add_argument("--out", default=None)
    d.add_argument("--force", action="store_true", help="recalcula un D existente (queda en history; ST-M3)")
    t = sub.add_parser("attainability")
    t.add_argument("--out", default=None)
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["run"] + argv
    a = ap.parse_args(argv)
    for k in ("aggregate", "descriptor", "out_dir", "out", "x13_claims"):
        v = getattr(a, k, None)
        if v is not None and not os.path.isabs(v):
            print(f"ERROR: rutas absolutas: {v}")
            return 2
    if a.cmd == "run":
        if a.friedman_b != FRIEDMAN_B:
            print(f"AVISO: --friedman-b {a.friedman_b} != {FRIEDMAN_B} del plan (solo pruebas)")
        return main_run(a)
    if a.cmd == "descriptor-d":
        return main_descriptor(a)
    if a.cmd == "attainability":
        return main_attain(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
