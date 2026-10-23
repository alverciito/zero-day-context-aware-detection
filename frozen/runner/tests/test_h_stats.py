#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_h_stats.py - tests of aggregate.py and stats.py (analysis plan §4, §5, §6, §7.1, §12 and §14.3) on synthetic metrics."""
import argparse
import csv
import itertools
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent
sys.path.insert(0, str(RUNNER))
sys.path.insert(0, str(HERE))
import numpy as np
import common as C
import metrics_rev as M
import aggregate as AG
import stats as ST
from _util import Checker, make_root

REPO = M.REPO
ATT = json.loads((REPO / "revision/analysis/plan_support/attainability.json").read_text())
PLAN_SHA = M.plan_guard()["plan_sha256"]
DATASETS = AG.DATASETS


def _cmp(a, b, path, tol, bad):
    if isinstance(b, dict):
        for k in b:
            if k not in a:
                bad.append((path + "/" + str(k), "falta"))
            else:
                _cmp(a[k], b[k], f"{path}/{k}", tol, bad)
    elif isinstance(b, list):
        if len(a) != len(b):
            bad.append((path, "longitud"))
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp(x, y, f"{path}[{i}]", tol, bad)
    elif isinstance(b, float) or isinstance(a, float):
        if a is None or b is None or abs(a - b) > tol:
            bad.append((path, a, b))
    elif a != b:
        bad.append((path, a, b))


def s1(ck, tmp):
    A = ST.attainability_tables()
    for key in ("wilcoxon_p", "min_p_two_sided", "holm", "hl_ci", "sign_test", "spearman"):
        bad = []
        _cmp(A[key], ATT[key], key, 1e-12, bad)
        ck.check(f"s1.{key}", not bad, f"{len(bad)} discrepancias frente a attainability.json {bad[:3]}")
    bad = []
    _cmp(A["iman_davenport"], ATT["iman_davenport"], "iman_davenport", 1e-9, bad)
    ck.check("s1.iman_davenport", not bad, f"F critico (k, n) = (11|5, 9|8|6) dentro de 1e-9 {bad[:3]}")
    ck.check("s1.n_min_holm", A["n_min_holm"] == {"1": 6, "2": 7, "4": 8, "5": 8}, f"{A['n_min_holm']} (§5.1: 6/7/8/8)")
    ck.check("s1.holm_n9_m5", [r["max_W"] for r in A["holm"]["n9_m5"]] == [1, 2, 2, 3, 5], "W max H1 n = 9: 1/2/2/3/5")
    ck.check("s1.holm_n7_m5", [r["max_W"] for r in A["holm"]["n7_m5"]] == [None, None, 0, 0, 2],
             "n = 7, m = 5: pasos 1-2 inalcanzables (§A.2)")


def s2(ck, tmp):
    from scipy import stats as ss
    rng = np.random.default_rng(20260918)
    worst = 0.0
    n_ties = n_zero = 0
    for it in range(600):
        n = int(rng.integers(1, 10))
        if it % 3 == 0:
            d = rng.normal(0, 0.02, n)
        elif it % 3 == 1:
            d = rng.choice([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.015], size=n)
        else:
            d = np.round(rng.normal(0, 0.02, n), 2)
        r = ST.wilcoxon_test(d)
        if r["p"] is None:
            continue
        bf = ST.wilcoxon_bruteforce(d)
        worst = max(worst, abs(bf - r["p_enum"]), r["abs_diff_enum_scipy"])
        n_ties += bool(r["ties"])
        n_zero += bool(r["n_zeros"])
    ck.check("s2.cross", worst <= 1e-12, f"max |enum - fuerza bruta|, |enum - scipy| = {worst:.2e} en 600 vectores "
                                         f"({n_ties} con empates, {n_zero} con ceros)")
    d = np.array([1.0, 1.0, 3.0, -4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    pe = float(ss.wilcoxon(d, zero_method="wilcox", method="exact").pvalue)
    r = ST.wilcoxon_test(d)
    ok = (r["ties"] and "PermutationMethod" in r["method"] and abs(r["p"] - 0.0234375) < 1e-15 and
          abs(pe - 0.02734375) < 1e-15 and abs(ST.wilcoxon_bruteforce(d) - r["p"]) < 1e-15)
    ck.check("s2.ties_m2", ok, f"empates: scipy exact {pe:.6f} != enumerado {r['p']:.6f} (= fuerza bruta); se usa "
                               f"PermutationMethod")
    r = ST.wilcoxon_test([0.0, 0.0, 0.01, 0.02, -0.03, 0.04, 0.05, 0.06, 0.07])
    ck.check("s2.zeros", r["n_eff"] == 7 and r["n_zeros"] == 2 and "PermutationMethod" in r["method"] and r["testable"],
             f"ceros eliminados (wilcox): n_eff = {r['n_eff']}, metodo permutacion")
    r = ST.wilcoxon_test([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    ck.check("s2.all_pos", r["W"] == 0 and abs(r["p"] - 2 / 512) < 1e-15 and r["r_rank_biserial"] == 1.0 and
             r["t_plus"] == 45.0, f"9/9 positivas: W = 0, p = 2/512, r = 1")
    r = ST.wilcoxon_test([0.0] * 5)
    ck.check("s2.all_zero", r["p"] is None and r["n_eff"] == 0 and not r["testable"], "todas 0 -> sin test")
    r = ST.wilcoxon_test([0.01, -0.02, 0.03, 0.04, 0.05])
    ck.check("s2.not_testable", not r["testable"] and r["p"] is not None, "n_eff = 5 -> not testable (§5.1)")


def s3(ck, tmp):
    d = np.array([0.0, 0.01, -0.02, 0.03, 0.05, 0.02, -0.01, 0.04, 0.06])
    w = sorted((d[i] + d[j]) / 2 for i in range(9) for j in range(i, 9))
    ck.check("s3.hl", ST.hodges_lehmann(d) == float(np.median(w)), "HL = mediana de los 45 promedios de Walsh (ceros "
                                                                   "incluidos)")
    ci = ST.hl_ci(d)
    ck.check("s3.ci9", ci["lower"] == w[5] and ci["upper"] == w[39] and abs(ci["coverage"] - 0.9609375) < 1e-15,
             "n = 9: [W(6), W(40)], cobertura 96,09 %")
    ok = True
    for n, (lo, hi, cov) in {8: (4, 33, 0.9609375), 7: (3, 26, 0.953125), 6: (1, 21, 0.96875),
                             5: (1, 15, 0.9375)}.items():
        o = ST.hl_ci_orders(n)
        ok &= (o["lower_order"], o["upper_order"]) == (lo, hi) and abs(o["coverage"] - cov) < 1e-15
    ck.check("s3.orders", ok, "ordenes n = 8/7/6/5 = §5.2 y §A.2 (n = 5 no alcanza el 95 %)")
    ck.check("s3.n5", not ST.hl_ci_orders(5)["reaches_95"] and ST.hl_ci_orders(6)["reaches_95"], "n = 5: rango con "
                                                                                                   "cobertura 93,75 %")


def s4(ck, tmp):
    adj, rej = ST.holm([0.01, 0.02, None, 0.04, 0.001], 5)
    ck.check("s4.formula", adj == [0.04, 0.06, None, 0.08, 0.005] and rej == [True, False, False, False, True],
             f"p_adj = max_(i<=j) min(1, (m-i+1) p_(i)): {adj}")
    adj, _ = ST.holm([0.03, 0.01], 2)
    ck.check("s4.monotone", adj == [0.03, 0.02], "monotonia de bajada")
    adj, _ = ST.holm([0.5, 0.9], 5)
    ck.check("s4.cap", adj == [1.0, 1.0], "tope 1")
    V = {x: {c: {"auroc_FUS": None, "auc_loc_RAW": None} for c in ST.CONFIGS} for x in ST.DS}
    for i, x in enumerate(ST.DS[:7]):
        for c in ST.CONFIGS:
            V[x][c] = {"auroc_FUS": 0.6 + (0.05 if c != "C10" else 0.0) + 0.001 * i, "auc_loc_RAW": 0.6}
    f = ST.h1(V, robust=False)
    c0 = f["comparisons"]["LSTM_vs_original"]
    ck.check("s4.uninformative", (not f["informative"]) and c0["p_holm"] is None and not c0["reject"] and
             "uninformative" in f["status"] and c0["wilcoxon"]["p"] is not None,
             f"n_eff = 7 en las 5 de H1: {f['status'][:60]}...")
    for x in ST.DS[7:]:
        for c in ST.CONFIGS:
            V[x][c] = {"auroc_FUS": 0.6 + (0.05 if c != "C10" else 0.0), "auc_loc_RAW": 0.6}
    f = ST.h1(V, robust=False)
    ck.check("s4.informative", f["informative"] and all(c["reject"] for c in f["comparisons"].values()) and
             all(abs(c["p_holm"] - 5 * 2 / 512) < 1e-15 for c in f["comparisons"].values()),
             "n = 9, 9/9 positivas: p = 2/512 y Holm 5 x p = 0,0195 en las cinco")


def s5(ck, tmp):
    rng = np.random.default_rng(7)
    worst = 0.0
    for n in (5, 6, 7, 8, 9):
        for _ in range(3):
            x, y = rng.normal(size=n), rng.normal(size=n)
            r = ST.spearman_exact(x, y)
            worst = max(worst, r["abs_diff_p"], r["abs_diff_rho"])
    r = ST.spearman_exact([1, 1, 2, 3, 4, 5, 5, 6, 7], [3, 1, 2, 2, 5, 4, 6, 8, 7])
    worst = max(worst, r["abs_diff_p"], r["abs_diff_rho"])
    ck.check("s5.cross", worst <= 1e-12, f"enumeracion = scipy.permutation_test / spearmanr (con y sin empates): {worst:.1e}")
    r = ST.spearman_exact(np.arange(9.0), np.arange(9.0) ** 2)
    ck.check("s5.min_p", r["rho"] == 1.0 and abs(r["p"] - 2 / math.factorial(9)) < 1e-18,
             f"n = 9, rho = 1: p = 2/9! = {2 / math.factorial(9):.2e} (§A.4)")
    r = ST.spearman_exact([1, 1, 1, 1, 1], [1, 2, 3, 4, 5])
    ck.check("s5.const", r["rho"] is None, "variable constante -> rho no definido")


def s6(ck, tmp):
    from scipy import stats as ss
    rng = np.random.default_rng(3)
    X = rng.normal(size=(9, 11))
    f = ST.friedman_iman_davenport(X, B=20_000)
    chi = float(ss.friedmanchisquare(*X.T).statistic)
    F = (9 - 1) * f["chi2_demsar"] / (9 * 10 - f["chi2_demsar"])
    ck.check("s6.chi2", abs(chi - f["chi2_demsar"]) < 1e-9 and abs(F - f["F"]) < 1e-12,
             "chi2 de Demsar = scipy sin empates; F_ID = (n-1) chi2 / (n(k-1) - chi2)")
    g = ST.friedman_iman_davenport(X, B=20_000)
    ck.check("s6.repro", g["p_permutation"] == f["p_permutation"] and g["count_ge"] == f["count_ge"],
             "p por permutacion reproducible (default_rng(20260917))")
    R = ss.rankdata(X, axis=1)
    rg = np.random.default_rng(ST.PERM_SEED)
    P = rg.permuted(np.broadcast_to((2 * R).astype(np.int16), (20_000, 9, 11)).copy(), axis=2) / 2.0
    Rj = P.sum(axis=1)
    chi2 = 12.0 / (9 * 11 * 12) * (Rj ** 2).sum(axis=1) - 3 * 9 * 12
    Fs = 8 * chi2 / (90 - chi2)
    p2 = (1 + int((Fs >= f["F"] - 1e-12).sum())) / 20_001
    ck.check("s6.direct", abs(p2 - f["p_permutation"]) < 1e-12, f"p = (1 + #F* >= F)/(B + 1) = {p2:.5f}")
    Xt = np.tile(np.arange(11.0), (9, 1))
    ft = ST.friedman_iman_davenport(Xt, B=1000)
    ck.check("s6.perfect", ft["F"] == math.inf and ft["p_F_asymptotic"] == 0.0, "concordancia perfecta: F = inf")


EFF = {"conv1d-AR": 0.03, "conv1d-TSR": 0.03, "conv2d-AR": 0.02, "conv2d-TSR": 0.02, "lstm-AR": 0.05,
       "lstm-TSR": 0.02, "mha-AR": 0.04, "mha-TSR": 0.01, "mlp-AR": 0.08, "mlp-TSR": 0.08, "original": 0.0,
       "stat": -0.02}
VAR_AR = {"conv1d-AR": ("conv1d", True), "conv1d-TSR": ("conv1d", False), "conv2d-AR": ("conv2d", True),
          "conv2d-TSR": ("conv2d", False), "lstm-AR": ("lstm", True), "lstm-TSR": ("lstm", False),
          "mha-AR": ("mha", True), "mha-TSR": ("mha", False), "mlp-AR": ("mlp", True), "mlp-TSR": ("mlp", False),
          "original": ("original", True), "stat": ("stat", True)}
NOISE = {"A": 0.001, "B": -0.001, "C": 0.0005}


def synth_auc(did, rc, seed):
    i = int(did[1:])
    base = 0.60 + 0.01 * i
    extra = 0.002 * i if rc == "lstm-AR" else 0.0
    nz = 0.0 if rc in ("original", "stat") else NOISE[seed] * (1 + (hash((rc,)) % 3 == 0))
    return base + EFF[rc] + extra + nz


def manifest_row(block, did, rc, seed, extra=None):
    var, ar = VAR_AR[rc]
    ds = AG.DS_NAME[did]
    r = {k: "" for k in C.MANIFEST_COLUMNS}
    r.update(block=block, priority="1", dataset=ds, variant=var, ar=str(ar), paradigm="AR" if ar else "TSR",
             pipeline="corrected", pipeline_version="v1", window="800", k_clusters="4", grouping="dbscan",
             partition_seed="NA", warmup_frac="NA", contamination="0", decontaminate="0", seed=seed, rep="1",
             save_ckpt="0", timing_ref="0", notes="sintetico")
    r.update(extra or {})
    r["job_id"] = C.derive_job_id(r)
    return r


def t1_block(seed, did):
    return f"T1-{seed}-{AG.T1_CORPUS_BLOCK.get(did, 'KNAD')}"


def fake_metrics(row, auc, auc_loc=None, divergent=False, section_errors=None, extra_flat=None):
    """metrics.json minimo con el esquema jisa-metrics-rev/1 (solo campos que usan aggregate/stats + estructura)."""
    did = AG.DS_ID[row["dataset"]]
    p = 0.2 + 0.01 * int(did[1:])
    mb = lambda a: {"auroc": a, "auprc": p + 0.1, "apn": 0.1, "eer": 0.4, "threshold": 1.0, "tp": 1, "fp": 1, "tn": 1,
                    "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 2 * p / (1 + p) + 0.01, "fpr": 0.01,
                    "alarm_rate": 0.01, "n_alarms": 2, "f1n": 0.01, "f1_le_trivial_plus_0.02": True,
                    "windowdiff": {"wd": 0.3, "k": 10}}
    E = {"trivial": {"n": 1000, "n_attack": 200, "n_benign": 800, "prevalence": p, "f1_trivial": 2 * p / (1 + p),
                     "ap_trivial": p, "auroc_trivial": 0.5, "auc_t": 0.7, "duration_hours": 1.0},
         "FUS": dict(mb(auc), o2={"f1_max": 0.9, "gap_vs_calibrated_f1": 0.3}),
         "KMD_f": dict(mb(auc - 0.02), o2={"f1_max": 0.8, "gap_vs_calibrated_f1": 0.2}),
         "MVD_f": dict(mb(auc - 0.03), o2={"f1_max": 0.7, "gap_vs_calibrated_f1": 0.1}), "RAW": mb(auc - 0.05)}
    loc = auc if auc_loc is None else auc_loc
    sens = {"S-LOC": {X: {"auc_loc": (loc if X == "RAW" else auc)} for X in AG.X4},
            "S-NF": {"FUS_NF": mb(auc - 0.01), "KMD_raw": mb(auc - 0.03), "MVD_raw": mb(auc - 0.04)},
            "S-MAX": {"FUS_MAX": mb(auc)}, "S-Q95": {X: mb(auc) for X in AG.X4}, "S-Q995": {X: mb(auc) for X in AG.X4},
            "S-PS": {"applicable": False}, "S-DUP": {"applicable": False},
            "S-INS": dict({X: mb(auc - 0.1) for X in AG.X4}, in_sample=True, regions={"candidate": "calib_in_sample_train_tail"}),
            "S-EP": {X: {"wd": 0.2, "k": 100} for X in AG.X4}}
    t4p = {"applicable": row["block"].startswith("T1"),
           "a": {"applicable": did == "D9", "natural_calib_contamination": 0.45, "n_clean": 141968, "estimable": True,
                 "diff_natural_minus_clean": {X: {k: -0.01 for k in ("auroc", "auprc", "threshold", "f1", "precision",
                                                                        "recall", "fpr")} for X in ("FUS", "KMD_f", "MVD_f")},
                 "natural": {X: mb(auc) for X in ("FUS", "KMD_f", "MVD_f")},
                 "clean": {X: mb(auc + 0.01) for X in ("FUS", "KMD_f", "MVD_f")}} if did in ("D9",) else
           ({"applicable": True, "estimable": False, "n_clean": 139, "natural_calib_contamination": 0.97}
            if did == "D7" else {"applicable": False}),
           "b": ({"applicable": True, "E_T4_rows": 900, "trivial_E_T4": {"prevalence": p, "auc_t": 0.7},
                  "levels": [{"c": c, "effective_contamination": c, "metrics": {X: mb(auc - c) for X in ("FUS", "KMD_f", "MVD_f")}}
                             for c in (0.0, 0.01, 0.05, 0.1, 0.2)]} if did in ("D1", "D8") else {"applicable": False}),
           "c": {"trivial_common_eval": {"n": 800, "prevalence": p, "auc_t": 0.7},
                 "fractions": {f: {"candidate": f"calib_first_{int(f):02d}pct_post_train", "effective_contamination": 0.0,
                                   "metrics": {X: mb(auc) for X in ("FUS", "KMD_f", "MVD_f")}} for f in ("5", "10", "20")}}}
    per_day = ({"available": True, "days": {"Tue": {"eligible_auc_per_day": True, "auroc_fus": auc, "auc_t": 0.6, "fpr": 0.01},
                                            "Mon": {"eligible_auc_per_day": False, "fpr": 0.02}}}
               if did == "D8" else {"available": False})
    m = {"schema": M.SCHEMA, "status": "ok", "job_id": row["job_id"], "created": C.now_iso(),
         "job": {"block": row["block"], "family": C.block_family(row["block"]), "dataset": row["dataset"],
                 "dataset_id": did, "variant": row["variant"], "ar": C.parse_bool(row["ar"]), "seed": row["seed"],
                 "pipeline": row["pipeline"], "pipeline_version": row["pipeline_version"]},
         "regions": {"candidate": "calib_first_10pct_post_train"},
         "label_consistency": {"hard": {"regions": True}, "all_soft_match": True},
         "nonfinite": {"divergent": divergent},
         "divergence": ({"imputed_itt": {"auroc_fus": 0.5, "auc_loc_raw": 0.5, "auprc_fus": p, "f1_fus": 0.0}}
                        if divergent else None),
         "primary": {"value": auc, "auc_loc_raw": loc, "auc_t": 0.7},
         "E": E, "operational": {"G10000": {"detection_rate": 1.0, "n_episodes": 1}, "G800": {"detection_rate": 1.0}},
         "per_day": per_day, "per_attack_type": {"available": True, "types": {"atk": {f"recall_{X}": 0.5 for X in AG.X4}}},
         "sensitivities": sens, "t4p": t4p, "t4e_reference": None, "section_errors": section_errors or [],
         "provenance": {"metrics_rev_sha256": "synthetic-sha", "plan": {"plan_sha256": PLAN_SHA},
                        "inputs": {}}}
    if row["block"] == "T1-A-KNAD" and did == "D1" and row["seed"] == "A":
        m["t4e_reference"] = {X: {"auroc": auc - 0.001, "auprc": p, "f1": 0.3} for X in AG.X4}
    return m


DONE_AT = "2026-09-18T12:00:00"


def write_job(R, row, m=None, state="DONE", fail_count=0, nonexec=False, done_at=DONE_AT):
    J = R / row["job_id"]
    J.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(J / "config.json", {"job_id": row["job_id"], "manifest_row": row})
    sha = f"{abs(hash(row['job_id'])):064d}"[:64]
    C.write_json_atomic(J / "arrays.json", {"scores.npy": {"sha256": sha}})
    if state == "DONE":
        (J / "DONE").write_text(f"{done_at}\n")
        if m is not None:
            m["provenance"]["inputs"] = {"config_sha256": C.sha256_file(J / "config.json"), "scores_sha256_arrays_json": sha}
            C.write_json_atomic(J / "metrics.json", m)
    st = R / ".state" / row["job_id"]
    if fail_count:
        st.mkdir(parents=True, exist_ok=True)
        (st / "fail_count").write_text(f"{fail_count}\n")
    if nonexec:
        st.mkdir(parents=True, exist_ok=True)
        C.write_json_atomic(st / "nonexec_1.json", {"reason": "sintetico"})


def write_manifest(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=C.MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_complete_root(base):
    """Raiz 1: 9 datasets x 12 filas (A) + 10 filas (B, C), todas DONE; T2/T3/T4-E en AW; valores conocidos."""
    R = base / "results_complete"
    R.mkdir(parents=True)
    rows = []
    for did, _ in DATASETS:
        for seed in ("A", "B", "C"):
            for rc in AG.ROWCFGS:
                if rc in ("original", "stat") and seed != "A":
                    continue
                row = manifest_row(t1_block(seed, did), did, rc, seed)
                rows.append(row)
                auc = synth_auc(did, rc, seed)
                write_job(R, row, fake_metrics(row, auc))
    for rc, L in itertools.product(("lstm-AR", "mha-AR"), ("200", "2000")):
        row = manifest_row("T2-min", "D1", rc, "A", {"window": L})
        rows.append(row)
        write_job(R, row, fake_metrics(row, synth_auc("D1", rc, "A") + (0.02 if L == "200" else 0.001)))
    for rc, L in itertools.product(("lstm-TSR", "mha-TSR"), ("200", "2000")):
        row = manifest_row("T2-ext", "D1", rc, "A", {"window": L})
        rows.append(row)
        write_job(R, row, fake_metrics(row, synth_auc("D1", rc, "A")))
    for did, _ in DATASETS:
        for rc, k in itertools.product(("original", "stat", "mlp-AR"), ("1", "2", "8")):
            row = manifest_row("T3-min-a", did, rc, "A", {"k_clusters": k})
            rows.append(row)
            write_job(R, row, fake_metrics(row, synth_auc(did, rc, "A") - 0.01 * int(k) / 8))
    for rc, c in itertools.product(("original", "mlp-AR", "lstm-AR"), ("5", "20")):
        row = manifest_row("T4-E", "D1", rc, "A", {"contamination": c})
        rows.append(row)
        write_job(R, row, fake_metrics(row, synth_auc("D1", rc, "A") - 0.03))
    man = base / "manifest_complete.csv"
    write_manifest(man, rows)
    return R, man, rows


def build_scenario_root(base):
    """Root 2: scenarios of S*_d and of lost cells."""
    R = base / "results_scen"
    R.mkdir(parents=True)
    rows, ladder = [], []
    spec = {}
    for did, _ in DATASETS:
        for seed in ("A", "B", "C"):
            for rc in AG.ROWCFGS:
                if rc in ("original", "stat") and seed != "A":
                    continue
                row = manifest_row(t1_block(seed, did), did, rc, seed)
                rows.append(row)
                spec[(did, rc, seed)] = row
    done = {k: "DONE" for k in spec}
    extra = {}
    done[("D2", "lstm-AR", "C")] = "PENDING"
    done[("D3", "mha-TSR", "B")] = "NONEXEC"
    done[("D4", "mlp-TSR", "A")] = "BLOCKED"
    done[("D4", "mlp-AR", "C")] = "PENDING"
    done[("D7", "conv1d-AR", "C")] = "POSTFREEZE"
    done[("D5", "original", "A")] = "PENDING"
    done[("D6", "conv1d-AR", "B")] = "NOMETRICS"
    extra[("D7", "lstm-TSR", "A")] = {"divergent": True}
    done[("D8", "conv2d-TSR", "A")] = "BLOCKED"
    for rc in AG.ROWCFGS:
        if ("D9", rc, "C") in spec:
            ladder.append(spec[("D9", rc, "C")]["job_id"])
            done[("D9", rc, "C")] = "LADDER"
    done[("D1", "stat", "A")] = "SECTIONERR"
    for k, row in spec.items():
        st = done[k]
        auc = synth_auc(*k)
        if st == "DONE":
            write_job(R, row, fake_metrics(row, auc, **extra.get(k, {})))
        elif st == "POSTFREEZE":
            write_job(R, row, fake_metrics(row, auc), done_at="2026-10-02T10:00:00")
        elif st == "NOMETRICS":
            write_job(R, row, None)
        elif st == "SECTIONERR":
            write_job(R, row, fake_metrics(row, auc, section_errors=["S-INS: boom"]))
        elif st == "NONEXEC":
            write_job(R, row, None, state="NONE", nonexec=True)
        elif st == "BLOCKED":
            write_job(R, row, None, state="NONE", fail_count=3)
        elif st in ("PENDING", "LADDER"):
            write_job(R, row, None, state="NONE")
    att = R / ".state" / spec[("D3", "lstm-AR", "A")]["job_id"] / "attempt_1_host_123"
    att.mkdir(parents=True)
    np.save(att / "scores.npy", np.zeros(3))
    man = base / "manifest_scen.csv"
    write_manifest(man, rows)
    lj = base / "ladder.json"
    C.write_json_atomic(lj, {"removed_job_ids": ladder, "source": "sintetico (escalon 6)"})
    return R, man, lj, spec


def s7(ck, tmp):
    R, man, lj, spec = build_scenario_root(tmp)
    out = tmp / "agg_scen"
    rc = AG.main(["--results-root", str(R), "--manifest", str(man), "--out-dir", str(out), "--ladder-json", str(lj)])
    J = json.loads((out / "aggregate" / "aggregate.json").read_text())
    S = {d: J["datasets"][d]["S_star"] for d in ST.DS}
    exp = {"D1": ["A", "B", "C"], "D2": ["A", "B"], "D3": ["A", "B", "C"], "D4": ["A", "B", "C"], "D5": [],
           "D6": ["A"], "D7": ["A", "B"], "D8": ["A", "B", "C"], "D9": ["A", "B"]}
    ck.check("s7.exit1", rc == 1 and not J["complete"], "DONE sin metricas validas -> codigo 1 y complete = false")
    ck.check("s7.S_star", S == exp, f"S*_d = {S}")
    jse = spec[("D1", "stat", "A")]["job_id"]
    ck.check("s7.section_errors_F04", jse in J["section_error_jobs"] and not J["cells"]["D1"]["C11"]["lost"] and
             J["cells"]["D1"]["C11"]["mean"]["auroc_FUS"] == synth_auc("D1", "stat", "A"),
             "F-04: section_errors no invalidan el primario (D1 no se acorta); aggregate incompleto")
    ck.check("s7.sibling_C9_STm5", J["cells"]["D4"]["C9"]["lost"] and S["D4"] == ["A", "B", "C"],
             "ST-m5: replica MLP-AR sin ejecutar con su hermana perdida no acorta S*_d")
    jpf = spec[("D7", "conv1d-AR", "C")]["job_id"]
    ck.check("s7.post_freeze_AG11", J["jobs"][jpf]["state"] == "DONE_POST_FREEZE" and S["D7"] == ["A", "B"] and
             J["freeze"]["is_plan_default"] and J["freeze"]["n_done_post_freeze"] == 1,
             "AG-11: DONE posterior a la congelacion acorta S*_d y no cuenta")
    cells = J["cells"]
    ck.check("s7.lost_nonexec", cells["D3"]["C8"]["lost"] and cells["D3"]["C7"]["mean"]["auroc_FUS"] is not None and
             J["value_sets"]["main"]["D3"]["C8"]["auroc_FUS"] is None,
             "no ejecutable en B: celda perdida en todas las semillas sin acortar S*_d")
    ck.check("s7.C9_lost", cells["D4"]["C9"]["lost"] and "AG-1" in (cells["D4"]["C9"]["lost_reason"] or ""),
             "replica MLP-TSR bloqueada -> C9 perdida (AG-1)")
    ck.check("s7.blocked", cells["D8"]["C4"]["lost"] and J["value_sets"]["s_itt"]["D8"]["C4"]["auroc_FUS"] == 0.5,
             "fail_count >= 3 -> perdida; S-ITT imputa 0,5")
    a, b = synth_auc("D2", "lstm-AR", "A"), synth_auc("D2", "lstm-AR", "B")
    ck.check("s7.mean", cells["D2"]["C5"]["mean"]["auroc_FUS"] == float((a + b) / 2),
             "media sobre S*_d = (A + B)/2 exacta (float64)")
    ma = (synth_auc("D3", "mlp-AR", "A") + synth_auc("D3", "mlp-TSR", "A")) / 2
    ck.check("s7.mlp_mean", cells["D3"]["C9"]["seeds"]["A"]["auroc_FUS"] == float(ma),
             "C9 = media de replicas AR/TSR con la misma semilla")
    dm = J["mlp_noise"]["D3"]["A"]["delta_AR_minus_TSR"]
    ck.check("s7.delta_mlp", dm == synth_auc("D3", "mlp-AR", "A") - synth_auc("D3", "mlp-TSR", "A"),
             "Delta_MLP = AR - TSR")
    ck.check("s7.orig_once", cells["D3"]["C10"]["mean"]["auroc_FUS"] == synth_auc("D3", "original", "A") and
             cells["D3"]["C10"]["job_ids"]["C"] == cells["D3"]["C10"]["job_ids"]["A"],
             "original: una ejecucion (semilla A) vale para todas las semillas")
    c7 = cells["D7"]["C6"]
    ck.check("s7.itt", c7["divergent_seeds"] == ["A"] and c7["seeds"]["A"]["auroc_FUS"] == 0.5 and
             c7["seeds"]["A"]["auroc_KMD_f"] == 0.5 and J["value_sets"]["s_div_noimp"]["D7"]["C6"]["auroc_FUS"] is None,
             "divergida: ITT 0,5 en main (AG-5); s_div_noimp la elimina")
    sp7 = J["datasets"]["D7"]["sd_pool"]
    ck.check("s7.sd_noimp_STm7", c7["auroc_FUS_n_seeds_noimp"] == 1 and c7["auroc_FUS_sd_ddof1"] is None and
             c7["auroc_FUS_sd_ddof1_itt"] is not None and "C6" not in sp7["configs_used"] and
             sp7["sd_pool_itt"] is not None and sp7["df"] == len(sp7["configs_used"]),
             "AG-9 (ST-m7): SD y SD_pool sin imputar, semilla divergida fuera; version ITT aparte")
    sp = J["datasets"]["D3"]["sd_pool"]
    ck.check("s7.sd_pool", sp["df"] == len(sp["configs_used"]) * 2 and sp["threshold"] >= 0.01 and
             "C8" not in sp["configs_used"], f"SD_pool con gl = {sp['df']} (C8 perdida fuera)")
    ck.check("s7.sd_pool_1seed", J["datasets"]["D6"]["sd_pool"]["threshold"] == 0.01 and
             "no between-seed" in J["datasets"]["D6"]["sd_pool"]["note"], "|S*_d| = 1 -> umbral 0,01")
    with open(out / "runs.csv") as fh:
        rr = list(csv.DictReader(fh))
    uses = {r["job_id"]: r["use"] for r in rr if r["analysis"] != "attempt"}
    main_rows = [r for r in rr if r["analysis"] == "main"]
    o2_rows = [r for r in rr if r["analysis"] == "O2"]
    ck.check("s7.o2_rows_F03", main_rows and all(r["o2_f1_max"] == "" and r["o2_gap"] == "" for r in main_rows) and
             o2_rows and all(r["tag"].startswith("[ORACULO") and r["o2_f1_max"] for r in o2_rows),
             "F-03: O2 en filas propias con [ORACULO]; filas main sin columnas oraculo")
    ck.check("s7.post_freeze_use", "post-freeze" in uses[jpf], "runs.csv: uso post-freeze [EXPLORATORIO]")
    j_c = spec[("D2", "lstm-TSR", "C")]["job_id"]
    j_l = spec[("D9", "lstm-AR", "C")]["job_id"]
    ck.check("s7.runs_use", "fuera de S*_d" in uses[j_c] and "escalera" in uses[j_l] and
             "celda perdida" in uses[spec[("D8", "conv2d-TSR", "A")]["job_id"]],
             "runs.csv: uso por job (fuera de S*_d, escalera, perdida)")
    ck.check("s7.runs_all", len({r["job_id"] for r in rr}) == len(spec), f"runs.csv lista las {len(spec)} filas T1")
    att_rows = [r for r in rr if r["analysis"] == "attempt"]
    ck.check("s7.attempt_rows_STm9", len(att_rows) == 1 and att_rows[0]["attempt"] == "attempt_1_host_123" and
             "no cuenta" in att_rows[0]["use"], "ST-m9 (b): una fila por intento conservado en runs.csv")
    ok_sha = all(C.sha256_file(out / p) == h for p, h in J["outputs_sha256"].items())
    sums = (out / "aggregate" / "SHA256SUMS").read_text()
    ck.check("s7.sha", ok_sha and C.sha256_file(out / "aggregate" / "aggregate.json") in sums,
             "sha256 de runs.csv, cells.csv y aggregate.json registrados")
    ck.check("s7.warn", any("section_errors" in w for w in J["warnings"]), "section_errors -> aviso (F-04)")
    ck.check("s7.s_seed", J["value_sets"]["_meta"]["s_seed_computable"] is False, "algun |S*_d| = 0 -> S-SEED NA (AG-7)")
    jid = spec[("D3", "lstm-AR", "A")]["job_id"]
    ck.check("s7.attempts", J["jobs"][jid]["attempts_preserved"] == ["attempt_1_host_123"] and
             {r["n_attempts_preserved"] for r in rr if r["job_id"] == jid} == {"1"},
             "intentos con scores conservados listados (§12.2)")
    ch = J["chronology_R7"]
    ck.check("s7.chrono", ch["tag_before_first_metrics_json"] is True and ch["first_metrics_json_created"],
             f"R-7: hora del tag {ch['tag_commit_time']} < primer metrics.json {ch['first_metrics_json_created']}")


def s8(ck, tmp):
    R, man, rows = build_complete_root(tmp)
    out = tmp / "agg_complete"
    rc = AG.main(["--results-root", str(R), "--manifest", str(man), "--out-dir", str(out)])
    J = json.loads((out / "aggregate" / "aggregate.json").read_text())
    ck.check("s8.agg", rc == 0 and J["complete"] and all(len(J["datasets"][d]["S_star"]) == 3 for d in ST.DS),
             "aggregate completo: |S*_d| = 3 en 9/9")
    so = tmp / "stats_out"
    with jisa_now("2026-09-20 10:00"):
        rc = ST.main(["run", "--aggregate", str(out / "aggregate" / "aggregate.json"), "--out-dir", str(so),
                      "--friedman-b", "5000"])
    S = json.loads((so / "stats.json").read_text())
    ck.check("s8.run_noD", rc == 0 and S["D_guard"]["status"] == "bloqueado" and "bloqueado" in S["H2"]["status"] and
             "bloqueado" in S["R14"]["status"], "sin descriptor_d.json: H2 y R1.4 bloqueados (ST-8)")
    nm0 = (so / "numbers_proposed.md").read_text()
    rows0 = [ln for ln in nm0.splitlines() if ln.startswith("| R.")]
    ck.check("s8.partial_inputs_STM4", S["run_type"] == "INTERIM" and S["H1"]["tag"].startswith("[PARTIAL") and
             "PARTIAL INPUTS" in nm0.splitlines()[0] and rows0 and all("[PARTIAL INPUTS]" in ln for ln in rows0),
             f"ST-M4: antes de la congelacion -> [PARTIAL INPUTS] en stats.json y en cada fila de numbers ({S['run_type_reasons']})")
    ck.check("s8.x13_not_assigned_F08", S["x13_summary"]["claims"] is None and "not assigned" in S["x13_summary"]["status"],
             "F-08: sin tabla de afirmaciones no hay estado X13 por afirmacion")
    h1 = S["H1"]["comparisons"]
    ok = all(c["reject"] and abs(c["p_holm"] - 5 * 2 / 512) < 1e-15 and c["k_pos"] == 9 and c["robust"]
             for c in h1.values())
    ck.check("s8.H1", ok, "H1: 9/9 positivas en las 5 -> Holm 0,0195, robustas")
    c = h1["LSTM_vs_original"]
    exp = [((synth_auc(x, "lstm-AR", "A") + synth_auc(x, "lstm-AR", "B") + synth_auc(x, "lstm-AR", "C")) / 3 +
            (synth_auc(x, "lstm-TSR", "A") + synth_auc(x, "lstm-TSR", "B") + synth_auc(x, "lstm-TSR", "C")) / 3) / 2 -
           synth_auc(x, "original", "A") for x in ST.DS]
    ck.check("s8.H1_d", max(abs(c["d"][x] - e) for x, e in zip(ST.DS, exp)) < 1e-12 and
             c["hl"] == ST.hodges_lehmann(np.array([c["d"][x] for x in ST.DS])),
             "d = AUC(arquitectura: media de paradigmas y semillas) - AUC(original)")
    ck.check("s8.wording", c["wording"]["rule"] in ("outperforms", "significant_below_margin") and
             c["x13_if_claimed_first_higher"] == "mantenida", f"§5.6/X13: {c['wording']['rule']}, mantenida")
    h4 = S["H4"]
    ck.check("s8.H4", h4["claim_mlp_highest"]["x13"] == "mantenida" and
             h4["claim_mlp_highest"]["extension_to_kitnet_style_allowed"] is True,
             "H4: MLP > las 4 con HL > 0 y robusta -> 'MLP highest' mantenida (ST-10)")
    h3 = S["H3"]["family"]["comparisons"]
    ck.check("s8.H3", abs(h3["FUS_vs_KMD_f"]["hl"] - 0.02) < 1e-12 and h3["FUS_vs_KMD_f"]["reject"] and
             all(v == 11 for v in h3["FUS_vs_KMD_f"]["n_configs_per_dataset"].values()),
             "H3: media sobre 11 configuraciones de FUS - KMD_f = 0,02")
    om = S["H1_descriptive"]["omnibus"]
    ck.check("s8.omnibus", om["n"] == 9 and om["B"] == 5000 and 0 < om["p_permutation"] <= 1, "Friedman/ID con n = 9")
    delta = {x: ST.delta_h2(J["value_sets"]["main"], x) for x in ST.DS}
    Dj = {"schema": ST.SCHEMA_D, "datasets": {x: {"D": float(i), "created": "2026-09-18T00:00:00", "d_code_sha256": "c"}
                                              for i, x in enumerate(sorted(ST.DS, key=lambda z: delta[z]))}}
    dp = tmp / "descriptor_d.json"
    C.write_json_atomic(dp, Dj)
    claims = {"schema": ST.SCHEMA_X13, "claims": [
        {"id": "X-1", "tex": "TEX:1", "family": "H4_claim", "claimed_sign": 1},
        {"id": "X-2", "tex": "TEX:2", "family": "H1", "comparison": "MLP_vs_original", "claimed_sign": 1},
        {"id": "X-3", "tex": "TEX:3", "family": "H1", "comparison": "MLP_vs_original", "claimed_sign": -1}]}
    cp = tmp / "x13_claims.json"
    C.write_json_atomic(cp, claims)
    with jisa_now("2026-10-01 10:00"):
        rc = ST.main(["run", "--aggregate", str(out / "aggregate" / "aggregate.json"), "--out-dir", str(so),
                      "--friedman-b", "5000", "--descriptor", str(dp), "--x13-claims", str(cp)])
    S = json.loads((so / "stats.json").read_text())
    ck.check("s8.final_STM4", S["run_type"] == "FINAL" and not S["H1"]["tag"].startswith("[PARTIAL") and
             len((so / "runs_log.jsonl").read_text().splitlines()) == 2,
             "ST-M4: tras la congelacion, aggregate completo y congelacion del plan -> FINAL; runs_log.jsonl append-only")
    xs = {c["id"]: c["x13"] for c in S["x13_summary"]["claims"]}
    ck.check("s8.x13_claims_F08", xs == {"X-1": "mantenida", "X-2": "mantenida", "X-3": "retirada"} and
             S["provenance"]["inputs"]["x13_claims"]["sha256"] == C.sha256_file(cp) and
             S["x13_summary"]["fixed_in_git"] is False and "no fijada" in S["x13_summary"]["status"] and
             not any(c["citable"] for c in S["x13_summary"]["claims"]),
             f"F-08: un estado X13 por afirmacion; tabla fuera de git -> 'no fijada', no citable ({xs})")
    fx = ST.x13_summary_block(S, claims, {"fixed_in_git": True, "git": {"last_commit": "abc 2026-09-16T10:00:00+02:00",
                                                                         "clean_vs_HEAD": True}},
                              {"first_metrics_json_created": "2026-09-17T08:00:00"})
    ck.check("s8.x13_fixed_F08", fx["status"] == "ok" and all(c["citable"] for c in fx["claims"]) and
             fx["committed_before_first_metrics_json"] is True and S["S_s0_KNAD"]["status"].startswith("no implementada"),
             "F-08: tabla fijada en git -> citable; commit anterior al primer metrics.json; ST-21 S-s0-KNAD declarada")
    h2p = S["H2"]["primary"]["comparisons"]["H2_primary"]
    ck.check("s8.H2", rc == 0 and S["D_guard"]["status"] == "ok" and h2p["reject"] and S["H2"]["gate_open"] and
             S["H2"]["secondary"]["comparisons"]["H2a_LSTM"]["reject"] and
             "wilcoxon" not in S["H2"]["conv_descriptive"]["comparisons"]["Conv1D"],
             "con D: H2 primaria rechaza, guardian abierto, H2.a/H2.b con Holm m = 2; Conv1D/2D sin p")
    ck.check("s8.H2_conv_eq", S["H2"]["conv_descriptive"]["comparisons"]["Conv1D"]["equivalent_rope"] is True,
             "Conv1D AR - TSR = 0 -> 'equivalent' (IC dentro de +-0,01)")
    r = S["R14"]
    ck.check("s8.R14", r["n"] == 9 and r["spearman"]["rho"] == 1.0 and abs(r["spearman"]["p"] - 2 / math.factorial(9))
             < 1e-18 and r["reject"] and r["confirmatory"] and r["tag"] == ST.T["R14"] and
             r["critical_abs_rho"] == 0.7, "R1.4: rho = 1, p = 2/9! (permutacion completa); |rho| critico de la tabla")
    ck.check("s8.lodo_ties", "majority_ties" in r["exploratory"]["LODO_threshold_rule"] and
             r["exploratory"]["N_computed"] is not None, "ST-m8: empates de la regla mayoritaria y N_computed")
    ck.check("s8.R14_knad", "rho_KNAD" in r and "knad_occupy_extreme_D_ranks" in r and
             r["exploratory"]["N_expl"] == 11 and r["exploratory"]["E4.4"]["LSTM"]["n"] == 9,
             "rho_KNAD, rangos extremos y E4.x contados (N = 11)")
    ab = S["ablations"]["blocks"]
    t2 = [i for i in ab["T2-min"]["cells"] if i["value"] == "2000"]
    ck.check("s8.abl_noise", all(i["within_noise"] for i in t2) and
             all(not i["within_noise"] for i in ab["T2-min"]["cells"] if i["value"] == "200"),
             "T2: |Delta| = 0,001 dentro del ruido; 0,02 fuera (umbral max(sqrt2 SD_pool, 0,01))")
    t4e = ab["T4-E"]["cells"]
    ck.check("s8.t4e_ref", all(i.get("reference_field", "").startswith("t4e_reference") and
                               abs(i["delta_auroc"] - (-0.03 + 0.001)) < 1e-12 for i in t4e),
             "T4-E: referencia c = 0 = t4e_reference del job T1-A (AG-8)")
    ck.check("s8.t3a_hl", all(v["n"] == 9 and "hl_ci" in v for v in S["ablations"]["T3-min-a_hl"].values()),
             "T3-min-a: HL e IC exacto sobre 9 datasets, sin p")
    ck.check("s8.t4p", "D9" in S["T4P"]["a"] and "D7" in S["T4P"]["a_substitute_S_INS"] and "D1" in S["T4P"]["b"],
             "T4-P: a (CIC2018), sustituto S-INS (ACI), b (KNAD/CIC2017)")
    ck.check("s8.fail", S["failure_cases"]["selected"] and "is_failure_case" in
             S["failure_cases"]["per_dataset"]["D1"], f"casos de fallo: {S['failure_cases']['selection_rule'][:50]}")
    ck.check("s8.abstract", S["abstract"]["1_control_auroc"]["n"] == 9 and
             S["abstract"]["3_H1"]["eligible"] and S["abstract"]["5_f1"]["control_n_above_f1_trivial"] == 9,
             "abstract §11: control, media con contexto, H1 elegibles, F1 > trivial")
    sens = S["sensitivities"]
    ck.check("s8.sens", all(k in sens for k in ("s_itt", "s_div_noimp", "s_seed")) and
             sens["s_seed"]["H1"]["comparisons"]["LSTM_vs_original"]["n"] == 9, "S-ITT, sin imputar y S-SEED")
    bad, rules = [], set()
    for sname, blk in sens.items():
        fams = [blk.get("H1"), blk.get("H4"), (blk.get("H3") or {}).get("family"),
                (blk.get("H2") or {}).get("primary"), (blk.get("H2") or {}).get("secondary")]
        for f in fams:
            for k, c in ((f or {}).get("comparisons") or {}).items():
                w = (c.get("wording") or {}).get("rule")
                rules.add(w)
                if w in ("outperforms", "significant_below_margin") or c.get("abstract_eligible"):
                    bad.append((sname, k, w))
    ck.check("s8.sens_wording_STM1", not bad and rules <= {"sensitivity_consistent", "sensitivity_not_consistent"} and
             sens["s_itt"]["H4"]["claim_mlp_highest"]["x13"] is None,
             f"ST-M1: ninguna comparacion de sensibilidad con redaccion confirmatoria ({sorted(rules)}; {bad[:2]})")
    ck.check("s8.best_oracle_F09", all("ORACULO" in b["tag"] for b in S["R110"]["best_per_dataset"].values()),
             "F-09: best_per_dataset de R1.10 con [ORACULO: selected on evaluation results]")
    sums = dict(ln.split("  ")[::-1] for ln in (so / "SHA256SUMS").read_text().strip().splitlines())
    ck.check("s8.sha", all(C.sha256_file(so / n) == h for n, h in sums.items() if not n.startswith("input:")) and
             sums.get("input:runs_csv_sha256") == C.sha256_file(out / "runs.csv"),
             "SHA256SUMS de las salidas + sha256 de runs.csv y aggregate.json de entrada")
    nm = (so / "numbers_proposed.md").read_text()
    srcs = [ln.split(" | ")[3] for ln in nm.splitlines() if ln.startswith("| R.")]
    ck.check("s8.numbers", "| R.H1.AGG.LSTMvsORIG.HL |" in nm and "[CONFIRMATORIO plan §5.3]" in nm and
             "| R.T1.D3.C5.AUROC_FUS |" in nm and "[PARTIAL" not in nm, "numbers_proposed.md con IDs de §14.1 y etiqueta")
    ck.check("s8.numbers_vocab_STm4", srcs and all(x.startswith(("[CONFIRMATORIO", "[SENSIBILIDAD", "[EXPLORATORIO"))
                                                  for x in srcs) and "[DESCRIPTIVO" not in nm,
             "ST-m4: toda fila de numbers empieza por el vocabulario de §14.1")
    Dl = json.loads(json.dumps(Dj))
    Dl["datasets"]["D5"]["created"] = "2099-01-01T00:00:00"
    dl = tmp / "descriptor_late.json"
    C.write_json_atomic(dl, Dl)
    so2 = tmp / "stats_late"
    with jisa_now("2026-10-01 10:00"):
        ST.main(["run", "--aggregate", str(out / "aggregate" / "aggregate.json"), "--out-dir", str(so2),
                 "--friedman-b", "2000", "--descriptor", str(dl)])
    S2 = json.loads((so2 / "stats.json").read_text())
    nm2 = (so2 / "numbers_proposed.md").read_text()
    ck.check("s8.D_late_STM3", S2["D_guard"]["D_after_metrics"] == ["D5"] and not S2["R14"]["confirmatory"] and
             S2["R14"]["tag"].startswith("[EXPLORATORIO") and "H2_primary" in S2["H2"]["primary"]["comparisons"] and
             "| R.R14.AGG.D_DELTA.rho |" in nm2 and "[EXPLORATORIO: D calculado" in nm2,
             "ST-M3: D posterior al primer metrics.json del dataset -> R1.4 [EXPLORATORIO]; H2 se calcula")
    with open(so / "tests.csv") as fh:
        tr = list(csv.DictReader(fh))
    ck.check("s8.tests_csv", any(r["family"] == "H1" for r in tr) and any(r["family"].startswith("s_itt") for r in tr),
             f"tests.csv: {len(tr)} filas (confirmatorias y sensibilidades)")
    V = json.loads(json.dumps(J["value_sets"]["main"]))
    for x in ST.DS:
        for cfg in ST.CONFIGS:
            V[x][cfg]["auc_loc_RAW"] = 0.7 - (0.05 if cfg != "C10" else 0.0)
    f = ST.h1(V)
    cc = f["comparisons"]["LSTM_vs_original"]
    ck.check("s8.drift", cc["reject"] and not cc["robust"] and not cc["abstract_eligible"] and
             cc["x13_if_claimed_first_higher"] == "debilitada",
             "§5.8.1: AUC_loc(RAW) con signo opuesto -> no robusta, fuera del abstract, X13 debilitada")
    V = json.loads(json.dumps(J["value_sets"]["main"]))
    for x in ST.FLOW:
        V[x]["C5"]["auroc_FUS"] = V[x]["C10"]["auroc_FUS"] - 0.005
        V[x]["C6"]["auroc_FUS"] = V[x]["C10"]["auroc_FUS"] - 0.005
    f = ST.h1(V)
    cc = f["comparisons"]["LSTM_vs_original"]
    dep = cc["robustness"]["dependence"]["rule"]
    ck.check("s8.dependence", cc["reject"] is False or not dep["dependence_ok"],
             f"§1.2: sin KNAD con signo opuesto -> direccion no preservada ({dep})")
    ck.check("s8.dependence_rule", dep["direction_preserved_without_KNAD"] is False, "regla sin KNAD aplicada")


class jisa_now:
    """JISA_NOW (common.now) temporal para simular la hora frente a common.FREEZE."""
    def __init__(self, v):
        self.v, self.old = v, None

    def __enter__(self):
        self.old = os.environ.get("JISA_NOW")
        os.environ["JISA_NOW"] = self.v

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("JISA_NOW", None)
        else:
            os.environ["JISA_NOW"] = self.old


def s13(ck, tmp):
    """Checks of the wording and consistency rules: trend, equivalence with n <= 5, uninformative family, guard of D, chronology."""
    c9 = {"n": 9, "k_pos": 7, "k_neg": 2, "status": ""}
    c8 = {"n": 8, "k_pos": 8, "k_neg": 0, "status": ""}
    ck.check("s13.trend_rule", ST.wording(c9, False)["trend_allowed"] and not ST.wording(c8, False)["trend_allowed"] and
             not ST.wording(c9, False, trend_forbidden=True)["trend_allowed"], "ST-14: n = 9 y k >= 7; 8/8 no")
    V = {x: {c: {"auroc_FUS": 0.7, "auc_loc_RAW": 0.7} for c in ST.CONFIGS} for x in ST.DS}
    for i, x in enumerate(ST.DS):
        V[x]["C5"]["auroc_FUS"] = 0.7 + (0.02 if i < 7 else -0.02) * (1 + i)
        V[x]["C7"]["auroc_FUS"] = 0.7 + (0.001 if i % 2 else -0.001)
        V[x]["C9"]["auroc_FUS"] = 0.7 + (0.01 if i < 8 else -0.01)
    h = ST.h2(V, robust=False)
    wl = h["secondary"]["comparisons"]["H2a_LSTM"]["wording"]
    ck.check("s13.h2_gate_closed_STM2", not h["gate_open"] and not wl["trend_allowed"] and
             h["secondary"]["comparisons"]["H2a_LSTM"]["k_pos"] == 7 and
             not h["conv_descriptive"]["comparisons"]["Conv1D"]["wording"]["trend_allowed"],
             "ST-M2: H2.a con 7/9 y la primaria sin rechazar -> sin trend; Conv1D/Conv2D sin trend")
    hd = ST.h1_descriptive(V, friedman_b=200)
    ck.check("s13.h1d_C9", hd["vs_control"]["C9"]["k_pos"] == 8 and not hd["vs_control"]["C9"]["wording"]["trend_allowed"],
             "ST-M2: C9 frente a original (= H1.e) sin trend en la parte descriptiva")
    b = ST.describe({"a": 0.001, "b": -0.002, "c": 0.0, "d": 0.003}, ["a", "b", "c", "d"])
    ck.check("s13.equiv_n4_STm1", b["ci_within_rope"] and not b["equivalent_rope"] and "not assessable" in b["equivalence"],
             "ST-m1: n = 4 dentro de la ROPE -> 'equivalence not assessable' (cobertura 87,5 %)")
    Vu = {x: {c: {"auroc_FUS": None, "auc_loc_RAW": None, "f1_FUS": None} for c in ST.CONFIGS} for x in ST.DS}
    for i, x in enumerate(ST.DS[:7]):
        for c in ST.CONFIGS:
            Vu[x][c] = {"auroc_FUS": 0.6 + (0.05 if c != "C10" else 0.0) + 0.001 * i, "auc_loc_RAW": 0.6, "f1_FUS": 0.3}
    f = ST.h1(Vu, robust=False)
    rows = ST.numbers_rows({"H1": f}, "cmd", "ref")
    ab = ST.abstract_numbers(Vu, f, {"primary": {"comparisons": {"H2_primary": {}}}}, {})
    ck.check("s13.uninformative_STm2", f["comparisons"]["LSTM_vs_original"]["wording"]["rule"] ==
             "uninformative_by_construction" and not any(".p |" in r or ".pHolm |" in r for r in rows) and
             not ab["3_H1"]["w5_applies"] and ab["3_H1"]["uninformative_note"],
             "ST-m2: 'uninformative_by_construction', sin .p/.pHolm y sin w5")
    Vm = json.loads(json.dumps(V))
    Vm["D2"]["C3"]["auroc_FUS"] = None
    ab = ST.abstract_numbers(Vm, ST.h1(Vm, robust=False), {"primary": {"comparisons": {"H2_primary": {}}}}, {})
    ck.check("s13.ctx8_STm9", ab["2_context8_auroc"]["n"] == 8 and ab["2_context8_n_datasets"] == 8,
             "ST-m9 (a): un dataset sin una de las 8 queda fuera de la media con contexto")
    x = next(v for v in np.linspace(0.3, 0.99, 10_001) if (v + v + v) / 3 != v)
    cell = {"deterministic": True, "seeds": {"A": {"k": float(x)}, "B": {"k": float(x)}, "C": {"k": float(x)}}}
    ck.check("s13.det_mean_STm3", AG.cell_mean(cell, ["A", "B", "C"], "seeds", "k") == x and
             AG.mean_or_none([x, x, x]) != x, "ST-m3: original/stat sin la media de copias (1 ulp)")
    Dj = {"schema": ST.SCHEMA_D, "datasets": {d: {"D": 0.1, "created": "2026-09-18T10:00:00", "d_code_sha256": "c"}
                                              for d in ("D1", "D3")}}
    ch = {"first_metrics_json_created": "2026-09-18T09:00:00",
          "first_metrics_json_created_by_dataset": {"D1": "2026-09-18T11:00:00", "D3": "2026-09-18T12:00:00"}}
    ok1, g1 = ST.d_guard(Dj, ["D1", "D3"], ch)
    Dj["datasets"]["D3"]["created"] = "2026-09-18T13:00:00"
    ok2, g2 = ST.d_guard(Dj, ["D1", "D3"], ch)
    Dj["datasets"]["D3"]["created"] = "2026-09-18T10:00:00"
    Dj["datasets"]["D3"]["d_code_sha256"] = "otro"
    ok3, g3 = ST.d_guard(Dj, ["D1", "D3"], ch)
    ok4, g4 = ST.d_guard(Dj, ["D1", "D3"], None)
    ok5, _g5 = ST.d_guard(Dj, ["D1", "D2"], ch)
    ck.check("s13.d_guard_STM3", ok1 and g1["r14_confirmatory"] and ok2 and g2["D_after_metrics"] == ["D3"] and
             not g2["r14_confirmatory"] and not g3["r14_confirmatory"] and not g3["d_code_consistent"] and
             not g4["r14_confirmatory"] and g4["not_verifiable"] == ["D1", "D3"] and not ok5,
             "ST-M3: D antes del primer metrics.json de su dataset y mismo codigo; si no, R1.4 exploratorio; sin D, "
             "bloqueado")
    orig = ST.descriptor_dataset
    calls = []

    def fake(did, lc, with_e41=False, log=print):
        calls.append(did)
        return {"dataset": did, "D": 0.5 + len(calls), "created": C.now_iso()}
    ST.descriptor_dataset = fake
    try:
        dp = tmp / "d" / "descriptor_d.json"
        r1 = ST.main(["descriptor-d", "--datasets", "D3", "--out", str(dp)])
        r2 = ST.main(["descriptor-d", "--datasets", "D3", "--out", str(dp)])
        r3 = ST.main(["descriptor-d", "--datasets", "D3", "--out", str(dp), "--force"])
    finally:
        ST.descriptor_dataset = orig
    J = json.loads(dp.read_text())
    e = J["datasets"]["D3"]
    ck.check("s13.descriptor_append_only", r1 == r2 == r3 == 0 and calls == ["D3", "D3"] and e["D"] == 2.5 and
             len(e["history"]) == 1 and e["history"][0]["D"] == 1.5 and e["d_code_sha256"] and e["stats_sha256"],
             "ST-M3: descriptor-d no sobrescribe sin --force; con --force guarda historial y huellas de codigo")
    R = tmp / "mix"
    R.mkdir(parents=True)
    rows = []
    for i, rc_ in enumerate(("original", "stat")):
        row = manifest_row("T1-A-KNAD", "D3", rc_, "A")
        rows.append(row)
        m = fake_metrics(row, 0.7)
        m["provenance"]["helpers_sha256"] = {"common.py": f"h{i}"}
        write_job(R, row, m)
    M.record_chronology(R, {"created": "2026-09-17T08:00:00", "job_id": "x", "dataset_id": "D3"})
    M.record_chronology(R, {"created": "2026-09-18T08:00:00", "job_id": "y", "dataset_id": "D1"})
    first_ok = not M.record_chronology(R, {"created": "2026-09-19T08:00:00", "job_id": "z", "dataset_id": "D3"})
    man = tmp / "mix.csv"
    write_manifest(man, rows)
    out = tmp / "mix_out"
    rc = AG.main(["--results-root", str(R), "--manifest", str(man), "--out-dir", str(out)])
    J = json.loads((out / "aggregate" / "aggregate.json").read_text())
    ch = J["chronology_R7"]
    ck.check("s13.helpers_F05", rc == 1 and any("auxiliares" in e for e in J["consistency_errors"]),
             "F-05: helpers_sha256 mezclados -> error de consistencia (AG-10)")
    ck.check("s13.chronology_F10", first_ok and ch["first_metrics_json_created"] == "2026-09-17T08:00:00" and
             ch["first_metrics_json_created_by_dataset"]["D3"] == "2026-09-17T08:00:00" and
             ch["first_metrics_json_created_by_dataset"]["D1"] == "2026-09-18T08:00:00" and
             ch["record"]["first"]["job_id"] == "x",
             "F-10: primer metrics.json del registro O_EXCL (no se reescribe) y por dataset desde el log")


def s9(ck, tmp):
    orig = M.plan_guard

    def boom(*a, **k):
        raise M.PlanGuardError("sintetico: sin tag")
    M.plan_guard = boom
    try:
        o1, o2 = tmp / "r7_agg", tmp / "r7_stats"
        rc1 = AG.main(["--results-root", str(tmp / "nada"), "--manifest", str(tmp / "nada.csv"), "--out-dir", str(o1)])
        rc2 = ST.main(["run", "--aggregate", str(tmp / "nada.json"), "--out-dir", str(o2)])
        rc3 = ST.main(["descriptor-d", "--datasets", "D3", "--out", str(tmp / "r7_d" / "d.json")])
    finally:
        M.plan_guard = orig
    ck.check("s9.R7", rc1 == 3 and rc2 == 3 and rc3 == 3 and not o1.exists() and not o2.exists() and
             not (tmp / "r7_d").exists(), "guarda R-7 -> codigo 3 y nada escrito (aggregate, stats run, descriptor-d)")
    orig_e = M.env_guard

    def boom_e(*a, **k):
        raise M.EnvGuardError("sintetico: otro entorno")
    M.env_guard = boom_e
    try:
        rc = ST.main(["run", "--aggregate", str(tmp / "nada.json"), "--out-dir", str(tmp / "env_stats")])
    finally:
        M.env_guard = orig_e
    ck.check("s9.env", rc == 3 and not (tmp / "env_stats").exists(), "guarda de entorno (RF-2) -> codigo 3")
    ck.check("s9.usage", ST.main(["run", "--out-dir", "relativa"]) == 2, "rutas relativas -> codigo 2")


def s10(ck, tmp):
    rng = np.random.default_rng(5)
    n = 5000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    r_ar, _ = ST.rank_autocorr_lag1(x)
    r_iid, _ = ST.rank_autocorr_lag1(rng.normal(size=n))
    ck.check("s10.ar1", r_ar > 0.8 and abs(r_iid) < 0.1, "AR(1) phi = 0,9 -> rho1 alto; iid -> ~0")
    r_c, why = ST.rank_autocorr_lag1(np.full(100, 3.0))
    ck.check("s10.const", r_c is None and why == "constante", "feature constante excluida")
    y = np.array([1.0, np.inf, 2.0, -np.inf, np.inf, 3.0] * 50)
    yr = np.array([1.0, 5.0, 2.0, 0.0, 5.0, 3.0] * 50)
    ck.check("s10.inf", ST.rank_autocorr_lag1(y)[0] == ST.rank_autocorr_lag1(yr)[0], "+-inf ordenados como +-inf")
    X = np.column_stack([x, np.full(n, 1.0), rng.normal(size=n)])
    dd = ST.descriptor_from_matrix(X, ["ar", "const", "iid"])
    exp = float(np.median([r_ar, ST.rank_autocorr_lag1(X[:, 2])[0]]))
    ck.check("s10.median", dd["n_features_used"] == 2 and dd["constant_features"] == ["const"] and dd["D"] == exp,
             "D = mediana de rho1 de las features no constantes")
    mono = np.log1p(np.abs(x)) * np.sign(x) * 7 + 3
    ck.check("s10.monotone", abs(ST.rank_autocorr_lag1(mono)[0] - r_ar) < 1e-12, "invariante a transformaciones "
                                                                                  "monotonas (min-max, z-score)")


REAL_JOBS = ["T1-A-KNAD__knad-os_scan__mlp__arT__corrected.v1__L800__k4__sA__r1",
             "T1-A-KNAD__knad-os_scan__mlp__arF__corrected.v1__L800__k4__sA__r1"]


def s11(ck, tmp):
    RR = C.RESULTS_ROOT
    R = tmp / "real_root"
    R.mkdir(parents=True)
    rows = []
    for jid in REAL_JOBS:
        src = RR / jid
        if not (src / "DONE").exists():
            ck.warn("s11", f"{jid} no esta DONE; se salta")
            return
        J = R / jid
        J.mkdir()
        for f in ("config.json", "arrays.json", "DONE", "scores.npy", "labels.npy"):
            if (src / f).exists():
                os.symlink(src / f, J / f)
        t0 = time.time()
        rc = M.main(["--job-dir", str(src), "--out-dir", str(J)])
        ck.check(f"s11.metrics_rev.{jid.split('__')[3]}", rc == 0 and (J / "metrics.json").exists(),
                 f"metrics_rev a raiz temporal en {time.time() - t0:.0f} s")
        rows.append(C.read_json(src / "config.json")["manifest_row"])
    man = tmp / "manifest_real.csv"
    write_manifest(man, rows)
    out = tmp / "agg_real"
    rc = AG.main(["--results-root", str(R), "--manifest", str(man), "--out-dir", str(out)])
    J = json.loads((out / "aggregate" / "aggregate.json").read_text())
    ok = all(J["jobs"][j]["metrics_state"] == "ok" for j in REAL_JOBS)
    ck.check("s11.aggregate", rc in (0, 1) and ok and not J["consistency_errors"],
             "aggregate valida los metrics.json reales (esquema, plan, config y scores) sin errores de consistencia")
    with open(out / "runs.csv") as fh:
        rr = [r for r in csv.DictReader(fh) if r["job_id"] in REAL_JOBS]
    an = {r["analysis"] for r in rr}
    need = {"main", "S-NF", "S-Q95", "S-Q995", "S-MAX", "S-PS", "S-INS", "S-EP", "operational-G10000", "T4P-b", "T4P-c"}
    ck.check("s11.runs", need <= an, f"runs.csv con {len(rr)} filas de los 2 jobs y analisis {sorted(need - an) or 'OK'}")
    flat = AG.job_metrics(json.loads((R / REAL_JOBS[0] / "metrics.json").read_text()))
    ck.check("s11.flat", all(k in flat for k in ("auroc_FUS", "auc_loc_RAW", "auroc_KMD_f", "t4pb_c0.2_FUS_auroc",
                                                 "t4pc_f5_FUS_auroc", "auroc_FUS_NF")),
             f"job_metrics extrae {len(flat)} claves del esquema real")


def s12(ck, tmp):
    lc = C.read_json(REPO / M.LABEL_COUNTS_REL)
    t0 = time.time()
    rec = ST.descriptor_dataset("D3", lc, log=lambda *_: None)
    ck.check("s12.D_os_scan", rec["n_features_total"] == 115 and rec["n_features_used"] + rec["n_constant_excluded"] +
             len(rec["undefined_features"]) == 115 and rec["D"] is not None and rec["n_nan"] == 0,
             f"D de OS Scan desde la cache: {rec['n_features_used']} features usadas, "
             f"{rec['n_constant_excluded']} constantes, {time.time() - t0:.0f} s (valor no impreso)")


TESTS = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7, "s8": s8, "s9": s9, "s10": s10,
         "s13": s13}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--real-jobs", action="store_true")
    ap.add_argument("--real-d", action="store_true")
    a = ap.parse_args()
    tests = dict(TESTS)
    if a.real_jobs:
        tests["s11"] = s11
    if a.real_d:
        tests["s12"] = s12
    if a.only:
        sel = [s.strip() for s in a.only.split(",")]
        tests = {k: v for k, v in {**TESTS, "s11": s11, "s12": s12}.items() if k in sel}
    ck = Checker("test_h_stats")
    tmp = make_root("stats")
    try:
        for name, fn in tests.items():
            t0 = time.time()
            try:
                fn(ck, tmp / name)
            except Exception as e:
                import traceback
                traceback.print_exc()
                ck.check(f"{name}.exception", False, f"{type(e).__name__}: {e}")
            ck.info(f"{name}: {time.time() - t0:.1f} s")
    finally:
        keep = os.environ.get("JISA_KEEP_TEST_ROOT") == "1" or not ck.ok
        if keep:
            ck.info(f"raiz temporal conservada: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
