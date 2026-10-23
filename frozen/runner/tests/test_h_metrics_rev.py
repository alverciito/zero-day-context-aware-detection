#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_h_metrics_rev.py - tests of metrics_rev.py (analysis plan §14.3: R-1, R-2, R-7, R-8, and region sanity)."""
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent
sys.path.insert(0, str(RUNNER))
sys.path.insert(0, str(HERE))
import numpy as np
import common as C
import metrics_rev as M
from _util import Checker

REPO = M.REPO
RES = REPO / "results" / "revision"
CACHE = REPO / "data" / "cache" / "streams"
D25 = REPO / "postprocess" / "detectors_2025"
VALUES = REPO / "data" / "remote_chiquitin" / "detectors_2025"
LC = json.loads((REPO / M.LABEL_COUNTS_REL).read_text())
LX = json.loads((REPO / M.LABEL_EXTRAS_REL).read_text())
KNAD_KEYS = {"knad_active_wiretap": "Active_Wiretap", "knad_fuzzing": "Fuzzing", "knad_os_scan": "OS_Scan",
             "knad_ssdp_flood": "SSDP_Flood", "knad_ssl_renegotiation": "SSL_Renegotiation",
             "knad_video_injection": "Video_Injection"}
CSV_CACHE = {"cic2017": "cic2017__corrected__v1", "cic2018": "cic2018__corrected__v1", "aci": "aci-iot-2023__corrected__v1"}
CSV_DS = {"cic2017": "CIC2017", "cic2018": "CIC2018", "aci": "ACI-IOT-2023"}
JOB_OS_STAT = "T1-A-KNAD__knad-os_scan__stat__arT__corrected.v1__L800__k4__sA__r1"
JOB_AW_STAT = "T1-A-KNAD__knad-active_wiretap__stat__arT__corrected.v1__L800__k4__sA__r1"
JOB_AW_LSTM = "T1-A-KNAD__knad-active_wiretap__lstm__arT__corrected.v1__L800__k4__sA__r1"
X2_JOB = "X2__knad-os_scan__stat__arT__orig.v1__L800__k4__sNone__r1"
PY = sys.executable


def synth_scores(n, seed=0):
    """Scores sinteticos (independientes de las etiquetas) con deriva lenta y colas: solo para ejercitar el codigo."""
    rng = np.random.default_rng(seed)
    base = np.log10(1 + rng.gamma(2.0, 0.05, n))
    drift = 0.02 * np.sin(np.arange(n) / 50_000.0)
    return (base + drift + 0.001 * rng.standard_normal(n)).astype(np.float64)


def h1(ck, tmp):
    import inspect
    sig = inspect.signature(M.fit)
    ck.check("h1.api_fit_sin_etiquetas", not any("label" in p or p in ("y", "E") for p in sig.parameters),
             f"fit{sig}")
    ck.check("h1.api_evaluate", list(inspect.signature(M.evaluate).parameters)[:4] == ["params", "scores", "labels", "E"],
             f"evaluate{inspect.signature(M.evaluate)}")
    n, T = 330_000, 60_000
    Cn = (n - T) // 10
    s = synth_scores(n, 1)
    mask = np.zeros(n, bool)
    mask[T:T + Cn] = True
    mask[T + 7:T + 20] = False
    p0 = M.fit(s[T:T + Cn], mask[T:T + Cn], T)
    ser0 = M.transform(p0, s[T:])
    rng = np.random.default_rng(7)
    ok_all = True
    for t in (T + Cn, T + Cn + 1, T + Cn + 12_345, T + Cn + 60_000, n - 2):
        s2 = s.copy()
        s2[t + 1:] = s2[t + 1:] * rng.uniform(0.2, 5.0, n - t - 1) + rng.uniform(0, 3)
        p2 = M.fit(s2[T:T + Cn], mask[T:T + Cn], T)
        same_p = json.dumps(p2, sort_keys=True) == json.dumps(p0, sort_keys=True)
        ser2 = M.transform(p2, s2[T:])
        k = t + 1 - T
        eq = {x: bool(np.array_equal(ser0[x][:k], ser2[x][:k])) for x in ("FUS", "KMD_f", "MVD_f", "FUS_MAX", "FUS_NF",
                                                                         "KMD_raw", "MVD_raw", "RAW")}
        changed_after = bool(not np.array_equal(ser0["FUS"][k:], ser2["FUS"][k:]))
        ok = same_p and all(eq.values()) and changed_after
        ok_all &= ok
        ck.check(f"h1.impulso_sintetico_t={t}", ok, f"params iguales={same_p}; prefijo identico={eq}; cambia despues={changed_after}")
    t = T + Cn // 2
    s3 = s.copy()
    s3[t + 1:] += 1.0
    ser3 = M.transform(p0, s3[T:])
    k = t + 1 - T
    ck.check("h1.transform_causal_en_calibracion", all(np.array_equal(ser0[x][:k], ser3[x][:k]) for x in ser0),
             "mismos params: series identicas hasta t")
    J = RES / JOB_OS_STAT
    if not (J / "scores.npy").exists():
        ck.warn("h1.real", f"sin {J}")
        return
    cfg = json.loads((J / "config.json").read_text())
    sr = np.load(J / "scores.npy")
    calib, E, _ = M.get_regions(cfg["regions"], M.PRIMARY_CANDIDATE)
    Tr = calib[0]
    m = np.zeros(len(sr), bool)
    m[calib[0]:calib[1]] = True
    pr = M.fit(sr[Tr:calib[1]], m[Tr:calib[1]], Tr)
    sa = M.transform(pr, sr[Tr:])
    for t in (E[0] + 1000, (E[0] + E[1]) // 2):
        s2 = sr.copy()
        s2[t + 1:] = rng.permutation(s2[t + 1:])
        sb = M.transform(pr, s2[Tr:])
        k = t + 1 - Tr
        ck.check(f"h1.impulso_real_{JOB_OS_STAT[:24]}..._t={t}",
                 all(np.array_equal(sa[x][:k], sb[x][:k]) for x in ("FUS", "KMD_f", "MVD_f", "FUS_MAX", "FUS_NF")),
                 "FUS/KMD_f/MVD_f/FUS_MAX/FUS_NF identicos hasta t tras permutar s[t+1:]")


def h2(ck, tmp):
    rng = np.random.default_rng(20260918)
    bad = 0
    n_def = 0
    for i in range(1000):
        n = int(rng.integers(2, 160))
        p_ref, p_hyp = rng.uniform(0.02, 0.9, 2)
        ref = (rng.random(n) < p_ref).astype(np.int8)
        if i % 3 == 0:
            ref = np.repeat(rng.integers(0, 2, max(1, n // 7)), 7)[:n].astype(np.int8)
            n = len(ref)
        hyp = (rng.random(n) < p_hyp).astype(np.int8)
        k_rule = M.wd_k(ref)[0]
        for k in {k_rule, int(rng.integers(1, max(2, n))), 1, n - 1}:
            a = M.windowdiff(ref, hyp, k)
            b = M.windowdiff_bruteforce(ref.tolist(), hyp.tolist(), k)
            if a is not None:
                n_def += 1
            if a != b:
                bad += 1
    ck.check("h2.wd_int64_vs_fuerza_bruta", bad == 0, f"1.000 secuencias; {n_def} WD definidos; discrepancias={bad}")
    ref = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1], np.int8)
    k, nseg = M.wd_k(ref)
    ck.check("h2.regla_k", (k, nseg) == (int(math.floor(10 / (2 * 4) + 0.5)), 4), f"k={k}, segmentos={nseg}")
    big = np.zeros(3_000_000, np.int8)
    big[::3] = 1
    t0 = time.time()
    v = M.windowdiff(big, big[::-1].copy(), 7)
    ck.check("h2.int64_sin_desbordamiento", v is not None and 0 <= v <= 1 and time.time() - t0 < 10,
             f"n = 3e6 con 2e6 fronteras en {time.time() - t0:.2f} s")


def h3(ck, tmp, full=False):
    if not (D25 / "execute.py").exists() or not VALUES.exists():
        ck.warn("h3", "sin datos del post-proceso original")
        return
    datasets = sorted(p.name for p in (VALUES / "experiments").iterdir()) if full else ["os_scan", "aci_iot_2023"]
    worst, n, bad = 0.0, 0, []
    exact = 0
    t0 = time.time()
    for sub in ("experiments", "experiments_kmeans"):
        for ds in datasets:
            for vf in sorted((VALUES / sub / ds).glob("__*_values.npy")):
                name = vf.name[2:-len("_values.npy")]
                js = D25 / sub / ds / f"__{name}_metrics.json"
                if not js.exists():
                    bad.append(f"falta {js}")
                    continue
                J = json.loads(js.read_text())
                v = np.load(vf, mmap_mode="r")
                for kind, i_det, i_tgt in (("Raw", 0, 2), ("Smooth", 1, 3)):
                    det = np.asarray(v[i_det])
                    tgt = np.asarray(v[i_tgt]).astype(int)
                    dec = (det >= J[f"Threshold {kind}"]).astype(int)
                    wd, pk, info = M.wd_pk_uint8(tgt, dec)
                    tp = int(np.sum((dec == 1) & (tgt == 1)))
                    for key, val in ((f"Window Diff {kind}", wd), (f"Pk {kind}", pk)):
                        d = abs(val - float(J[key]))
                        worst = max(worst, d)
                        exact += d == 0
                        n += 1
                        if d > 3e-8:
                            bad.append(f"{sub}/{ds}/{name} {key}: |d| = {d:.3g}")
                    if tp != int(J[f"TP {kind}"]):
                        bad.append(f"{sub}/{ds}/{name} TP {kind} distinto")
                del v
    ck.check("h3.wd_pk_uint8_vs_json_enviados", n > 0 and not bad,
             f"{n} valores WD/Pk ({n // 4} JSON, {'todos' if full else 'OS Scan + ACI'}); max|d| = {worst:.3g} "
             f"(tol 3e-8); exactos {exact}/{n}; problemas {bad[:5]}; {time.time() - t0:.1f} s")


def _labels_stream(key):
    if key in KNAD_KEYS:
        with np.load(M.T0_CACHE / "labels" / "knad" / f"{KNAD_KEYS[key]}.npz") as z:
            return z["codes"].astype(np.int8), None
    cdir = CACHE / CSV_CACHE[key]
    return np.load(cdir / "y.npy").astype(np.int8), cdir


def h4(ck, tmp):
    for key in [k for k in LX if not k.startswith("_")]:
        y, cdir = _labels_stream(key)
        lce, lx = LC[key], LX[key]
        N = len(y)
        T = lce["T_train_end"]
        Cn = (N - T) // 10
        E = np.arange(T + Cn, N)
        yE = y[E].astype(bool)
        a = M.auc_index(yE)
        cov = M.loc_coverage(yE)
        d_auc = abs(a - lx["AUC_t"])
        dcov = max(abs(cov[k] - lx["S_LOC_cover"][k]) for k in cov)
        ck.check(f"h4.R8_{key}", N == lx["N"] and [T + Cn, N] == lce["evaluation"] and d_auc <= 1e-12 and dcov <= 1e-12,
                 f"AUC_t |d| = {d_auc:.2g}; cobertura S-LOC max|d| = {dcov:.2g} (fuente de etiquetas: "
                 f"{'cache T0' if cdir is None else cdir.name})")
        if key in CSV_DS:
            cfg = {"cache": {"path": str(cdir)}}
            aux = M.load_aux(cfg, N, T, y, CSV_DS[key])
            parts = M.per_day_partition(aux["day"], aux["day_names"], N, E, y)
            ref = lx.get("per_day") or {}
            mism = []
            for d, _keep, _yy, rec in parts:
                r = ref.get(d)
                if r is None:
                    mism.append(f"{d}: no esta en label_extras")
                    continue
                for k2 in ("rows_E", "rows_after_guard", "attack_rows", "benign_rows", "episodes_G10000_within_day",
                           "eligible_auc_per_day"):
                    if r[k2] != rec[k2]:
                        mism.append(f"{d}.{k2}: {rec[k2]} != {r[k2]}")
            elig = sorted(d for d, _k, _y, rec in parts if rec["eligible_auc_per_day"])
            ck.check(f"h4.dias_{key}", not mism and len(parts) == len(ref),
                     f"{len(parts)} dias; elegibles {elig}; discrepancias {mism[:4]}")


def h5(ck, tmp):
    import run_job as RJ
    J = RES / JOB_AW_STAT
    cfg = json.loads((J / "config.json").read_text())
    reg = cfg["regions"]
    calib, E, info = M.get_regions(reg, M.PRIMARY_CANDIDATE)
    lce = LC["knad_active_wiretap"]
    ck.check("h5.primaria_real", calib == lce["calibration"] and E == lce["evaluation"] and info["mode"] == "plain",
             f"calib {calib}, E {E}")
    bad = copy.deepcopy(reg)
    bad["detector_protocol_candidates"][M.PRIMARY_CANDIDATE]["calib"] = [200_000, 500_000]
    try:
        M.get_regions(bad, M.PRIMARY_CANDIDATE)
        ok = False
    except C.ProtocolRegionError as e:
        ok = "solapa" in str(e) or "no termina" in str(e)
    ck.check("h5.solape_calib_eval_rechazado", ok, "ProtocolRegionError")
    bad = copy.deepcopy(reg)
    bad["detector_protocol_candidates"][M.PRIMARY_CANDIDATE]["eval"] = [150_000, reg["n_rows"]]
    try:
        M.get_regions(bad, M.PRIMARY_CANDIDATE)
        ok = False
    except C.ProtocolRegionError:
        ok = True
    ck.check("h5.eval_en_entrenamiento_rechazada", ok, "ProtocolRegionError")
    x2 = json.loads((RES / X2_JOB / "config.json").read_text()) if (RES / X2_JOB / "config.json").exists() else None
    if x2:
        r1 = r2 = False
        try:
            M.get_regions(x2["regions"], "as_submitted_regression_only")
        except C.ProtocolRegionError:
            r1 = True
        try:
            M.applicable(x2, RES / X2_JOB)
        except M.NotApplicable:
            r2 = True
        ck.check("h5.regression_only_y_X2_rechazados", r1 and r2, "candidata del envio y job X2 no aplicables")
    ins = copy.deepcopy(reg)
    ins["detector_protocol_candidates"][M.INS_CANDIDATE]["calib_in_sample"] = False
    try:
        M.get_regions(ins, M.INS_CANDIDATE)
        ok = False
    except C.ProtocolRegionError:
        ok = True
    c_ins, e_ins, _ = M.get_regions(reg, M.INS_CANDIDATE)
    ck.check("h5.S_INS_requiere_marca", ok and c_ins == [100_000, 200_000] and
             M.segments_subset(M.segments_of(E), M.segments_of(e_ins)), f"calib in-sample {c_ins}; E ⊂ eval in-sample")
    ex = C.t4e_eval_exclusion("KNAD:Active_Wiretap", REPO)
    _, Er, ir = M.get_regions(reg, M.PRIMARY_CANDIDATE, extra_eval_exclude=[ex["excluded_with_guards"]])
    rows = M.rows_of(Er)
    xs, xe = ex["excluded_with_guards"]
    ck.check("h5.extra_eval_exclude_T4E_ref", len(Er) == 2 and not ((rows >= xs) & (rows < xe)).any() and
             len(rows) == lce["t4p_donor_pool"]["remaining_eval_rows"], f"segmentos {Er}")
    spec = C.Spec(_manifest_row("T4-E__knad-active_wiretap__original__arT__corrected.v1__L800__k4__c20__sA__r1"), False)
    cmeta = json.loads((CACHE / "knad-active_wiretap__corrected__v1" / "meta.json").read_text())
    t4e = _t4e_info(lce, reg)
    reg4 = RJ.regions_for(spec, reg["n_rows"], {"fm": 50_000, "ad": 150_000}, cmeta, t4e=t4e)
    try:
        C.check_protocol_regions(reg4, M.PRIMARY_CANDIDATE)
        plain_ok = False
    except C.ProtocolRegionError:
        plain_ok = True
    _, E4, i4 = M.get_regions(reg4, M.PRIMARY_CANDIDATE)
    ck.check("h5.T4E_with_exclusions", plain_ok and i4["mode"] == "with_exclusions" and len(E4) == 2 and
             M.rows_of(E4).size == len(rows), f"sin with_exclusions rechaza; segmentos {E4}")


def _manifest_row(job_id):
    rows = C.read_manifest(RUNNER / "manifest.csv")
    for r in rows:
        if r["job_id"] == job_id:
            return r
    raise KeyError(job_id)


def _t4e_info(lce, reg):
    fm, ad = 50_000, 150_000
    B = int(round(20 * ad / 100.0))
    start = fm + (ad - B) // 2
    pool = lce["t4p_donor_pool"]
    cal10 = reg["detector_protocol_candidates"][M.PRIMARY_CANDIDATE]["calib"]
    return {"calib": cal10, "eval_exclude_with_guards": pool["excluded_with_guards"], "donor_pool": pool["pool"],
            "injected_rows": [start, start + B], "B_c": B, "donor_rows": {"first": pool["pool"][0],
                                                                         "last": pool["pool"][0] + B - 1}}


def h6(ck, tmp):
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, f1_score
    rng = np.random.default_rng(3)
    stats = {"auc_bitwise": 0, "ap_bitwise": 0, "o2": 0.0, "o2_thr_mismatch": 0, "f1": 0.0, "eer": 0.0, "n": 0}
    for _ in range(400):
        n = int(rng.integers(4, 4000))
        y = rng.random(n) < rng.uniform(0.02, 0.98)
        if y.all() or not y.any():
            continue
        x = np.round(rng.normal(size=n) * rng.choice([1, 10, 1000]), int(rng.integers(0, 3)))
        cm = M.curve_metrics(x, y, o2=True)
        stats["n"] += 1
        stats["auc_bitwise"] += cm["auroc"] != roc_auc_score(y, x)
        stats["ap_bitwise"] += cm["auprc"] != average_precision_score(y, x)
        p, r, _t = precision_recall_curve(y, x)
        f = np.where(p + r > 0, 2 * p * r / np.where(p + r > 0, p + r, 1), 0)
        stats["o2"] = max(stats["o2"], abs(f.max() - cm["o2"]["f1_max"]))
        stats["o2_thr_mismatch"] += cm["o2"]["threshold"] != float(_t[int(np.argmax(f[:-1]))])
        tau = float(np.quantile(x, rng.uniform(0.1, 0.99)))
        stats["f1"] = max(stats["f1"], abs(f1_score(y, x > tau, zero_division=0) - M.threshold_metrics(x, y, tau)["f1"]))
        fpr, tpr, _ = roc_curve(y, x, drop_intermediate=False)
        d = (1 - tpr) - fpr
        j = next(i for i in range(len(d)) if d[i] <= 0)
        e_ref = fpr[j] if d[j] == 0 else fpr[j - 1] + d[j - 1] / (d[j - 1] - d[j]) * (fpr[j] - fpr[j - 1])
        stats["eer"] = max(stats["eer"], abs(e_ref - cm["eer"]))
    ck.check("h6.replica_sklearn", stats["auc_bitwise"] == 0 and stats["ap_bitwise"] == 0 and stats["o2"] <= 1e-12 and
             stats["o2_thr_mismatch"] == 0 and stats["f1"] <= 1e-12 and stats["eer"] <= 1e-12, f"{stats}")
    tie = M.curve_metrics(np.array([4.0, 3.0, 2.0, 1.0]), np.array([1, 0, 0, 1], bool), o2=True)["o2"]
    ck.check("h6.o2_empate", tie["threshold"] == 1.0 and tie["recall"] == 1.0,
             "RA-6: F1 maximo empatado en los umbrales 4 y 1 -> se reporta el mas bajo (1)")
    y = np.zeros(8, bool)
    ck.check("h6.una_clase", M.curve_metrics(np.arange(8.0), y)["auroc"] is None and
             M.threshold_metrics(np.arange(8.0), y, 100.0)["f1"] == 0.0, "AUC no definido; F1 = 0 sin alarmas")
    n = 80_000
    yb = np.zeros(n, bool)
    for s0 in (3_000, 15_000, 27_000, 39_000, 51_000, 63_000):
        yb[s0:s0 + 300:2] = True
    xb = rng.normal(size=n) + yb * 0.3
    xb = np.round(xb, 2)
    res = M.bootstrap_auc_ci(xb, yb, B=4, return_values=True)
    eps, segs = res["blocks"]["episodes"], res["blocks"]["segments"]
    g = np.random.default_rng(M.BOOT_SEED)
    worst = 0.0
    for b in range(4):
        me = np.bincount(g.integers(0, len(eps), len(eps)), minlength=len(eps))
        ms = np.bincount(g.integers(0, len(segs), len(segs)), minlength=len(segs))
        idx = [np.arange(s, e) for (s, e), m in zip(eps, me) for _ in range(m)] + \
              [np.arange(s, e) for (s, e), m in zip(segs, ms) for _ in range(m)]
        idx = np.concatenate(idx)
        worst = max(worst, abs(roc_auc_score(yb[idx], xb[idx]) - res["values"][b]))
    ck.check("h6.bootstrap_ponderado_igual_concatenacion", worst <= 1e-12 and len(eps) == 6,
             f"4 replicas; max|d| = {worst:.2g}; episodios {len(eps)}, segmentos {len(segs)}")


def h7(ck, tmp):
    import pandas as pd
    n = 75_000
    s = synth_scores(n, 11)
    p = M.fit(s[:30_000], np.ones(30_000, bool), 0)
    ser = M.transform(p, s)
    sf = s[:30_000]
    ok_p = (p["mvd"]["mu_ref"] == float(np.mean(sf)) and p["mvd"]["sd_ref"] == float(np.std(sf)) and
            p["thresholds"]["0.99"]["RAW"] == float(np.quantile(sf, 0.99, method="linear")))
    ck.check("h7.params_mvd_umbral", ok_p, "mu/sd ddof 0 y P99 lineal")
    rng = np.random.default_rng(5)
    worst = {"mu": 0.0, "kmd_f": 0.0, "mvd_f": 0.0, "mvd_raw": 0.0}
    c = np.array(p["kmd"]["centroids"])
    kraw = np.min(np.abs(s[:, None] - c[None, :]), axis=1)
    ck.check("h7.kmd_raw", np.array_equal(kraw, ser["KMD_raw"]), "min_j |s - c_j|")
    for t in np.r_[0, 1, 9_999, 10_000, 50_000, 50_001, rng.integers(0, n, 40)]:
        w = s[max(0, t - M.W_MVD + 1):t + 1]
        mu, sd = w.mean(), w.std()
        md, sv = (mu - p["mvd"]["mu_ref"]) ** 2, (sd - p["mvd"]["sd_ref"]) ** 2
        worst["mvd_raw"] = max(worst["mvd_raw"], abs(2 * md * sv / (md + sv + 1e-9) - ser["MVD_raw"][t]))
        worst["kmd_f"] = max(worst["kmd_f"], abs(ser["KMD_raw"][max(0, t - M.F_FILT + 1):t + 1].mean() - ser["KMD_f"][t]))
        worst["mvd_f"] = max(worst["mvd_f"], abs(np.median(ser["MVD_raw"][max(0, t - M.F_FILT + 1):t + 1]) - ser["MVD_f"][t]))
    ck.check("h7.series_causales_vs_directo", worst["mvd_raw"] <= 1e-9 and worst["kmd_f"] <= 1e-9 and worst["mvd_f"] == 0,
             f"max|d| {worst} (media/std acumuladas de pandas; mediana exacta)")
    ck.check("h7.std_1obs_cero", float(pd.Series([3.0, 5.0]).rolling(10, min_periods=1).std(ddof=0).iloc[0]) == 0.0,
             "pandas std(ddof=0) con 1 observacion = 0")
    rs = p["fusion"]["robust"]["KMD_f"]
    z = (ser["KMD_f"] - rs["median"]) / rs["scale"]
    z2 = (ser["MVD_f"] - p["fusion"]["robust"]["MVD_f"]["median"]) / p["fusion"]["robust"]["MVD_f"]["scale"]
    ck.check("h7.fusion", np.array_equal(ser["FUS"], (z + z2) / 2) and np.array_equal(ser["FUS_MAX"], np.maximum(z, z2)) and
             rs["scale"] == 1.4826 * rs["mad"], "FUS = media de z robustos; FUS_MAX = max")
    a = M.robust_scale(np.r_[np.zeros(60), np.ones(40)])
    b = M.robust_scale(np.zeros(50))
    ck.check("h7.escala_alternativas", a["scale_source"] == "IQR/1.349" and a["scale"] == 1 / 1.349 and
             b["degenerate_scale"] and b["scale"] == 1e-9,
             f"{a['scale_source']}, {b['scale_source']}")
    s_nan = s.copy()
    s_nan[[5, 40_000, 40_001]] = [np.nan, np.inf, -np.inf]
    sd, sr, info = M.sanitize_scores(s_nan, 0)
    exp = np.maximum.accumulate(np.where(np.isfinite(s_nan), s_nan, -np.inf))
    ck.check("h7.no_finitos", np.isfinite(sd).all() and sd[40_000] == exp[40_000] and sr[5] == np.nanmax(s[np.isfinite(s_nan)]) + 1
             and info["replaced"] == 3, f"{info}")


def h8(ck, tmp):
    ok = M.plan_guard()
    ck.check("h8.R7_real", ok["plan_sha256"] == ok["plan_sha256_decisions_md"] == ok["plan_sha256_worktree"],
             f"commit {ok['plan_commit'][:10]} {ok['tag_commit_time']}")
    plan = tmp / "plan.md"
    plan.write_bytes((REPO / M.PLAN_REL).read_bytes() + b" ")
    dec = tmp / "decisions.md"
    dec.write_text((REPO / M.DECISIONS_REL).read_text(encoding="utf-8"), encoding="utf-8")
    res = []
    for pp, dd, what in ((plan, None, "plan modificado"), (None, tmp / "vacio.md", "decisions sin sha"),
                         (None, "wrong", "sha distinto")):
        if dd == "wrong":
            dd = tmp / "dec_wrong.md"
            dd.write_text(dec.read_text().replace(ok["plan_sha256"], "0" * 64))
        elif dd is not None:
            dd.write_text("nada\n")
        try:
            M.plan_guard(plan_path=pp, decisions_path=dd)
            res.append((what, False))
        except M.PlanGuardError:
            res.append((what, True))
    try:
        M.plan_guard(tag="no-existe-este-tag")
        res.append(("sin tag", False))
    except M.PlanGuardError:
        res.append(("sin tag", True))
    ck.check("h8.R7_aborta", all(v for _, v in res), f"{res}")


def h9(ck, tmp):
    rng = np.random.default_rng(9)
    n, T = 4_000, 1_000
    x = np.asfortranarray(rng.normal(size=(n, 115)))
    x[1500] = x[17]
    x[2500] = x[999]
    x[3000, :] = x[5, :]
    x[3000, 3] = -0.0 if x[5, 3] == 0 else x[3000, 3]
    x[3100] = x[5] + 1e-12
    cdir = tmp / "cache_knad"
    cdir.mkdir()
    np.save(cdir / "x.npy", x)
    (cdir / "meta.json").write_text(json.dumps({"x": {"sha256": "sha-x"}}))
    cfg = {"regions": {}, "spec": {"kind": "knad"}, "cache": {"path": str(cdir), "x": {"sha256": "sha-x"}}}
    m, info = M.fit_set_mask(cfg, n, T, log=lambda *a: None)
    ck.check("h9.knad_mascara", set(np.flatnonzero(m).tolist()) == {1500, 2500, 3000} and "train_duplicate_mask" in info["mask_source"],
             f"marcadas {np.flatnonzero(m).tolist()}")
    cfg2 = copy.deepcopy(cfg)
    cfg2["cache"]["x"]["sha256"] = "otra"
    M._KNAD_DUP_MEMO.clear()
    try:
        M.fit_set_mask(cfg2, n, T)
        ok = False
    except RuntimeError:
        ok = True
    mp = tmp / "dup.npy"
    np.save(mp, np.zeros(n, bool))
    cfg3 = {"regions": {"fit_set_exclusion": {"path": str(mp), "sha256": "0" * 64}}, "cache": {}}
    try:
        M.fit_set_mask(cfg3, n, T)
        ok3 = False
    except RuntimeError:
        ok3 = True
    cfg4 = {"regions": {"fit_set_exclusion": {"path": None}}, "manifest_row": {"dataset": "CIC2017"}, "cache": {}}
    try:
        M.fit_set_mask(cfg4, n, T)
        ok4 = False
    except RuntimeError:
        ok4 = True
    ck.check("h9.fail_closed", ok and ok3 and ok4, "x con otro sha / mascara con sha distinto / CIC sin mascara -> error")


def make_synth_job(root, job_id, cache_key, scores=None, labels=None, t4e=False):
    """Job sintetico: fila real del manifiesto, regiones de run_job.regions_for (como en produccion), etiquetas y
    auxiliares reales de la cache, scores SINTETICOS (o los dados)."""
    import run_job as RJ
    row = _manifest_row(job_id)
    spec = C.Spec(row, False)
    cdir = CACHE / cache_key
    meta = json.loads((cdir / "meta.json").read_text())
    y = np.load(cdir / "y.npy")
    N = len(y)
    T = C.corrected_train_rows(spec)
    cmeta = dict(meta)
    if meta.get("aux"):
        cmeta["aux"] = {k: dict(a, path=str(cdir / a["file"])) for k, a in meta["aux"].items()}
    t4 = None
    if t4e:
        lce = LC["knad_active_wiretap"]
        reg0 = RJ.regions_for(spec, N, {"fm": 50_000, "ad": T - 50_000}, cmeta)
        t4 = _t4e_info(lce, reg0)
    reg = RJ.regions_for(spec, N, {"fm": 50_000, "ad": T - 50_000}, cmeta, t4e=t4)
    J = root / "results" / job_id
    J.mkdir(parents=True)
    s = synth_scores(N, 123) if scores is None else scores
    np.save(J / "scores.npy", s)
    np.save(J / "labels.npy", (y if labels is None else labels).astype(np.int8))
    (J / "arrays.json").write_text(json.dumps({"scores.npy": {"sha256": C.sha256_file(J / "scores.npy"), "n_nan": 0,
                                                              "n_inf": 0}}))
    cfg = {"job_id": job_id, "manifest_row": row, "spec": spec.public(), "regions": reg,
           "cache": {"path": str(cdir), "x": meta["x"], "y_sha256": meta["y"]["sha256"]},
           "code": {"runner_tag": "sintetico-test_h"}}
    (J / "config.json").write_text(json.dumps(cfg, default=str))
    (J / "DONE").write_text("sintetico\n")
    return J


def _run_cli(J, out, extra=()):
    t0 = time.time()
    p = subprocess.run([PY, str(RUNNER / "metrics_rev.py"), "--job-dir", str(J), "--out-dir", str(out), *extra],
                       capture_output=True, text=True, timeout=1800,
                       env=dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4")))
    return p, time.time() - t0


def h10(ck, tmp, slow=False):
    cases = [("T1-A-CIC2017__cic2017__stat__arT__corrected.v1__L800__k4__sA__r1", "cic2017__corrected__v1", "cic2017"),
             ("T1-A-ACI__aci-iot-2023__mlp__arF__corrected.v1__L800__k4__sA__r1", "aci-iot-2023__corrected__v1", "aci")]
    if slow:
        cases.append(("T1-A-CIC2018__cic2018__original__arT__corrected.v1__L800__k4__sA__r1", "cic2018__corrected__v1",
                      "cic2018"))
    for job_id, ck_key, key in cases:
        try:
            J = make_synth_job(tmp, job_id, ck_key)
        except KeyError:
            ck.warn(f"h10.{key}", f"{job_id} no esta en el manifiesto")
            continue
        out = tmp / "out" / job_id
        p, dt_ = _run_cli(J, out)
        mj = out / "metrics.json"
        if p.returncode != 0 or not mj.exists():
            ck.check(f"h10.{key}.cli", False, f"rc={p.returncode}; {p.stdout[-800:]} {p.stderr[-1500:]}")
            continue
        m = json.loads(mj.read_text())
        lce = LC[key]
        fs = m["regions"]["fit_set"]
        ok = [m["status"] == "ok", not m["section_errors"], m["label_consistency"]["all_soft_match"],
              fs["n_fit"] == lce["calib"]["fit_rows_dedup"], "aux_dup_train" in fs["mask_source"],
              m["per_day"]["available"], m["per_day"]["label_extras_check"]["matches"]]
        det = {"soft": m["label_consistency"]["all_soft_match"], "n_fit": fs["n_fit"],
               "per_day_match": m["per_day"]["label_extras_check"]["matches"], "t": round(dt_, 1),
               "errores": m["section_errors"]}
        if key == "cic2017":
            sd = m["sensitivities"]["S-DUP"]
            ok += [sd["applicable"], sd["n_removed"] == lce["eval"]["dup_of_train_rows"] == 2985,
                   sd["n_removed_attack"] == lce["eval"]["dup_of_train_attack_rows"] == 351,
                   m["t4p"]["b"]["applicable"], m["t4p"]["a"]["applicable"] is False,
                   m["operational"]["G10000"]["duration_hours"] is None,
                   m["exploratory"]["fpr_by_block_fus"]["b"] == 10_000,
                   m["per_day"]["days"]["Mon"]["adjacent_to_calibration"]]
            det["S-DUP"] = [sd["n_removed"], sd["n_removed_attack"]]
            dc = m["sensitivities"]["S-DUPCAL"]
            ok += ["error" not in dc, dc["applicable"], dc["n_E_rows_with_twin_in_F_cal"] > 0,
                   dc["n_F_cal_rows_with_twin_in_E"] > 0, "FUS" in dc, dc["tag"].startswith("[EXPLORATORIO]")]
            det["S-DUPCAL"] = [dc.get("n_E_rows_with_twin_in_F_cal"), dc.get("n_F_cal_rows_with_twin_in_E"),
                               dc.get("seconds")]
        if key == "aci":
            a = m["t4p"]["a"]
            ok += [a["applicable"], a["estimable"] is False, a["n_clean"] == lce["calib"]["clean_rows_guard800_dedup"] == 139,
                   m["sensitivities"]["S-PS"]["applicable"] is False, m["t4p"]["b"]["applicable"] is False,
                   m["regions"]["T"] == 72_264, m["operational"]["G10000"]["duration_hours"] is not None]
            det["T4-P-a"] = [a["n_clean"], a["estimable"]]
            dc = m["sensitivities"]["S-DUPCAL"]
            ok += ["error" not in dc, dc["applicable"] is False, dc["n_E_rows_with_twin_in_F_cal"] == 0]
            det["S-DUPCAL"] = [dc.get("n_E_rows_with_twin_in_F_cal"), dc.get("n_F_cal_rows_with_twin_in_E")]
        if key == "cic2018":
            a = m["t4p"]["a"]
            sd = m["sensitivities"]["S-DUP"]
            ok += [a["estimable"], a["n_clean"] == lce["calib"]["clean_rows_guard800_dedup"],
                   sd["n_removed"] == lce["eval"]["dup_of_train_rows"], m["regions"]["fit_set"]["n_dup_excluded"] == 45_563]
            det["T4-P-a"] = [a["n_clean"], a["estimable"]]
        ck.check(f"h10.{key}.extremo_a_extremo", all(ok), f"{det}; checks={ok}")
    src = RES / JOB_AW_STAT
    if (src / "scores.npy").exists():
        jid = "T4-E__knad-active_wiretap__original__arT__corrected.v1__L800__k4__c20__sA__r1"
        J = make_synth_job(tmp, jid, "knad-active_wiretap__corrected__v1", scores=np.load(src / "scores.npy"),
                           labels=np.load(src / "labels.npy"), t4e=True)
        out = tmp / "out" / jid
        p, dt_ = _run_cli(J, out)
        mj = out / "metrics.json"
        if p.returncode != 0 or not mj.exists():
            ck.check("h10.t4e.cli", False, f"rc={p.returncode}; {p.stdout[-800:]} {p.stderr[-1500:]}")
        else:
            m = json.loads(mj.read_text())
            ex = LC["knad_active_wiretap"]["t4p_donor_pool"]
            ck.check("h10.t4e.segmentos", m["regions"]["mode"] == "with_exclusions" and len(m["regions"]["E_segments"]) == 2
                     and m["regions"]["E_rows"] == ex["remaining_eval_rows"] and m["t4p"]["applicable"] is False and
                     not m["section_errors"], f"E {m['regions']['E_segments']}; {round(dt_, 1)} s")
    jid = "T1-A-ACI__aci-iot-2023__mlp__arF__corrected.v1__L800__k4__sA__r1"
    try:
        sa = synth_scores(len(np.load(CACHE / "aci-iot-2023__corrected__v1" / "y.npy")), 321)
        sa[[60_000, 60_001]] = np.nan
        sa[65_000] = np.inf
        J = make_synth_job(tmp / "f06", jid, "aci-iot-2023__corrected__v1", scores=sa)
    except KeyError:
        J = None
    if J is not None:
        out = tmp / "out_f06" / jid
        p, dt_ = _run_cli(J, out)
        mj = out / "metrics.json"
        if p.returncode != 0 or not mj.exists():
            ck.check("h10.f06.cli", False, f"rc={p.returncode}; {p.stdout[-800:]} {p.stderr[-1500:]}")
        else:
            m = json.loads(mj.read_text())
            si = m["sensitivities"]["S-INS"]
            fm, (a, b) = m["regions"]["fm"], si["regions"]["calib"]
            s_c = M.sanitize_scores(sa, fm)[0][a:b]
            s_0 = M.sanitize_scores(sa, m["regions"]["T"])[0][a:b]
            ok = (m["nonfinite"]["replaced"] == 3 and not m["section_errors"] and a <= 60_000 < b and
                  si["params"]["mvd"]["mu_ref"] == float(np.mean(s_c)) and float(np.mean(s_0)) != float(np.mean(s_c)))
            ck.check("h10.f06_s_ins_causal", ok, f"S-INS calib [{a}, {b}); mu_ref = media de la serie causal desde fm "
                                                 f"({round(dt_, 1)} s)")
    first, log = M.read_chronology(tmp / "out")
    n_mj = len(list((tmp / "out").glob("*/metrics.json")))
    ck.check("h10.cronologia_F10", first is not None and len(log) == n_mj and n_mj >= 1 and
             first["created"] == min((e["created"] for e in log), key=M._iso_aware),
             f"primer metrics.json registrado una vez (O_EXCL) y {len(log)} entradas en el log")


def h11(ck, tmp):
    for jid in (JOB_AW_STAT, JOB_AW_LSTM):
        J = RES / jid
        if not (J / "DONE").exists():
            ck.warn(f"h11.{jid}", "no DONE")
            continue
        out = tmp / "real" / jid
        p, dt_ = _run_cli(J, out)
        mj = out / "metrics.json"
        ok = p.returncode == 0 and mj.exists()
        det = f"rc={p.returncode}; {dt_:.1f} s"
        if ok:
            m = json.loads(mj.read_text())
            want_ref = jid == JOB_AW_LSTM
            has_ref = m["t4e_reference"] is not None and "error" not in (m["t4e_reference"] or {})
            ok = (m["status"] == "ok" and not m["section_errors"] and m["label_consistency"]["all_soft_match"] and
                  m["primary"]["abs_diff"] <= 1e-12 and has_ref == want_ref
                  and m["provenance"]["plan"]["plan_sha256"] == M.plan_guard()["plan_sha256"])
            det += f"; tiempos {m['timing']}; t4e_reference={'si' if m['t4e_reference'] else 'no'}"
        else:
            det += f" {p.stdout[-600:]} {p.stderr[-1200:]}"
        ck.check(f"h11.real_{jid.split('__')[1]}_{jid.split('__')[2]}", ok, det)


def h12(ck, tmp):
    """(--o1) Ruta O1 de metrics_rev (codigo detectors_2025 sin cambios + replica WD/Pk uint8) sobre el canario X2 de
    OS Scan (scores = s0 bit a bit; labels = target.npy[:N]) frente a los JSON enviados (MVD y KMD, AR - STAT)."""
    J = RES / X2_JOB
    if not (J / "scores.npy").exists():
        ck.warn("h12", f"sin {J}")
        return
    out = tmp / "o1.json"
    code = ("import sys, json; sys.path.insert(0, %r); import metrics_rev as M; "
            "r = M.compute_o1(%r, wd_replica=True, _test_any_job=True); json.dump(r, open(%r, 'w'))"
            % (str(RUNNER), str(J), str(out)))
    t0 = time.time()
    p = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True, timeout=3600,
                       env=dict(os.environ, OMP_NUM_THREADS="4", MPLBACKEND="Agg"))
    if p.returncode != 0 or not out.exists():
        ck.check("h12.o1_canario", False, f"rc={p.returncode} {p.stderr[-1500:]}")
        return
    r = json.loads(out.read_text())
    worst, bad = {}, []
    for det, sub in (("MVD", "experiments"), ("KMD", "experiments_kmeans")):
        pub = json.loads((D25 / sub / "os_scan" / "__Autoregressive - STAT_metrics.json").read_text())
        for k in pub:
            d = abs(float(r[det][k]) - float(pub[k]))
            worst[f"{det}.{k}"] = d
            tol = 3e-8 if ("Window Diff" in k or k.startswith("Pk")) else 1e-6
            if d > tol:
                bad.append(f"{det}.{k}: {d:.3g}")
    ck.check("h12.o1_canario_vs_json_enviados", not bad and r["selected"]["detector"] in ("MVD", "KMD"),
             f"40 campos; max|d| = {max(worst.values()):.3g}; exactos {sum(v == 0 for v in worst.values())}/40; "
             f"fuera de tol {bad[:4]}; {time.time() - t0:.0f} s (wd_impl = replica)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--full-r2", action="store_true")
    ap.add_argument("--slow", action="store_true")
    ap.add_argument("--real-jobs", action="store_true")
    ap.add_argument("--o1", action="store_true", help="h12: ruta O1 sobre el canario X2 (~10 min, /usr/bin/python3)")
    a = ap.parse_args()
    ck = Checker("test_h_metrics_rev")
    tmp = Path(tempfile.mkdtemp(prefix="jisa_test_h_"))
    ck.info(f"raiz temporal {tmp}")
    tests = {"h1": h1, "h2": h2, "h3": lambda c, t: h3(c, t, a.full_r2), "h4": h4, "h5": h5, "h6": h6, "h7": h7,
             "h8": h8, "h9": h9, "h10": lambda c, t: h10(c, t, a.slow)}
    if a.real_jobs:
        tests["h11"] = h11
    if a.o1:
        tests["h12"] = h12
    only = set(a.only.split(",")) if a.only else None
    try:
        for name, fn in tests.items():
            if only and name not in only:
                continue
            t0 = time.time()
            try:
                sub = tmp / name
                sub.mkdir()
                fn(ck, sub)
            except Exception as e:
                import traceback
                traceback.print_exc()
                ck.check(f"{name}.excepcion", False, f"{type(e).__name__}: {e}")
            ck.info(f"{name}: {time.time() - t0:.1f} s")
    finally:
        if os.environ.get("JISA_KEEP_TEST_ROOT") != "1" and ck.ok:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            ck.info(f"raiz temporal conservada: {tmp}")
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
