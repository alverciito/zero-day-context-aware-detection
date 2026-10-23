#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics and exploratory statistics of the per-record MLP trained in stream order against the shuffled MLP."""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

R = Path("/home/acid137/Projects/review-zd")
sys.path.insert(0, str(R / "revision" / "experiments"))

import common as C
import metrics_rev as MR
import stats as ST

RESULTS = R / "results" / "revision"
AGG_DIR = RESULTS / "final-20261007-1206"
OUT = R / "revision" / "analysis" / "mlp_sin_barajar_metricas.json"
OUT_TEX = R / "revision" / "analysis" / "mlp_sin_barajar_metricas.tex"
ABREV = {"D1": "AW", "D2": "Fuzz", "D3": "OS", "D4": "SSDP", "D5": "SSL", "D6": "Video", "D7": "ACI",
         "D8": "CIC17", "D9": "CIC18"}
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
TAG = "[EXPLORATORIO]"
TAG_LARGO = ("[EXPLORATORIO] sensibilidad posterior a la congelacion con el MLP por registro entrenado sin barajar "
             "en lugar de C9; fuera del plan congelado, sin 'significant'")
BLOCK, BLOCK_CTRL = "T9-MLPORD", "T9-MLPORD-CTRL"
DATASETS = ("ACI-IOT-2023", "KNAD:Active_Wiretap", "KNAD:Fuzzing", "KNAD:OS_Scan", "KNAD:SSDP_Flood",
            "KNAD:SSL_Renegotiation", "KNAD:Video_Injection", "CIC2017", "CIC2018")
SEEDS = ("A", "B")
CHECK_KEYS = ("auroc_FUS", "auc_loc_RAW", "auroc_KMD_f", "auroc_MVD_f", "auroc_RAW", "auprc_FUS", "f1_FUS",
              "fpr_FUS", "auc_t")
TOL = 1e-12


def corpus(ds):
    """Corpus del bloque T1 de un dataset."""
    return "KNAD" if ds.upper().startswith("KNAD:") else {"ACI-IOT-2023": "ACI"}.get(ds, ds)


def ref_job_id(ds, seed):
    """Job T1 barajado de referencia (MLP, arF, misma semilla)."""
    return "T1-%s-%s__%s__mlp__arF__corrected.v1__L800__k4__s%s__r1" % (seed, corpus(ds), C.ds_slug(ds), seed)


def t9_job_id(ds, seed, control=False):
    """Job identifier of a run of the MLP trained in stream order, or of its control run with the change off."""
    return "%s__%s__%s__arF__corrected.v1__L800__k4__s%s__r1" % (
        BLOCK_CTRL if control else BLOCK, C.ds_slug(ds), "mlp" if control else "mlp_ord", seed)


def metricas_job(job_dir):
    """Metricas del protocolo principal de un job, por el mismo camino que metrics_rev.compute_job.

    Regiones solo via common.check_protocol_regions (candidata primaria del 10 %); F_cal sin
    los duplicados exactos del entrenamiento; no finitos saneados como en §12.2; ajuste sin
    etiquetas (`fit`), series causales (`transform`) y evaluacion en E.

    :param job_dir: directorio del job (scores.npy, labels.npy, config.json).
    :type job_dir: Path
    :return: metricas del job y lo necesario para auditar que la region es la de la rejilla.
    :rtype: dict
    """
    from sklearn.metrics import roc_auc_score

    J = Path(job_dir)
    cfg = C.read_json(J / "config.json")
    if not cfg:
        raise RuntimeError("config.json ilegible en %s" % J)
    s = np.load(J / "scores.npy")
    labels = np.load(J / "labels.npy")
    if s.dtype != np.float64 or s.ndim != 1:
        raise RuntimeError("scores.npy %s/%s: se esperaba float64 (N,)" % (s.dtype, s.shape))
    N = int(len(s))
    regions = cfg["regions"]
    if labels.shape != (N,) or int(regions.get("n_rows", -1)) != N:
        raise RuntimeError("labels.npy o regions.n_rows no casan con len(scores) = %d" % N)
    T = int(regions["train"][1])
    y = labels.astype(bool)
    calib, E, _info = MR.get_regions(regions, MR.PRIMARY_CANDIDATE)
    if calib[0] != T:
        raise RuntimeError("calibracion primaria %s no empieza en T = %d" % (calib, T))
    E_rows = MR.rows_of(E)
    yE = y[E_rows]
    dup, fsinfo = MR.fit_set_mask(cfg, N, T, log=lambda m: None)
    fcal = np.zeros(N, bool)
    fcal[calib[0]:calib[1]] = True
    fcal &= ~dup
    s_det, s_raw, nf = MR.sanitize_scores(s, T)
    frac = float((~np.isfinite(s[T:])).mean()) if N > T else 0.0
    params = MR.fit(s_det[T:calib[1]], fcal[T:calib[1]], T)
    ser = MR.transform(params, s_det[T:])
    ser["RAW"] = s_raw[T:]
    taus = MR._thr(params)
    main = MR._series_eval(ser, T, E_rows, yE, MR.X_MAIN, taus, o2=False, wd=False)
    sk = float(roc_auc_score(yE, ser["FUS"][E_rows - T]))
    if abs(sk - main["FUS"]["auroc"]) > TOL:
        raise RuntimeError("replica de ROC-AUC %r != sklearn %r (MR-1)" % (main["FUS"]["auroc"], sk))
    loc = MR.auc_loc(ser["RAW"][E_rows - T], yE)
    triv = main["trivial"]
    out = {"auroc_FUS": sk, "auc_loc_RAW": loc["auc_loc"], "auc_t": triv["auc_t"],
           "prevalence": triv["prevalence"], "n_E": int(len(E_rows)),
           "calib": [int(calib[0]), int(calib[1])], "eval_segments": MR.segments_of(E), "T": T, "N": N,
           "n_fit": int(fcal.sum()), "n_dup_excluded": int((calib[1] - calib[0]) - fcal.sum()),
           "frac_nonfinite_exec": frac, "divergent": bool(frac > MR.DIVERGENCE_FRAC),
           "nonfinite_replaced": nf.get("replaced")}
    for x in MR.X_MAIN:
        for k in ("auroc", "auprc", "eer", "f1", "precision", "recall", "fpr", "threshold"):
            out["%s_%s" % (k, x)] = main[x].get(k)
    return out


def metricas_de_metrics_json(job_dir):
    """Las mismas claves, leidas del metrics.json que escribio metrics_rev.py para un job T1."""
    m = C.read_json(Path(job_dir) / "metrics.json")
    if not m:
        return None
    E = m["E"]
    return {"auroc_FUS": m["primary"]["value"], "auc_loc_RAW": m["primary"].get("auc_loc_raw"),
            "auroc_KMD_f": E["KMD_f"]["auroc"], "auroc_MVD_f": E["MVD_f"]["auroc"], "auroc_RAW": E["RAW"]["auroc"],
            "auprc_FUS": E["FUS"]["auprc"], "f1_FUS": E["FUS"]["f1"], "fpr_FUS": E["FUS"]["fpr"],
            "auc_t": E["trivial"]["auc_t"],
            "helpers_sha256": (m.get("provenance") or {}).get("helpers_sha256"),
            "version_key": (m.get("provenance") or {}).get("version_key")}


def validar_job_t9(d, ds, seed, control=False):
    """Comprueba que un job T9 es lo que dice ser antes de usar sus scores.

    :return: (dict con la procedencia que interesa, lista de motivos de rechazo).
    """
    cfg = C.read_json(d / "config.json") or {}
    t9 = cfg.get("t9_mlpord") or {}
    vs = t9.get("vs_reference") or {}
    fits = t9.get("fits") or []
    visto = [(f.get("data_handlers") or [{}])[0].get("shuffle_recibido") for f in fits]
    malos = []
    if not (d / "DONE").exists():
        malos.append("sin DONE")
    if bool(t9.get("control")) != bool(control):
        malos.append("control=%r" % t9.get("control"))
    if t9.get("ensayo_cpu") or cfg.get("gpu_arg") != "0":
        malos.append("no es una ejecucion en /GPU:0")
    if (cfg.get("reference") or {}).get("job_id") != ref_job_id(ds, seed):
        malos.append("referencia %r" % (cfg.get("reference") or {}).get("job_id"))
    if not t9.get("all_fits_assert_ok") or len(fits) != 4:
        malos.append("assert del fit no esta en verde en los 4 autoencoders")
    if visto != [bool(control)] * 4:
        malos.append("Keras vio shuffle=%r, se esperaba %r" % (visto, [bool(control)] * 4))
    if t9.get("requirements_failed"):
        malos.append("requisitos incumplidos: %s" % t9["requirements_failed"])
    for k, sub in (("feature_map", "identical"), ("normalization", "all_identical"), ("labels.npy", "identical")):
        if not (vs.get(k) or {}).get(sub):
            malos.append("%s no identico al del job T1" % k)
    if not vs.get("regions_identical"):
        malos.append("regiones distintas de las del job T1")
    prov = {"harness_sha256": (t9.get("harness") or {}).get("sha256"),
            "runner_tag": (cfg.get("code") or {}).get("runner_tag"), "shuffle_seen_by_keras": visto,
            "scores_sha256": (vs.get("scores") or {}).get("sha256"),
            "scores_identical_to_reference": (vs.get("scores") or {}).get("identical"),
            "control_identico": t9.get("control_identico"),
            "done": (d / "DONE").read_text().strip() if (d / "DONE").exists() else None,
            "wall_seconds": (C.read_json(d / "timing.json") or {}).get("wall_seconds"),
            "anomalies": cfg.get("anomalies")}
    return prov, malos


def resumen_comparacion(c):
    """Lo que se cita de una comparacion de stats.compare / stats.family."""
    w = c.get("wilcoxon") or {}
    ci = c.get("hl_ci") or {}
    rob = c.get("robustness") or {}
    return {"contrast": c.get("contrast"), "n": c.get("n"), "n_eff": w.get("n_eff"), "datasets": c.get("datasets"),
            "d": c.get("d"), "hl": c.get("hl"), "ci_lower": ci.get("lower"), "ci_upper": ci.get("upper"),
            "ci_coverage": ci.get("coverage"), "W": w.get("W"), "T_plus": w.get("t_plus"), "p_exact": w.get("p"),
            "testable": w.get("testable"), "p_holm": c.get("p_holm"), "holm_le_0.05": c.get("reject"),
            "status": c.get("status"), "k_pos": c.get("k_pos"), "k_neg": c.get("k_neg"), "k_zero": c.get("k_zero"),
            "robust_rule_5.8": c.get("robust"),
            "drift_robust": (rob.get("drift") or {}).get("robust"),
            "hl_auc_loc_raw": (rob.get("drift") or {}).get("hl_auc_loc_raw"),
            "dependence_ok": ((rob.get("dependence") or {}).get("rule") or {}).get("dependence_ok"),
            "wording_rule": (c.get("wording") or {}).get("rule")}


def reproduce(fam_new, fam_ref):
    """True si una familia recalculada coincide con la del stats.json FINAL (hl, p, Holm, k, robustez)."""
    dif = {}
    for name, c in fam_ref["comparisons"].items():
        n = fam_new["comparisons"].get(name) or {}
        for k in ("hl", "p_holm", "k_pos", "reject", "robust"):
            if n.get(k) != c.get(k):
                dif["%s.%s" % (name, k)] = [n.get(k), c.get(k)]
        if (n.get("wilcoxon") or {}).get("p") != (c.get("wilcoxon") or {}).get("p"):
            dif["%s.p" % name] = [(n.get("wilcoxon") or {}).get("p"), (c.get("wilcoxon") or {}).get("p")]
    return (not dif), dif


def escribir_tex(out, path):
    """Escribe la tabla del suplemento y el parrafo del manuscrito con las cifras del JSON.

    Todo numero sale de `out` (el mismo dict que va al JSON), de modo que el texto no se
    transcribe a mano. Si faltan datasets, el fichero lo dice en la cabecera y NO debe pegarse.

    :param dict out: contenido de mlp_sin_barajar_metricas.json.
    :param path: fichero .tex de salida.
    :type path: Path
    :return: True si el fichero corresponde a los 9 datasets completos.
    :rtype: bool
    """
    comp = out["datasets_complete"]
    n = len(comp)
    if not n:
        return False

    def f3(v):
        return "NA" if v is None else "%.3f" % v

    def s3(v):
        return "NA" if v is None else "$%+.3f$" % v

    def it(txt):
        return "{%s}" % txt

    pd_, med = out["per_dataset"], out["medians"]
    e = out["paired_ordered_minus_shuffled"]["auroc_FUS"]
    lstm = out["H4_with_ordered_MLP"]["comparisons"]["MLP_vs_LSTM"]
    orig = out["H1_with_ordered_MLP"]["comparisons"]["MLP_vs_original"]
    lo, lr_, oo, or_ = (lstm["with_ordered_MLP"], lstm["confirmatory_with_shuffled_MLP"],
                        orig["with_ordered_MLP"], orig["confirmatory_with_shuffled_MLP"])
    integ = out["arm_integrity"]
    control_ok = bool(integ["runner_tags_used"]) and set(integ["runner_tags_used"]) <= set(
        integ["runner_tags_with_identical_control"])
    cov = "%.1f" % (100.0 * e["ci_coverage"]) if e.get("ci_coverage") else "NA"
    n_ctrl = sum(1 for c in integ["controls"] if c.get("control_identico") and not c.get("problemas"))
    seccion = ("\\section{Per-record MLP trained in stream order}\n\nSection~6.1 of the manuscript declares that the "
               "per-record MLP is the only configuration of the grid whose autoencoders are trained on shuffled "
               "records. Table~\\ref{tab:s_mlpord} gives the exploratory experiment that isolates that factor: the %d "
               "grid runs of the per-record MLP (%s streams, two seeds) repeated with the training records in stream "
               "order. Nothing else changes: each run uses the runner version, the stream, the feature map, the seed "
               "and the hyperparameters of the grid run it repeats, and the only modification is that the fitting "
               "call receives the records unshuffled, which is checked at run time for each of the four "
               "autoencoders.%s The experiment was run after the analysis plan was frozen, so it is outside the "
               "confirmatory families: the main table and the hypothesis tests keep the shuffled model."
               % (2 * n, NUM.get(n, str(n)),
                  (" With the modification switched off, the same harness reproduces the scores of the grid run bit "
                   "for bit in the %s control runs, one per runner version used by the grid." % NUM.get(n_ctrl, str(n_ctrl)))
                  if control_ok else ""))
    L = ["%% GENERADO por revision/analysis/mlp_sin_barajar_metricas.py el %s. No editar a mano." % out["created"],
         "% [EXPLORATORIO] T9-MLPORD (MLP por registro sin barajar). Fuente de cada cifra:",
         "%% revision/analysis/mlp_sin_barajar_metricas.json (sha256 del guion %s)." % out["provenance"]["script_sha256"][:16],
         "%% Datasets completos: %d de 9. Jobs sin terminar: %d. Rechazados: %d." % (n, out["n_pending"], len(out["rejected"]))]
    if n < 9:
        L += ["% >>>>>> PARCIAL: FALTAN DATASETS <<<<<<"]
    if not control_ok:
        L += ["% >>>>>> AVISO: hay versiones del runner sin control positivo identico; la frase 'bit for bit' se omite."]
    for w in out["warnings"]:
        L += ["%% AVISO: %s" % w]
    L += ["", "% --- (1) SUPLEMENTO: seccion nueva tras 'Retraining experiment: the three declared changes' ------------------",
          seccion, "",
          "\\begin{table}[ht]\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{lrrrrr}", "\\hline",
          "\\textbf{Stream} & \\textbf{Shuffled} & \\textbf{In order} & \\textbf{Difference} & "
          "\\textbf{Seeds, shuffled} & \\textbf{Seeds, in order}\\\\", "\\hline"]
    cols = {"s": [], "o": [], "d": [], "rs": [], "ro": []}
    for did in ST.DS:
        if did not in comp:
            continue
        x = pd_[did]
        vals = (x["shuffled_C9"]["auroc_FUS"], x["ordered"]["auroc_FUS"], x["diff_ordered_minus_shuffled"]["auroc_FUS"],
                x["seed_range_shuffled"], x["seed_range_ordered"])
        for k, v in zip(("s", "o", "d", "rs", "ro"), vals):
            cols[k].append(v)
        L.append("%s & %s & %s & %s & %s & %s\\\\" % (ABREV[did], f3(vals[0]), f3(vals[1]), s3(vals[2]), f3(vals[3]),
                                                     f3(vals[4])))
    L += ["\\hline",
          "Median & %s & %s & %s & %s & %s\\\\" % tuple([f3(float(np.median(cols["s"]))), f3(float(np.median(cols["o"]))),
                                                        s3(float(np.median(cols["d"]))), f3(float(np.median(cols["rs"]))),
                                                        f3(float(np.median(cols["ro"])))]),
          "\\hline", "\\end{tabular}",
          "\\caption{Exploratory experiment, run after the analysis plan was frozen. ROC-AUC of "
          "the fused detector for the per-record MLP of the main grid, whose autoencoders are "
          "trained on shuffled records, and for the same model trained on the records in stream "
          "order, with the stream, the feature map, the seed, the normalisation, the learning "
          "rate, the batch size and the single epoch unchanged. Each value is the mean of the "
          "two seeds, the difference is in order minus shuffled, and the last two columns give "
          "the absolute difference between the two seeds. The last row is the median of each "
          "column over the %s datasets, so its difference is the median of the differences and "
          "not the difference of the medians.}\\label{tab:s_mlpord}" % NUM.get(n, str(n)),
          "\\end{table}"]
    rank = med.get("ordered_rank_among_11")
    if rank is None:
        lugar = ""
    elif rank == 1:
        lugar = ", and its median is still the highest of the eleven configurations"
    else:
        lugar = ", which places its median %s among the eleven configurations" % {2: "second", 3: "third", 4: "fourth",
                                                                              5: "fifth"}.get(rank, "%dth" % rank)
    frase_control = (", and with the change switched off the same harness reproduces the scores of the grid run "
                     "bit for bit" if control_ok else "")
    P = ("\\rev{The ordering can also be isolated. In an exploratory experiment, run after the analysis plan was "
         "frozen, we repeated the per-record MLP of the grid on the %s datasets and the two seeds with a single "
         "change: its autoencoders receive the training records in stream order instead of shuffled. The stream, the "
         "feature map, the seed, the normalisation, the learning rate, the batch size and the single epoch are those "
         "of the grid run%s. Trained in stream order, the per-record MLP has a median ROC-AUC of %s over the %s "
         "datasets, against %s when shuffled%s. The paired difference, in order minus shuffled, has a Hodges--Lehmann "
         "estimate of %s (unadjusted %s\\%% CI [%s, %s]; exploratory, unadjusted exact $p = %s$), and the in-order "
         "model is lower on %s and higher on %s of the %s datasets. With the in-order MLP in place of the shuffled "
         "one, its difference from the recurrent family is %s (unadjusted %s\\%% CI [%s, %s]; exact $p = %s$, %s "
         "after the Holm adjustment within the four comparisons of that family; higher on %s of %s), against %s and "
         "%s with the shuffled MLP, and its difference from the KitNET-style control is %s, against %s. These figures "
         "are exploratory and no confirmatory claim rests on them; the values per dataset are in the supplementary "
         "material.}"
         % (NUM.get(n, str(n)), frase_control, it(f3(med.get("ordered_median"))), NUM.get(n, str(n)),
            it(f3(med.get("shuffled_median_same_datasets"))), lugar, it(s3(e["hl"])), cov, it(s3(e["ci_lower"])),
            it(s3(e["ci_upper"])), it(f3(e["p_exact"])), it(str(e["k_neg"])), it(str(e["k_pos"])), NUM.get(n, str(n)),
            it(s3(lo["hl"])), cov, it(s3(lo["ci_lower"])), it(s3(lo["ci_upper"])), it(f3(lo["p_exact"])),
            it(f3(lo["p_holm"])), it(str(lo["k_pos"])), NUM.get(lo["n"], str(lo["n"])), it(s3(lr_["hl"])),
            it(f3(lr_["p_holm"])), it(s3(oo["hl"])), it(s3(or_["hl"]))))
    L += ["", "% --- (2) MANUSCRITO, Seccion 6.1: tras '...so it gives an upper bound on the effect of the ordering and "
              "does not isolate it.' ---",
          "% The paragraph below gives the paired difference, in order minus shuffled, with its interval and "
          "its exact p-value.", P, ""]
    C.write_text_atomic(path, "\n".join(L) + "\n")
    return n == 9


def main(argv=None):
    ap = argparse.ArgumentParser(description="Metricas y estadistica exploratoria del experimento T9-MLPORD (MLP por registro entrenado sin barajar).")
    ap.add_argument("--aggregate-dir", default=str(AGG_DIR), help="agregado FINAL (aggregate/aggregate.json, stats.json)")
    ap.add_argument("--recalcular", action="store_true", help="ignora las metricas por job guardadas en el JSON")
    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        env = MR.env_guard(R)
        plan = MR.plan_guard(R)
    except (MR.EnvGuardError, MR.PlanGuardError) as e:
        print("GUARDA: %s" % e)
        return 3
    agg_dir = Path(a.aggregate_dir)
    agg_path, stats_path = agg_dir / "aggregate" / "aggregate.json", agg_dir / "stats.json"
    agg, stats_final = C.read_json(agg_path), C.read_json(stats_path)
    if not agg or not stats_final:
        print("no se puede leer el agregado en %s" % agg_dir)
        return 1
    if stats_final.get("run_type") != "FINAL":
        print("AVISO: %s tiene run_type = %r, no FINAL" % (stats_path, stats_final.get("run_type")))
    V = agg["value_sets"]["main"]
    cells = agg["cells"]
    helpers = MR.helper_hashes()
    previo = (C.read_json(OUT) or {}).get("job_metrics") or {}
    job_metrics, avisos = {}, []

    def calcular(job_dir):
        """Metricas de un job, reutilizando las guardadas si scores y funciones no han cambiado."""
        arr = C.read_json(job_dir / "arrays.json") or {}
        key = hashlib.sha256(json.dumps({"s": (arr.get("scores.npy") or {}).get("sha256"), "h": helpers,
                                         "v": MR.METRICS_REV_VERSION}, sort_keys=True).encode()).hexdigest()
        hit = previo.get(job_dir.name)
        if hit and hit.get("cache_key") == key and not a.recalcular:
            job_metrics[job_dir.name] = hit
            return hit["metrics"]
        t = time.time()
        met = metricas_job(job_dir)
        job_metrics[job_dir.name] = {"cache_key": key, "scores_sha256": (arr.get("scores.npy") or {}).get("sha256"),
                                     "seconds": round(time.time() - t, 1), "metrics": met}
        return met

    filas, pendientes, rechazados, control_calc = [], [], [], []
    for ds in DATASETS:
        did = MR.DATASET_IDS[ds]
        for seed in SEEDS:
            d = RESULTS / t9_job_id(ds, seed)
            if not (d / "DONE").exists():
                pendientes.append(d.name)
                continue
            prov, malos = validar_job_t9(d, ds, seed)
            if malos:
                rechazados.append({"job_id": d.name, "motivos": malos})
                continue
            rdir = RESULTS / ref_job_id(ds, seed)
            mj = metricas_de_metrics_json(rdir)
            if mj is None:
                rechazados.append({"job_id": d.name, "motivos": ["el job T1 de referencia no tiene metrics.json"]})
                continue
            ref = calcular(rdir)
            dif = {k: abs(ref[k] - mj[k]) for k in CHECK_KEYS if ref.get(k) is not None and mj.get(k) is not None}
            ok = bool(dif) and max(dif.values()) <= TOL and mj.get("helpers_sha256") == helpers
            control_calc.append({"job_id": rdir.name, "max_abs_diff": max(dif.values()) if dif else None,
                                 "helpers_sha256_equal": mj.get("helpers_sha256") == helpers, "ok": ok})
            if not ok:
                print("PARADA: la replica del calculo no reproduce %s/metrics.json: %s; helpers iguales: %s"
                      % (rdir.name, dif, mj.get("helpers_sha256") == helpers))
                return 1
            c9 = (((cells.get(did) or {}).get("C9") or {}).get("seeds") or {}).get(seed) or {}
            c9_ok = c9.get("auroc_FUS") is not None and abs(c9["auroc_FUS"] - ref["auroc_FUS"]) <= TOL
            if not c9_ok:
                avisos.append("%s semilla %s: C9 del agregado (%r) != job arF de referencia (%r)"
                              % (ds, seed, c9.get("auroc_FUS"), ref["auroc_FUS"]))
            o = calcular(d)
            if o["divergent"]:
                avisos.append("%s: > 1 %% de scores no finitos en ejecucion; el agregado lo imputaria (AG-5), aqui no"
                              % d.name)
            if [o["calib"], o["eval_segments"], o["n_fit"], o["n_E"]] != [ref["calib"], ref["eval_segments"],
                                                                         ref["n_fit"], ref["n_E"]]:
                print("PARADA: %s no se evalua en la misma region que %s" % (d.name, rdir.name))
                return 1
            filas.append({"dataset": ds, "dataset_id": did, "seed": seed, "job_id": d.name, "reference_job": rdir.name,
                          "ordered": {k: o.get(k) for k in ("auroc_FUS", "auc_loc_RAW", "auroc_RAW", "auroc_KMD_f",
                                                           "auroc_MVD_f", "auprc_FUS", "f1_FUS", "fpr_FUS",
                                                           "recall_FUS", "eer_FUS")},
                          "shuffled": {k: ref.get(k) for k in ("auroc_FUS", "auc_loc_RAW", "auroc_RAW", "auroc_KMD_f",
                                                              "auroc_MVD_f", "auprc_FUS", "f1_FUS", "fpr_FUS",
                                                              "recall_FUS", "eer_FUS")},
                          "diff_ordered_minus_shuffled": {k: o[k] - ref[k] for k in ("auroc_FUS", "auc_loc_RAW",
                                                                                   "auroc_RAW")},
                          "auc_t": o["auc_t"], "prevalence": o["prevalence"], "calib": o["calib"],
                          "eval_segments": o["eval_segments"], "n_E": o["n_E"], "n_fit": o["n_fit"],
                          "same_region_as_reference": True, "c9_seed_value_equals_reference_job": c9_ok,
                          "provenance": prov})

    por_ds, completos = {}, []
    for ds in DATASETS:
        did = MR.DATASET_IDS[ds]
        fs = {f["seed"]: f for f in filas if f["dataset"] == ds}
        s_star = (agg["datasets"].get(did) or {}).get("S_star") or []
        ent = {"dataset": ds, "seeds_done": sorted(fs), "S_star_grid": s_star,
               "complete": sorted(fs) == sorted(s_star) == list(SEEDS)}
        c9 = (V.get(did) or {}).get("C9") or {}
        ent["shuffled_C9"] = {k: c9.get(k) for k in ("auroc_FUS", "auc_loc_RAW")}
        if fs:
            ent["ordered_by_seed"] = {s: fs[s]["ordered"]["auroc_FUS"] for s in sorted(fs)}
            ent["shuffled_by_seed"] = {s: fs[s]["shuffled"]["auroc_FUS"] for s in sorted(fs)}
        if ent["complete"]:
            completos.append(did)
            for k in ("auroc_FUS", "auc_loc_RAW", "auroc_RAW"):
                ent.setdefault("ordered", {})[k] = float(np.mean([fs[s]["ordered"][k] for s in SEEDS]))
                ent.setdefault("shuffled_mean_of_reference_jobs", {})[k] = float(
                    np.mean([fs[s]["shuffled"][k] for s in SEEDS]))
            if abs(ent["shuffled_mean_of_reference_jobs"]["auroc_FUS"] - c9["auroc_FUS"]) > TOL:
                avisos.append("%s: la media de los jobs arF (%r) no es el C9 del agregado (%r)"
                              % (ds, ent["shuffled_mean_of_reference_jobs"]["auroc_FUS"], c9["auroc_FUS"]))
            ent["diff_ordered_minus_shuffled"] = {k: ent["ordered"][k] - ent["shuffled_mean_of_reference_jobs"][k]
                                                  for k in ("auroc_FUS", "auc_loc_RAW", "auroc_RAW")}
            ent["seed_range_ordered"] = abs(fs["A"]["ordered"]["auroc_FUS"] - fs["B"]["ordered"]["auroc_FUS"])
            ent["seed_range_shuffled"] = abs(fs["A"]["shuffled"]["auroc_FUS"] - fs["B"]["shuffled"]["auroc_FUS"])
        por_ds[did] = ent

    emparejada = {}
    for k in ("auroc_FUS", "auc_loc_RAW", "auroc_RAW"):
        d = {did: (por_ds[did]["diff_ordered_minus_shuffled"][k] if did in completos else None) for did in ST.DS}
        c = ST.compare(d, test=True)
        c["contrast"] = "%s(MLP ordenado) - %s(MLP barajado)" % (k, k)
        emparejada[k] = dict(resumen_comparacion(c), tag=TAG + " sin ajustar; un contraste por metrica",
                             p_note="exploratory, unadjusted")

    h1_ref = ST.h1(copy.deepcopy(V))
    h4_ref = ST.h4(copy.deepcopy(V), h1fam=h1_ref)
    ok1, dif1 = reproduce(h1_ref, stats_final["H1"])
    ok4, dif4 = reproduce(h4_ref, stats_final["H4"])
    if not (ok1 and ok4):
        print("PARADA: stats.h1/h4 sobre el agregado no reproducen stats.json: %s %s" % (dif1, dif4))
        return 1
    V2 = copy.deepcopy(V)
    for did in ST.DS:
        nuevo = dict(V2[did].get("C9") or {})
        for k in list(nuevo):
            nuevo[k] = None
        if did in completos:
            nuevo["auroc_FUS"] = por_ds[did]["ordered"]["auroc_FUS"]
            nuevo["auc_loc_RAW"] = por_ds[did]["ordered"]["auc_loc_RAW"]
        V2[did]["C9"] = nuevo
    h1_new = ST.h1(V2, robust=True, tag=TAG_LARGO, confirmatory=False)
    h4_new = ST.h4(V2, robust=True, tag=TAG_LARGO, confirmatory=False)

    def familia(fam_new, fam_ref):
        out = {"m": fam_new["m"], "n_min": fam_new["n_min"], "informative": fam_new["informative"],
               "status": fam_new["status"], "tag": TAG_LARGO, "comparisons": {}}
        for name, c in fam_new["comparisons"].items():
            out["comparisons"][name] = {"with_ordered_MLP": resumen_comparacion(c),
                                        "confirmatory_with_shuffled_MLP": resumen_comparacion(fam_ref["comparisons"][name]),
                                        "involves_MLP": "MLP" in name}
        return out

    H1, H4 = familia(h1_new, h1_ref), familia(h4_new, h4_ref)

    med = {"shuffled_C9_median_9": float(np.median([V[d]["C9"]["auroc_FUS"] for d in ST.DS])),
           "grid_medians": {c: float(np.median([V[d][c]["auroc_FUS"] for d in ST.DS])) for c in ST.CONFIGS}}
    if completos:
        med["ordered_median"] = float(np.median([por_ds[d]["ordered"]["auroc_FUS"] for d in completos]))
        med["ordered_n_datasets"] = len(completos)
        med["shuffled_median_same_datasets"] = float(np.median([V[d]["C9"]["auroc_FUS"] for d in completos]))
    if len(completos) == 9:
        otros = {c: v for c, v in med["grid_medians"].items() if c != "C9"}
        med["ordered_rank_among_11"] = 1 + sum(1 for v in otros.values() if v > med["ordered_median"])
        med["ordered_is_highest_median"] = bool(all(med["ordered_median"] > v for v in otros.values()))

    controles = []
    for d in sorted(RESULTS.glob(BLOCK_CTRL + "__*")):
        cfg = C.read_json(d / "config.json") or {}
        row = cfg.get("manifest_row") or {}
        prov, malos = validar_job_t9(d, row.get("dataset"), row.get("seed"), control=True)
        controles.append(dict(prov, job_id=d.name, dataset=row.get("dataset"), seed=row.get("seed"),
                              problemas=malos))
    harness = sorted({f["provenance"]["harness_sha256"] for f in filas if f["provenance"]["harness_sha256"]})
    tags_usados = sorted({f["provenance"]["runner_tag"] for f in filas})
    tags_validados = sorted({c["runner_tag"] for c in controles if c.get("control_identico") and not c["problemas"]})
    if len(harness) > 1:
        avisos.append("los jobs del experimento se hicieron con %d versiones distintas de mlp_sin_barajar.py" % len(harness))
    sin_control = [t for t in tags_usados if t not in tags_validados]
    if sin_control:
        avisos.append("versiones del runner usadas sin control positivo identico: %s" % sin_control)

    replicas, t_ref, t_ord, pend_ref_s = [], {}, {}, 0.0
    for ds in DATASETS:
        for seed in SEEDS:
            rdir = RESULTS / ref_job_id(ds, seed)
            twin = RESULTS / rdir.name.replace("__arF__", "__arT__")
            sa = ((C.read_json(rdir / "arrays.json") or {}).get("scores.npy") or {}).get("sha256")
            sb = ((C.read_json(twin / "arrays.json") or {}).get("scores.npy") or {}).get("sha256")
            replicas.append(bool(sa) and sa == sb)
            t_ref[rdir.name] = (C.read_json(rdir / "timing.json") or {}).get("wall_seconds")
            d = RESULTS / t9_job_id(ds, seed)
            if (d / "DONE").exists():
                t_ord[rdir.name] = (C.read_json(d / "timing.json") or {}).get("wall_seconds")
            else:
                pend_ref_s += t_ref[rdir.name] or 0.0
    ratios = [t_ord[k] / t_ref[k] for k in t_ord if t_ord[k] and t_ref[k]]
    tiempos = {"reference_T1_wall_seconds_18_jobs": float(sum(v or 0.0 for v in t_ref.values())),
               "reference_T1_wall_seconds_by_job": t_ref, "t9_wall_seconds_by_reference_job": t_ord,
               "t9_over_reference_ratio_median": float(np.median(ratios)) if ratios else None,
               "pending_jobs_reference_wall_seconds": pend_ref_s,
               "pending_jobs_eta_seconds": (pend_ref_s * float(np.median(ratios))) if ratios else None,
               "note": "los jobs T1 corrieron solos en la GPU; la razon de hoy incluye la carga de la maquina. La "
                       "estimacion no cuenta los controles positivos que faltan (uno por version del runner)"}
    determinismo = {"mlp_grid_replicas_arT_arF_identical_scores": int(sum(replicas)), "n_pairs": len(replicas),
                    "meaning": "MLPAutoencoder ignora ar: las dos replicas de cada semilla tienen el mismo sha256 de "
                               "scores.npy, es decir, el job MLP de la rejilla es determinista bit a bit dada la "
                               "semilla. Por eso C9 por semilla = job arF, y por eso el control positivo del arnes "
                               "se puede exigir identico"}

    out = {"schema": "jisa-t9-mlpord-metrics/1", "created": C.now_iso(), "tag": TAG, "decision": "post hoc: per-record MLP trained in stream order",
           "scope": "posterior a la congelacion; fuera de S*_d, de H1-H4 confirmatorias y del agregado FINAL. "
                    "Ninguna cifra de este fichero admite 'significant' (plan §5.6, fila Exploratorio).",
           "n_jobs": len(DATASETS) * len(SEEDS), "n_done_used": len(filas), "n_pending": len(pendientes),
           "pending": pendientes, "rejected": rechazados, "datasets_complete": completos,
           "n_datasets_complete": len(completos), "all_complete": len(completos) == 9,
           "warnings": avisos,
           "arm_integrity": {"all_jobs_shuffle_false": bool(filas) and all(
               f["provenance"]["shuffle_seen_by_keras"] == [False] * 4 for f in filas),
               "harness_sha256": harness, "runner_tags_used": tags_usados,
               "runner_tags_with_identical_control": tags_validados, "controls": controles},
           "calculation_check": {"rule": "el mismo camino sobre el job T1 barajado reproduce su metrics.json "
                                         "(|dif| <= 1e-12 en %s) con los mismos helpers" % list(CHECK_KEYS),
                                 "jobs": control_calc, "all_ok": all(c["ok"] for c in control_calc),
                                 "stats_h1_h4_reproduce_final": bool(ok1 and ok4)},
           "per_job": filas, "per_dataset": por_ds, "paired_ordered_minus_shuffled": emparejada,
           "H1_with_ordered_MLP": H1, "H4_with_ordered_MLP": H4, "medians": med,
           "grid_determinism": determinismo, "timing": tiempos,
           "job_metrics": job_metrics,
           "provenance": {"script": str(Path(__file__).resolve()), "script_sha256": C.sha256_file(__file__),
                          "command": "nice -n 19 env OMP_NUM_THREADS=4 %s %s" % (sys.executable, Path(__file__).resolve()),
                          "metrics_rev_helpers_sha256": helpers, "metrics_rev_version": MR.METRICS_REV_VERSION,
                          "stats_py_sha256": C.sha256_file(ST.__file__), "stats_version": ST.STATS_VERSION,
                          "aggregate_json": str(agg_path), "aggregate_json_sha256": C.sha256_file(agg_path),
                          "stats_json": str(stats_path), "stats_json_sha256": C.sha256_file(stats_path),
                          "stats_run_type": stats_final.get("run_type"), "versions": MR.versions(), "env": env,
                          "plan": plan, "git_head": C.head_commit(R), "seconds": round(time.time() - t0, 1)}}
    C.write_json_atomic(OUT, out)
    tex_final = escribir_tex(out, OUT_TEX)

    print("%s experimento T9-MLPORD: %d/%d jobs usados, faltan %d, rechazados %d; datasets completos %d/9"
          % (TAG, len(filas), out["n_jobs"], len(pendientes), len(rechazados), len(completos)))
    print("  control del calculo: %d job(s) T1 reproducidos (|dif| max %.1e); stats.h1/h4 reproducen el FINAL: %s"
          % (len(control_calc), max([c["max_abs_diff"] for c in control_calc] or [0.0]), ok1 and ok4))
    print("  %-24s %-4s %10s %10s %9s   %9s %9s   %7s" % ("dataset", "sem", "ordenado", "barajado", "dif",
                                                           "loc ord", "loc bar", "AUC_t"))
    for f in filas:
        print("  %-24s %-4s %10.4f %10.4f %+9.4f   %9.4f %9.4f   %7.4f"
              % (f["dataset"], f["seed"], f["ordered"]["auroc_FUS"], f["shuffled"]["auroc_FUS"],
                 f["diff_ordered_minus_shuffled"]["auroc_FUS"], f["ordered"]["auc_loc_RAW"],
                 f["shuffled"]["auc_loc_RAW"], f["auc_t"]))
    if completos:
        e = emparejada["auroc_FUS"]
        print("  emparejada (media de semillas, n = %d): HL %+.4f, IC %.1f %% [%+.4f, %+.4f], p exacta %s, "
              "ordenado mayor en %d/%d" % (e["n"], e["hl"], 100 * (e["ci_coverage"] or 0), e["ci_lower"], e["ci_upper"],
                                          "%.4f" % e["p_exact"] if e["p_exact"] is not None else "NA", e["k_pos"], e["n"]))
        print("  mediana del MLP ordenado %.4f en %d datasets (barajado en los mismos: %.4f; C9 con 9: %.4f)"
              % (med["ordered_median"], len(completos), med["shuffled_median_same_datasets"],
                 med["shuffled_C9_median_9"]))
        for nom, fam in (("H1", H1), ("H4", H4)):
            for name, c in fam["comparisons"].items():
                if not c["involves_MLP"]:
                    continue
                n_, r_ = c["with_ordered_MLP"], c["confirmatory_with_shuffled_MLP"]
                print("  %s %-16s ordenado: n %s HL %s p %s Holm %s | barajado (FINAL): HL %+.4f p %.4f Holm %.4f"
                      % (nom, name, n_["n"], "%+.4f" % n_["hl"] if n_["hl"] is not None else "NA",
                         "%.4f" % n_["p_exact"] if n_["p_exact"] is not None else "NA",
                         "%.4f" % n_["p_holm"] if n_["p_holm"] is not None else "NA",
                         r_["hl"], r_["p_exact"], r_["p_holm"]))
    print("  rejilla: %d/%d pares de replicas arT/arF del MLP con scores.npy identico; tiempo de los 18 jobs T1 de "
          "referencia %.2f h%s"
          % (determinismo["mlp_grid_replicas_arT_arF_identical_scores"], determinismo["n_pairs"],
             tiempos["reference_T1_wall_seconds_18_jobs"] / 3600.0,
             ("; hoy x%.2f -> faltan ~%.2f h de jobs" % (tiempos["t9_over_reference_ratio_median"],
                                                         tiempos["pending_jobs_eta_seconds"] / 3600.0))
             if ratios else ""))
    for w in avisos:
        print("  AVISO: %s" % w)
    for r in rechazados:
        print("  RECHAZADO %s: %s" % (r["job_id"], "; ".join(r["motivos"])))
    if pendientes:
        print("  faltan %d jobs: %s%s" % (len(pendientes), ", ".join(p.split("__")[1] + "/" + p.split("__s")[-1][0]
                                                                   for p in pendientes[:18]), ""))
    if completos:
        print("TEX  -> %s (%s)" % (OUT_TEX, "9 datasets: listo para integrar" if tex_final else
                                   "PARCIAL, %d/9" % len(completos)))
    print("JSON -> %s (%.1f s)" % (OUT, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
