#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics_rev.py - metricas del protocolo SIN ORACULO de la revision JISAS-D-26-02300 (analysis_plan.md congelado,
tag analysis-plan-frozen). Lee scores.npy / labels.npy / config.json de cada job DONE del runner y escribe
<job_dir>/metrics.json. Nada se reentrena; nada se escribe fuera del job (salvo --out-dir / --out-root de prueba).

Especificacion: revision/analysis/analysis_plan.md §2 (protocolo), §3 (metricas), §4.3 (configuraciones C1-C11),
§7-§8 (metricas de ablaciones y T4-P/T4-E), §12.2 (no finitos, divergencia) y §14.2-§14.3 (procedencia, R-1..R-8).
Interpretaciones literales de huecos del plan: DEVIATIONS_PROPOSED (abajo) y metrics_rev.md; aparecen en metrics.json.

API (R-1):
    fit(s_prefix, fit_mask, series_start)  -> params   SIN etiquetas. s_prefix = scores[series_start : fin de la
        calibracion] (el unico tramo que ve: las ventanas causales de los filtros necesitan las filas de calibracion
        anteriores a cada fila de F_cal, duplicados incluidos, plan §2.1); fit_mask marca F_cal dentro de ese tramo.
        Ajusta mu_ref/sd_ref (MVD), centroides KMeans (KMD), mediana/escala robusta y umbrales (§2.4-§2.6).
    transform(params, s_from_start) -> series causales {RAW, KMD_raw, MVD_raw, KMD_f, MVD_f, FUS, FUS_MAX, FUS_NF}.
    evaluate(params, scores, labels, E) -> metricas de FUS, KMD_f, MVD_f y RAW en las filas E (unica funcion que
        recibe etiquetas).
Regiones: SOLO via common.check_protocol_regions (candidata primaria calib_first_10pct_post_train; with_exclusions en
T4-E; extra_eval_exclude en la referencia c = 0 de T4-E y en E_T4 de T4-P-b).

Uso (entorno fijado por el plan, RF-2: envs/postproc; nice y OMP_NUM_THREADS=4):
    R=<raiz del repositorio>
    nice -n 19 env OMP_NUM_THREADS=4 $R/envs/postproc/bin/python $R/revision/experiments/metrics_rev.py \
        --job-dir $R/results/revision/<job_id> [--force] [--out-dir DIR] [--bootstrap-ci]
    nice -n 19 env OMP_NUM_THREADS=4 $R/envs/postproc/bin/python $R/revision/experiments/metrics_rev.py --all \
        [--results-root DIR] [--out-root DIR] [--blocks T1-A-KNAD,...] [--max-jobs N] [--force]
    # O1 (cota oraculo "as-submitted", solo semilla A, codigo detectors_2025 sin cambios; necesita torch):
    nice -n 19 /usr/bin/python3 $R/revision/experiments/metrics_rev.py --o1 --job-dir <dir> [--o1-wd-replica]
    nice -n 19 /usr/bin/python3 $R/revision/experiments/metrics_rev.py --o1 --all [--blocks T1-A-KNAD]
Salida: 0 OK; 1 algun job fallo; 2 uso; 3 guarda R-7 o de entorno (no se escribe nada).
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C

REPO = C.MAIN_REPO

SCHEMA = "jisa-metrics-rev/1"
SCHEMA_O1 = "jisa-metrics-rev-o1/1"
METRICS_REV_VERSION = "1.2.0"
PLAN_TAG = "analysis-plan-frozen"
PLAN_REL = "revision/analysis/analysis_plan.md"
DECISIONS_REL = "revision/decisions.md"
LABEL_COUNTS_REL = "revision/analysis/plan_support/label_counts.json"
LABEL_EXTRAS_REL = "revision/analysis/plan_support/label_extras.json"
ENV_LOCK_REL = "revision/env/postproc.lock.txt"
ENV_PACKAGES = {"numpy": "numpy", "scipy": "scipy", "scikit-learn": "sklearn", "pandas": "pandas",
                "threadpoolctl": "threadpoolctl"}
T0_CACHE = REPO / "data" / "cache" / "t0" / "full"

PRIMARY_CANDIDATE = "calib_first_10pct_post_train"
INS_CANDIDATE = "calib_in_sample_train_tail"
FRAC_CANDIDATES = {5: "calib_first_05pct_post_train", 10: "calib_first_10pct_post_train",
                   20: "calib_first_20pct_post_train"}
W_MVD = 10_000
F_FILT = 50_001
K_KMD = 10
KMEANS_KW = dict(n_clusters=K_KMD, init="k-means++", n_init=10, max_iter=300, random_state=0)
Q_MAIN = 0.99
Q_SENS = {"S-Q95": 0.95, "S-Q995": 0.995}
MAD_K = 1.4826
IQR_K = 1.349
DEGENERATE_SCALE = 1e-9
G_EP = 10_000
G_EP_SENS = 800
B_LOC = 10_000
DAY_GUARD = 800
PS_MIN_ROWS = 10_000
REACH = (W_MVD - 1) + (F_FILT - 1)
T4P_LEVELS = (0.0, 0.01, 0.05, 0.10, 0.20)
T4P_POOL_FRAC = 0.20
T4P_GUARD = 800
T4PA_MIN_CONTAM = 0.01
T4PA_MIN_CLEAN = 1_000
L_CTX = 800
BOOT_B = 2_000
BOOT_SEED = 20260917
BOOT_MIN_EPISODES = 5
DIVERGENCE_FRAC = 0.01
F1_TRIVIAL_MARGIN = 0.02
X_MAIN = ("FUS", "KMD_f", "MVD_f", "RAW")
X_O2 = ("FUS", "KMD_f", "MVD_f")
X_T4 = ("FUS", "KMD_f", "MVD_f")
T4E_VARIANTS = {("original", None), ("mlp", None), ("lstm", True)}

DATASET_IDS = {"KNAD:Active_Wiretap": "D1", "KNAD:Fuzzing": "D2", "KNAD:OS_Scan": "D3", "KNAD:SSDP_Flood": "D4",
               "KNAD:SSL_Renegotiation": "D5", "KNAD:Video_Injection": "D6", "ACI-IOT-2023": "D7", "CIC2017": "D8",
               "CIC2018": "D9"}
ARCH_ORDER = ("conv1d", "conv2d", "lstm", "mha")
CTX_CONFIG_IDS = {("mlp_ctx", True): "C12", ("mlp_ctx", False): "C13",
                  ("mlp_deep_ctx", True): "C14", ("mlp_deep_ctx", False): "C15",
                  ("mlp_ctx_fix", True): "C16", ("mlp_ctx_fix", False): "C17",
                  ("mlp_fix", True): "C18", ("mlp_fix", False): "C19",
                  ("original_eps", True): "C20", ("original_eps", False): "C20",
                  ("mlp_ctx_fix_e5", False): "C21", ("mlp_fix_e5", False): "C22"}
CTX_CONFIG_NAMES = {"C12": "MLP-ctx-AR", "C13": "MLP-ctx-TSR",
                    "C14": "Deep-MLP-ctx-AR", "C15": "Deep-MLP-ctx-TSR",
                    "C16": "MLP-ctx-AR re-entrenado", "C17": "MLP-ctx-TSR re-entrenado",
                    "C18": "MLP-1-paso-AR re-entrenado (control, no equivale a C9)",
                    "C19": "MLP-por-registro-TSR re-entrenado (equivale a C9 salvo el protocolo)",
                    "C20": "dA de referencia con eps coherente (C10 con 1e-16 en las dos fases)",
                    "C21": "MLP-ctx-TSR re-entrenado, 5 epocas",
                    "C22": "MLP-por-registro-TSR re-entrenado, 5 epocas (control del presupuesto)"}
CIC2017_DAY_BY_PREFIX = {"Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed", "Thursday": "Thu", "Friday": "Fri"}
TS_NA = np.iinfo(np.int64).min

TAGS = {"primary": "[CONFIRMATORIO plan §3.1/§5]", "secondary": "[SECUNDARIA plan §3.2; sin afirmaciones confirmatorias]",
        "sens": "[SENSIBILIDAD plan §2.9]", "t4": "[SENSIBILIDAD plan §8]", "expl": "[EXPLORATORIO]",
        "oracle": "[ORACULO: evaluation-region oracle F1 (O2)]", "oracle1": "[ORACULO: as-submitted (O1)]",
        "dupcal": "[EXPLORATORIO] S-DUPCAL (F-01; fuera del plan: E sin gemelos exactos de F_cal)"}
CHRONO_DIR = ".metrics_rev"
CHRONO_FIRST = "first_metrics_json.json"
CHRONO_LOG = "metrics_json_log.jsonl"

DEVIATIONS_PROPOSED = [
    "MR-1 (§3.1-§3.2, §3.6) ROC-AUC, PR-AUC, EER y O2 se calculan con una replica de una sola ordenacion de "
    "sklearn 1.6.1 (_binary_clf_curve + roc_curve(drop_intermediate) + scipy.integrate.trapezoid; "
    "precision_recall_curve + AP escalonada) con las mismas operaciones: identica bit a bit a roc_auc_score / "
    "average_precision_score (test h6). En cada job el ROC-AUC primario de FUS en E se recalcula ademas con "
    "sklearn.metrics.roc_auc_score y se exige |diff| <= 1e-12; el valor citado es el de sklearn. O2 con empates en "
    "el F1 maximo: se reporta el umbral MAS BAJO entre los empatados (= primer argmax en el orden de "
    "precision_recall_curve, umbrales crecientes) con su precision y recall; f1_max no depende del desempate (RA-6).",
    "MR-2 (§2.4) KMeans se ajusta con threadpoolctl limitado a 1 thread OpenMP (el plan no fija threads; la "
    "reduccion multi-thread de sklearn no es bit a bit reproducible). Evidencia (recalculo independiente, 3 jobs AW "
    "stat/mlp/lstm arT sA, 2026-09-18): con OMP = 4 frente a 1 thread, centroides |d| <= 1,8e-15, FUS |d| <= "
    "2,2e-10, metricas de E <= 1,5e-13 y recuentos TP/FP/TN/FN identicos; con 1 thread, series identicas bit a bit. "
    "Toda replica o recomputacion independiente usa KMeans con 1 thread OpenMP; threadpoolctl queda en el lock de "
    "envs/postproc. Los centroides se guardan ORDENADOS de menor a mayor (RA-3; KMD usa la distancia minima y no "
    "depende del orden).",
    "MR-3 (§2.9 S-PS) 'estadisticos de calibracion' = TODOS los objetos ajustados (mu_ref, sd_ref, centroides, "
    "mediana/escala y umbrales) sobre F_cal ∩ {t >= T + W + F - 2}.",
    "MR-4 (§2.9 S-INS) F_cal in-sample = todas las filas de [fm + floor(ad/3), T) (son filas de entrenamiento: la "
    "deduplicacion frente al entrenamiento no aplica); E = E primaria, subconjunto de la eval de la candidata in-sample.",
    "MR-5 (§3.3) barrido k/2 = floor(k/2 + 0,5) (misma regla de redondeo que k); WindowDiff sobre subconjuntos de E "
    "(dias, E_T4, E sin duplicados, segmentos de T4-E) sobre las filas concatenadas en orden del stream.",
    "MR-6 (§3.4) episodio de falsa alarma = grupo de alarmas separadas por < G filas (filas entre alarmas "
    "consecutivas) cuyo intervalo [primera, ultima alarma] no contiene ninguna fila de ataque; en S-G800 se usa G = 800 "
    "tanto para los episodios de ataque como para fusionar alarmas.",
    "MR-7 (§12.2) -inf se trata como NaN/+inf (-> +inf); en los detectores, si aun no hay ningun valor finito en "
    "[inicio, t], se usa 0 (minimo de log10(1 + s)); en RAW, +inf -> (maximo finito + 1), que conserva los rangos. "
    "La sustitucion causal empieza en el inicio de las series de cada analisis: T en el protocolo principal y fm en "
    "S-INS (F-06; antes, los no finitos de [fm, T) se ponian a 0). tau_RAW se ajusta sobre la serie causal (el maximo "
    "global incluiria E) y RAW se evalua con +inf por encima de todo finito (§12.2); en E ambas sustituciones son "
    "equivalentes para rangos y para la regla X > tau.",
    "MR-8 (§3.6 O1) O1 se ejecuta con /usr/bin/python3 (torch; el entorno de la reproduccion 56/56) y no con "
    "envs/postproc (sin torch). Opcion --o1-wd-replica: WindowDiff/Pk uint8 con la replica numpy validada por R-2 "
    "(evita ~20 GB de RAM por llamada); queda registrado en metrics_o1.json.",
    "MR-9 (§3.7) IC bootstrap opcional (--bootstrap-ci): los episodios y segmentos benignos remuestreados se "
    "representan por su multiplicidad (ROC-AUC ponderado = ROC-AUC de la concatenacion; el orden no afecta al AUC); "
    "rng = numpy.random.default_rng(20260917), por replica primero los episodios y despues los segmentos.",
    "MR-10 (§3.4, §2.9) S-LOC, episodios y operativas sobre E segmentada (T4-E) usan las filas de E concatenadas.",
    "MR-11 (F-01; fuera del plan, [EXPLORATORIO]) S-DUPCAL: metricas en E sin las filas cuyo vector de entrada del "
    "modelo (x.npy de la cache) es identico al de alguna fila de F_cal (hash warmup_audit.hash_matrix + comparacion "
    "exacta de valores, -0 == +0, NaN == NaN); se reportan tambien las filas de F_cal con gemelo en E. Nunca se quitan "
    "filas de F_cal por tener gemelo en E (seria mirar hacia delante). El primario no cambia.",
    "MR-12 (F-10, §14.4 hito (i)) la hora del primer metrics.json se registra una sola vez (O_EXCL) en <raiz de "
    "salida>/.metrics_rev/first_metrics_json.json y cada escritura se anade a metrics_json_log.jsonl (append-only); "
    "provenance.first_metrics_json_created = min(registro, mtimes actuales, esta escritura). Cada entrada lleva "
    "family y exploratory_arm: aggregate descarta las de la familia T7 al reconstruir la cronologia R-7, para que un "
    "job exploratorio no alimente una guarda confirmatoria.",
    "MR-13 (fuera del plan congelado) la familia T7 (bloques T7-MLPW y T7-MLPDEEP, variantes 'mlp_ctx' y "
    "'mlp_deep_ctx') es aplicable: se calculan las MISMAS metricas del protocolo (mismas regiones, misma "
    "calibracion, mismos detectores) para que sean comparables con la rejilla, pero el job entero es [EXPLORATORIO]: "
    "primary.tag y E.tag son [EXPLORATORIO], primary.confirmatory = false, el metrics.json lleva 'exploratory_job' y "
    "aggregate.py lo deja fuera de S*_d y de los analisis confirmatorios (AG-15). config_id C12-C15, uno por "
    "(variante, paradigma): al contrario que C9 (multilayer.py:16-17 ignora 'ar'), estas variantes SI leen la "
    "ventana, asi que no se promedian sin comprobarlo. MEDIDO: en 'mlp_ctx' con L = 800 los dos paradigmas son casi "
    "el MISMO estimador. Sintetico (1 AE, nv = 6, misma semilla, solo cambia L): max|AR-TSR| = 4,5 % del rango a L = "
    "8, 3,2 % a L = 64, 0,12 % a L = 256 y 0,21 % a L = 800 (Pearson 0,999994). Datos reales (run_job --test-mode, "
    "KNAD:OS_Scan, corrected, L = 800, 4 AEs, score fusionado, 6000 filas): max|AR-TSR| = 16,5 % del rango pero "
    "Pearson 0,99996 y correlacion de RANGOS 0,99973. Como AUROC es un estadistico de rangos, C12 y C13 daran cifras "
    "practicamente iguales. El mecanismo es arquitectonico: la unica diferencia es una fila de 800 dentro de un "
    "Flatten de L*n_visible que entra en un cuello de botella de ceil(0,22*n_visible) unidades, y ninguna capa "
    "pondera la ultima fila (a diferencia del kernel L//4 de conv1d, de la ultima celda del LSTM o de la atencion de "
    "MHA). Se mantienen los cuatro config_id para NO promediar dos ejecuciones sin haber demostrado que son "
    "replicas, pero la justificacion 'AR no es replica de TSR' NO esta sostenida por la medida en 'mlp_ctx': C12 vs "
    "C13 no puede sostener ninguna afirmacion del manuscrito.",
    "MR-14 el metrics.json tiene TRES claves con 'exploratory' en el nombre y distinto alcance: 'exploratory' de "
    "primer nivel = el bloque de analisis EXPLORATORIOS de §11 de un job normal (existe tambien en los "
    "confirmatorios); 'job.exploratory_arm' = bandera booleana (este job es de un bloque fuera del plan); "
    "'exploratory_job' de primer nivel = el detalle de ese bloque. Un consumidor que quiera saber si el JOB es "
    "confirmatorio debe leer 'job.exploratory_arm' (o primary.confirmatory), NUNCA 'exploratory' de primer nivel.",
]


class PlanGuardError(RuntimeError):
    """R-7: plan congelado ausente o distinto (no se calcula ninguna metrica)."""


class EnvGuardError(RuntimeError):
    """RF-2: entorno de analisis distinto del fijado por el plan."""


class NotApplicable(RuntimeError):
    """Job sin metricas del protocolo (X2/orig, sondas S, sin scores)."""


def _git_bytes(root, *args):
    p = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args], capture_output=True, timeout=60)
    return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")


def plan_guard(repo=REPO, plan_path=None, decisions_path=None, tag=PLAN_TAG):
    """Guard R-7 (plan §14.3): abort unless the tag exists and the sha256 of the plan agrees everywhere; returns the provenance."""
    repo = Path(repo)
    plan_path = Path(plan_path) if plan_path else repo / PLAN_REL
    decisions_path = Path(decisions_path) if decisions_path else repo / DECISIONS_REL
    commit = C.tag_commit(repo, tag)
    if not commit:
        raise PlanGuardError(f"R-7: no existe el tag {tag} en {repo}")
    rc, blob, err = _git_bytes(repo, "show", f"{tag}:{PLAN_REL}")
    if rc != 0:
        raise PlanGuardError(f"R-7: git show {tag}:{PLAN_REL} fallo: {err[:300]}")
    sha_tag = hashlib.sha256(blob).hexdigest()
    if not plan_path.exists():
        raise PlanGuardError(f"R-7: falta {plan_path}")
    sha_wt = C.sha256_file(plan_path)
    if sha_wt != sha_tag:
        raise PlanGuardError(f"R-7: sha256 del plan en la copia de trabajo {sha_wt} != del tag {sha_tag}")
    txt = decisions_path.read_text(encoding="utf-8") if decisions_path.exists() else ""
    reg = None
    for ln in txt.splitlines():
        if tag in ln:
            m = re.search(r"sha256 de `analysis_plan\.md` = `([0-9a-f]{64})`", ln)
            if m:
                reg = m.group(1)
                mc = re.search(r"commit `([0-9a-f]{40})`", ln)
                if mc and mc.group(1) != commit:
                    raise PlanGuardError(f"R-7: el registro tiene el commit {mc.group(1)} != tag {commit}")
    if reg is None:
        raise PlanGuardError(f"R-7: {decisions_path} no registra el sha256 del plan congelado")
    if reg != sha_tag:
        raise PlanGuardError(f"R-7: sha256 registrado {reg} != del tag {sha_tag}")
    rc, tobj, _ = C.git(repo, "rev-parse", f"refs/tags/{tag}")
    rc2, ctime, _ = C.git(repo, "log", "-1", "--format=%cI", commit)
    return {"tag": tag, "plan_commit": commit, "tag_object": tobj if rc == 0 else None,
            "tag_commit_time": ctime if rc2 == 0 else None, "plan_sha256": sha_tag, "plan_sha256_worktree": sha_wt,
            "plan_sha256_decisions_md": reg, "check": "R-7 OK (tag, copia de trabajo y registro coinciden)"}


def expected_env(repo=REPO):
    out = {}
    p = Path(repo) / ENV_LOCK_REL
    for ln in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", ln.strip())
        if m and m.group(1).lower() in ENV_PACKAGES:
            out[m.group(1).lower()] = m.group(2)
    return out


def versions():
    import scipy
    import sklearn
    import pandas
    v = {"python": sys.version.split()[0], "executable": sys.executable, "numpy": np.__version__,
         "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pandas": pandas.__version__}
    try:
        import threadpoolctl
        v["threadpoolctl"] = threadpoolctl.__version__
    except ImportError:
        v["threadpoolctl"] = None
    return v


def env_guard(repo=REPO):
    """RF-2 (plan §5.1): metrics_rev corre en envs/postproc con las versiones del lock."""
    exp = expected_env(repo)
    v = versions()
    got = {"numpy": v["numpy"], "scipy": v["scipy"], "scikit-learn": v["sklearn"], "pandas": v["pandas"],
           "threadpoolctl": v["threadpoolctl"]}
    bad = {k: {"lock": exp[k], "instalada": got.get(k)} for k in exp if got.get(k) != exp[k]}
    if bad:
        raise EnvGuardError(f"RF-2: versiones distintas del lock {ENV_LOCK_REL}: {bad} (usa envs/postproc/bin/python)")
    return {"lock": ENV_LOCK_REL, "expected": exp, "matches_lock": True}


def helper_hashes():
    out = {"metrics_rev.py": C.sha256_file(Path(__file__).resolve()),
           "common.py": C.sha256_file(HERE / "common.py"), "pipelines.py": C.sha256_file(HERE / "pipelines.py")}
    return out


def version_key(options):
    h = helper_hashes()
    blob = json.dumps({"h": h, "o": options, "v": METRICS_REV_VERSION, "schema": SCHEMA}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _rolling(arr, w):
    import pandas as pd
    return pd.Series(np.asarray(arr, dtype=np.float64), copy=False).rolling(window=w, min_periods=1)


def nearest_centroid_dist(s, centroids):
    """min_j |s_t - c_j| (1-D: el minimo se alcanza en un vecino en el orden de los centroides)."""
    c = np.sort(np.asarray(centroids, dtype=np.float64))
    i = np.searchsorted(c, s)
    lo = np.clip(i - 1, 0, len(c) - 1)
    hi = np.clip(i, 0, len(c) - 1)
    return np.minimum(np.abs(s - c[lo]), np.abs(s - c[hi]))


def detector_series(s_seg, mu_ref, sd_ref, centroids):
    """Series causales desde el inicio de s_seg (T; fm en S-INS). pandas.rolling(min_periods=1) (plan §2.4)."""
    s_seg = np.asarray(s_seg, dtype=np.float64)
    r = _rolling(s_seg, W_MVD)
    mu_w = r.mean().to_numpy()
    sd_w = r.std(ddof=0).to_numpy()
    md = (mu_w - mu_ref) ** 2
    sdv = (sd_w - sd_ref) ** 2
    mvd_raw = 2.0 * md * sdv / (md + sdv + 1e-9)
    del mu_w, sd_w, md, sdv
    kmd_raw = nearest_centroid_dist(s_seg, centroids)
    kmd_f = _rolling(kmd_raw, F_FILT).mean().to_numpy()
    mvd_f = _rolling(mvd_raw, F_FILT).median().to_numpy()
    out = {"KMD_raw": kmd_raw, "MVD_raw": mvd_raw, "KMD_f": kmd_f, "MVD_f": mvd_f}
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise FloatingPointError(f"serie {k} con valores no finitos ({int((~np.isfinite(v)).sum())})")
    return out


def fit_kmeans(s_fit):
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    s_fit = np.asarray(s_fit, dtype=np.float64)
    n_distinct = int(np.unique(s_fit).size)
    k = min(K_KMD, n_distinct)
    kw = dict(KMEANS_KW, n_clusters=k)
    with threadpool_limits(limits=1, user_api="openmp"):
        km = KMeans(**kw).fit(s_fit.reshape(-1, 1))
    cents = np.sort(km.cluster_centers_.ravel().astype(np.float64))
    return cents, {"K": int(k), "k_reduced": bool(k < K_KMD), "centroids_order": "ascending (RA-3)",
                                                            "n_distinct_fit": n_distinct, "n_iter": int(km.n_iter_),
                                                            "inertia": float(km.inertia_), "kwargs": kw,
                                                            "openmp_threads": 1}


def robust_scale(v):
    v = np.asarray(v, dtype=np.float64)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad > 0:
        return {"median": med, "mad": mad, "scale": MAD_K * mad, "scale_source": "1.4826*MAD", "degenerate_scale": False}
    iqr = float(np.quantile(v, 0.75, method="linear") - np.quantile(v, 0.25, method="linear"))
    if iqr > 0:
        return {"median": med, "mad": mad, "iqr": iqr, "scale": iqr / IQR_K, "scale_source": "IQR/1.349",
                "degenerate_scale": False}
    return {"median": med, "mad": mad, "iqr": iqr, "scale": DEGENERATE_SCALE, "scale_source": "1e-9",
            "degenerate_scale": True}


def _z(d, rs):
    return (d - rs["median"]) / rs["scale"]


def _fused(det, robust):
    zk, zm = _z(det["KMD_f"], robust["KMD_f"]), _z(det["MVD_f"], robust["MVD_f"])
    zk0, zm0 = _z(det["KMD_raw"], robust["KMD_raw"]), _z(det["MVD_raw"], robust["MVD_raw"])
    return {"FUS": (zk + zm) / 2.0, "FUS_MAX": np.maximum(zk, zm), "FUS_NF": (zk0 + zm0) / 2.0}


THRESHOLD_SERIES = ("FUS", "KMD_f", "MVD_f", "RAW", "FUS_MAX", "FUS_NF", "KMD_raw", "MVD_raw")


def _quantile(v, q):
    return float(np.quantile(np.asarray(v, dtype=np.float64), q, method="linear"))


def fit(s_prefix, fit_mask, series_start):
    """Ajuste SIN etiquetas (R-1). s_prefix = scores[series_start : fin de la calibracion] ya saneados (no finitos,
    §12.2); fit_mask (bool, misma longitud) = F_cal. Devuelve params (solo numeros, serializable)."""
    s_prefix = np.asarray(s_prefix, dtype=np.float64)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if s_prefix.shape != fit_mask.shape:
        raise ValueError(f"fit: s_prefix {s_prefix.shape} y fit_mask {fit_mask.shape} no casan")
    n_fit = int(fit_mask.sum())
    if n_fit == 0:
        raise ValueError("fit: F_cal vacio")
    s_fit = s_prefix[fit_mask]
    mu_ref, sd_ref = float(np.mean(s_fit)), float(np.std(s_fit))
    centroids, kinfo = fit_kmeans(s_fit)
    det = detector_series(s_prefix, mu_ref, sd_ref, centroids)
    robust = {k: robust_scale(det[k][fit_mask]) for k in ("KMD_f", "MVD_f", "KMD_raw", "MVD_raw")}
    ser = dict(det, RAW=s_prefix, **_fused(det, robust))
    thr = {str(q): {x: _quantile(ser[x][fit_mask], q) for x in THRESHOLD_SERIES} for q in (0.95, 0.99, 0.995)}
    return {"series_start": int(series_start), "n_prefix": int(len(s_prefix)), "n_fit": n_fit,
            "mvd": {"W": W_MVD, "mu_ref": mu_ref, "sd_ref": sd_ref, "ddof": 0},
            "kmd": dict(kinfo, centroids=[float(c) for c in centroids]),
            "filters": {"F": F_FILT, "KMD": "media movil causal", "MVD": "mediana movil causal",
                        "impl": "pandas.Series.rolling(window, min_periods=1)"},
            "fusion": {"FUS": "(z_KMD_f + z_MVD_f)/2", "robust": robust,
                       "degenerate_scale": any(r["degenerate_scale"] for r in robust.values())},
            "thresholds": thr, "threshold_rule": "numpy.quantile(X[F_cal], q, method='linear'); alarma X > tau"}


def transform(params, s_from_start):
    """Series causales desde params['series_start'] con los objetos ajustados (sin etiquetas)."""
    s = np.asarray(s_from_start, dtype=np.float64)
    det = detector_series(s, params["mvd"]["mu_ref"], params["mvd"]["sd_ref"], params["kmd"]["centroids"])
    fused = _fused(det, params["fusion"]["robust"])
    return dict(det, RAW=s, **fused)


def evaluate(params, scores, labels, E, which=X_MAIN, q=Q_MAIN, raw_scores=None):
    """R-1: metricas de las series transformadas en las filas E (indices del stream o lista de segmentos [a, b)).
    Unica funcion de la API que recibe etiquetas. raw_scores (opcional) = scores para RAW (§12.2)."""
    st = params["series_start"]
    ser = transform(params, np.asarray(scores)[st:])
    if raw_scores is not None:
        ser["RAW"] = np.asarray(raw_scores, dtype=np.float64)[st:]
    rows = rows_of(E)
    y = np.asarray(labels)[rows].astype(bool)
    out = {"trivial": trivial_refs(y)}
    for x in which:
        xe = ser[x][rows - st]
        tau = params["thresholds"][str(q)][x]
        out[x] = dict(curve_metrics(xe, y), **threshold_metrics(xe, y, tau, out["trivial"]))
    return out


def clf_curve(x, y):
    """Replica de sklearn.metrics._ranking._binary_clf_curve (sin pesos): (fps, tps, umbrales) descendentes."""
    order = np.argsort(x, kind="mergesort")[::-1]
    xs = x[order]
    ys = y[order]
    distinct = np.where(np.diff(xs))[0]
    tidx = np.r_[distinct, ys.size - 1]
    tps = np.cumsum(ys, dtype=np.float64)[tidx]
    fps = 1 + tidx - tps
    return fps, tps, xs[tidx]


def _auroc_from_curve(fps, tps):
    from scipy.integrate import trapezoid
    if fps[-1] <= 0 or tps[-1] <= 0:
        return None
    if len(fps) > 2:
        opt = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps, tps = fps[opt], tps[opt]
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    return float(trapezoid(tps / tps[-1], fps / fps[-1]))


def _pr_from_curve(fps, tps):
    ps = tps + fps
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))
    recall = tps / tps[-1]
    return precision, recall


def _ap_from_pr(precision, recall):
    p = np.hstack((precision[::-1], 1))
    r = np.hstack((recall[::-1], 0))
    return float(max(0.0, -np.sum(np.diff(r) * np.array(p)[:-1])))


def _eer_from_curve(fps, tps):
    """EER sobre roc_curve(drop_intermediate=False) (§3.2): primer par consecutivo en que FNR - FPR cambia de signo,
    interpolacion lineal; EER = valor comun interpolado."""
    tp0, fp0 = np.r_[0, tps], np.r_[0, fps]
    fpr, tpr = fp0 / fp0[-1], tp0 / tp0[-1]
    d = (1.0 - tpr) - fpr
    j = int(np.argmax(d <= 0))
    if d[j] > 0:
        return None
    if j == 0:
        return float(fpr[0])
    i = j - 1
    if d[j] == 0:
        return float(fpr[j])
    a = d[i] / (d[i] - d[j])
    return float(fpr[i] + a * (fpr[j] - fpr[i]))


def curve_metrics(x, y, o2=False):
    """ROC-AUC, PR-AUC (AP), EER y, si o2, F1 maximo sobre todos los umbrales distintos (O2) de una ordenacion."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return {"auroc": None, "auprc": None, "eer": None, "undefined": "una sola clase"}
    fps, tps, thr = clf_curve(x, y)
    out = {"auroc": _auroc_from_curve(fps, tps), "eer": _eer_from_curve(fps, tps)}
    prec, rec = _pr_from_curve(fps, tps)
    out["auprc"] = _ap_from_pr(prec, rec)
    if o2:
        den = prec + rec
        f1 = np.zeros_like(den)
        np.divide(2 * prec * rec, den, out=f1, where=den > 0)
        i = int(len(f1) - 1 - np.argmax(f1[::-1]))
        out["o2"] = {"f1_max": float(f1[i]), "threshold": float(thr[i]), "precision": float(prec[i]),
                     "recall": float(rec[i]), "decision": "X >= threshold (precision_recall_curve)",
                     "tie_rule": "umbral mas bajo entre los que empatan en F1 maximo (MR-1)",
                     "n_thresholds": int(len(thr))}
    return out


def auroc(x, y):
    return curve_metrics(x, y)["auroc"]


def trivial_refs(y, rows=None):
    y = np.asarray(y, dtype=bool)
    n = int(y.size)
    n1 = int(y.sum())
    p = n1 / n if n else None
    out = {"n": n, "n_attack": n1, "n_benign": n - n1, "prevalence": p,
           "f1_trivial": (2 * p / (1 + p)) if p is not None else None, "ap_trivial": p, "auroc_trivial": 0.5,
           "auc_t": auc_index(y)}
    return out


def auc_index(y):
    """AUC_t: ROC-AUC con el indice de fila como score (estrictamente creciente, sin empates; §3.2). Formula de
    revision/analysis/plan_label_extras.py:auc_index."""
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def threshold_metrics(x, y, tau, triv=None):
    """Decision X > tau (§2.6); F1/precision con zero_division=0 (sklearn: F1 = 2tp/(2tp + fp + fn))."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    a = x > tau
    tp = int(np.count_nonzero(a & y))
    fp = int(np.count_nonzero(a & ~y))
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    fn, tn = n1 - tp, n0 - fp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / n1 if n1 > 0 else None
    den = 2 * tp + fp + fn
    f1 = (2 * tp / den) if den > 0 else 0.0
    out = {"threshold": float(tau), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec,
           "f1": f1, "fpr": (fp / n0) if n0 > 0 else None, "alarm_rate": (tp + fp) / y.size if y.size else None,
           "n_alarms": tp + fp}
    if triv is not None and triv.get("f1_trivial") is not None:
        f1t = triv["f1_trivial"]
        out["f1n"] = (f1 - f1t) / (1 - f1t) if f1t < 1 else None
        out["f1_le_trivial_plus_0.02"] = bool(f1 <= f1t + F1_TRIVIAL_MARGIN)
    return out


def _add_apn(d, triv):
    p = triv.get("prevalence")
    if d.get("auprc") is not None and p is not None and p < 1:
        d["apn"] = (d["auprc"] - p) / (1 - p)
    return d


def wd_k(ref):
    """k = floor(n / (2 * n_segmentos) + 0,5) (Beeferman et al.; §3.3)."""
    ref = np.asarray(ref)
    n = len(ref)
    nseg = int(np.count_nonzero(ref[1:] != ref[:-1])) + 1 if n else 0
    return (int(math.floor(n / (2 * nseg) + 0.5)) if nseg else None), nseg


def windowdiff(ref, hyp, k):
    """WD = (1/(n-k)) sum_{i=0}^{n-k-1} 1[b_ref(i, i+k) != b_hyp(i, i+k)], b = n.o de fronteras en (i, i+k];
    frontera entre t-1 y t si el valor cambia. Sumas acumuladas int64 (§3.3). None si k >= n."""
    ref = np.asarray(ref)
    hyp = np.asarray(hyp)
    n = len(ref)
    if k is None or k < 1 or k >= n:
        return None
    br = np.zeros(n, np.int64)
    bh = np.zeros(n, np.int64)
    br[1:] = ref[1:] != ref[:-1]
    bh[1:] = hyp[1:] != hyp[:-1]
    cr = np.concatenate([[0], np.cumsum(br, dtype=np.int64)])
    ch = np.concatenate([[0], np.cumsum(bh, dtype=np.int64)])
    wr = cr[k + 1:n + 1] - cr[1:n - k + 1]
    wh = ch[k + 1:n + 1] - ch[1:n - k + 1]
    return int(np.count_nonzero(wr != wh)) / (n - k)


def windowdiff_bruteforce(ref, hyp, k):
    n = len(ref)
    if k is None or k < 1 or k >= n:
        return None
    bad = 0
    for i in range(n - k):
        br = sum(1 for t in range(i + 1, i + k + 1) if ref[t] != ref[t - 1])
        bh = sum(1 for t in range(i + 1, i + k + 1) if hyp[t] != hyp[t - 1])
        bad += br != bh
    return bad / (n - k)


def run_stats(y):
    s, e, v = runs(y)
    ln = e - s
    a, b = ln[v == 1], ln[v == 0]
    return {"n_segments": int(len(s)), "attack_runs": int(len(a)), "mean_attack_run": float(a.mean()) if len(a) else None,
            "benign_runs": int(len(b)), "mean_benign_run": float(b.mean()) if len(b) else None}


def wd_block(ref, hyp, k=None):
    ref = np.asarray(ref, dtype=np.int8)
    hyp = np.asarray(hyp, dtype=np.int8)
    n = len(ref)
    k_rule, nseg = wd_k(ref)
    k = k_rule if k is None else k
    out = {"wd": windowdiff(ref, hyp, k), "k": k, "k_rule": "floor(n/(2*n_segmentos)+0.5)", "n": n,
           "n_windows": (n - k) if (k is not None and k < n) else 0}
    out.update(run_stats(ref))
    sweep = {}
    for name, kk in (("k_half", int(math.floor(k / 2 + 0.5)) if k else None), ("k_double", 2 * k if k else None)):
        ok = kk is not None and kk >= 1 and kk < n / 2
        sweep[name] = {"k": kk, "wd": windowdiff(ref, hyp, kk) if ok else None,
                       "defined": bool(ok), "rule": "NA si k >= n/2 (§3.3)"}
    out["sweep"] = sweep
    return out


def wd_pk_uint8(ref, hyp, k=None):
    """Replica de windowdiff_pk_score (torch, uint8): sumas por ventana MOD 256; n - k + 1 ventanas; media en float32.
    k por defecto = max(min(N//10, 2e10//N), 3) (metrics.py:11,51). Devuelve (wd, pk, info)."""
    ref = np.asarray(ref).astype(np.int64)
    hyp = np.asarray(hyp).astype(np.int64)
    n = len(ref)
    if k is None:
        k = max(min(n // 10, 20_000_000_000 // n), 3)
    cr = np.concatenate([[0], np.cumsum(ref)])
    ch = np.concatenate([[0], np.cumsum(hyp)])
    nw = n - k + 1
    sr = (cr[k:] - cr[:nw]) % 256
    sh = (ch[k:] - ch[:nw]) % 256
    wd_cnt = int(np.count_nonzero(sr != sh))
    mm = (ref != hyp).astype(np.int64)
    cm = np.concatenate([[0], np.cumsum(mm)])
    pk_cnt = int(np.count_nonzero(((cm[k:] - cm[:nw]) % 256) != 0))
    f32 = np.float32
    wd = float(f32(wd_cnt) / f32(nw))
    pk = float(f32(pk_cnt) / f32(nw))
    return wd, pk, {"k": int(k), "n_windows": int(nw), "wd_count": wd_cnt, "pk_count": pk_cnt,
                    "wd_exact": wd_cnt / nw, "pk_exact": pk_cnt / nw}


def runs(y):
    y = np.asarray(y, np.int8)
    if len(y) == 0:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.int8)
    ch = np.flatnonzero(np.diff(y) != 0) + 1
    starts = np.concatenate([[0], ch]).astype(np.int64)
    ends = np.concatenate([ch, [len(y)]]).astype(np.int64)
    return starts, ends, y[starts]


def episodes(y, gap):
    """Runs de ataque fusionados cuando los separan menos de `gap` filas benignas (plan_label_counts.episodes).
    Devuelve (inicios, finales exclusivos) en posiciones de y."""
    s, e, v = runs(y)
    s, e = s[v == 1], e[v == 1]
    if len(s) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    new = np.r_[True, (s[1:] - e[:-1]) >= gap]
    first = np.flatnonzero(new)
    last = np.r_[first[1:] - 1, len(s) - 1]
    return s[first].astype(np.int64), e[last].astype(np.int64)


def alarm_clusters(alarm, gap):
    """Grupos de alarmas: alarmas consecutivas separadas por < gap filas (filas entre ellas) se fusionan. Devuelve
    (primera, ultima) inclusivas."""
    p = np.flatnonzero(alarm)
    if len(p) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    new = np.r_[True, (p[1:] - p[:-1] - 1) >= gap]
    first = np.flatnonzero(new)
    last = np.r_[first[1:] - 1, len(p) - 1]
    return p[first].astype(np.int64), p[last].astype(np.int64)


def operational(y, alarm, gap, ts=None, per_episode=True):
    """Deteccion por episodio (>= 1 alarma en [s, e]), retardo (filas y segundos) y episodios de falsa alarma (MR-6)."""
    y = np.asarray(y, dtype=np.int8)
    alarm = np.asarray(alarm, dtype=bool)
    n = len(y)
    es, ee = episodes(y, gap)
    ap = np.flatnonzero(alarm)
    det, dly, dls, eps = [], [], [], []
    for s, e in zip(es, ee):
        i = np.searchsorted(ap, s)
        hit = i < len(ap) and ap[i] < e
        first = int(ap[i]) if hit else None
        d_rows = (first - int(s)) if hit else None
        d_s = None
        if hit and ts is not None and np.isfinite(ts[first]) and np.isfinite(ts[s]):
            d_s = float(ts[first] - ts[s])
        det.append(hit)
        if hit:
            dly.append(d_rows)
            if d_s is not None:
                dls.append(d_s)
        if per_episode:
            eps.append({"start": int(s), "end_excl": int(e), "n_rows": int(e - s), "n_attack": int(y[s:e].sum()),
                        "detected": bool(hit), "first_alarm": first, "delay_rows": d_rows, "delay_s": d_s})
    cy = np.concatenate([[0], np.cumsum(y, dtype=np.int64)])
    cs, ce = alarm_clusters(alarm, gap)
    false_cl = (cy[ce + 1] - cy[cs]) == 0 if len(cs) else np.array([], bool)
    n_fa = int(false_cl.sum())
    n0 = int(n - y.sum())
    dur_h = None
    if ts is not None:
        ok = np.isfinite(ts)
        if ok.any():
            dur_h = float((np.nanmax(ts[ok]) - np.nanmin(ts[ok])) / 3600.0)
    out = {"gap": gap, "n_episodes": int(len(es)), "n_detected": int(sum(det)),
           "detection_rate": (sum(det) / len(es)) if len(es) else None,
           "median_delay_rows": float(np.median(dly)) if dly else None,
           "median_delay_s": float(np.median(dls)) if dls else None,
           "n_alarm_clusters": int(len(cs)), "n_false_alarm_episodes": n_fa,
           "false_alarm_episodes_per_1e6_benign": (n_fa / n0 * 1e6) if n0 else None,
           "duration_hours": dur_h, "false_alarm_episodes_per_hour": (n_fa / dur_h) if dur_h else None,
           "rule_false_alarm": "MR-6: alarmas con < gap filas entre consecutivas; intervalo sin filas de ataque"}
    if per_episode:
        out["episodes"] = eps
    return out


def block_edges(n, b=B_LOC):
    nb = max(1, n // b)
    return [i * b for i in range(nb)] + [n]


def auc_loc(x, y, b=B_LOC):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    ed = block_edges(len(y), b)
    num = den = 0.0
    both = a_cov = b_cov = 0
    for s, e in zip(ed[:-1], ed[1:]):
        yy = y[s:e]
        n1 = int(yy.sum())
        n0 = (e - s) - n1
        if n1 > 0 and n0 > 0:
            w = float(n1) * float(n0)
            num += w * auroc(x[s:e], yy)
            den += w
            both += 1
            a_cov += n1
            b_cov += n0
    n1t = int(y.sum())
    n0t = len(y) - n1t
    return {"auc_loc": (num / den) if den > 0 else None, "blocks": len(ed) - 1, "blocks_both_classes": both,
            "attack_rows_covered": (a_cov / n1t) if n1t else None, "benign_rows_covered": (b_cov / n0t) if n0t else None,
            "b": b, "rule": "bloques de b filas desde el inicio de E; el ultimo parcial se une al anterior"}


def loc_coverage(y, b=B_LOC):
    y = np.asarray(y, dtype=bool)
    ed = block_edges(len(y), b)
    both = a_cov = b_cov = 0
    for s, e in zip(ed[:-1], ed[1:]):
        n1 = int(y[s:e].sum())
        n0 = (e - s) - n1
        if n1 > 0 and n0 > 0:
            both += 1
            a_cov += n1
            b_cov += n0
    n1t = int(y.sum())
    n0t = len(y) - n1t
    return {"blocks": len(ed) - 1, "blocks_both_classes": both, "attack_rows_covered": (a_cov / n1t) if n1t else None,
            "benign_rows_covered": (b_cov / n0t) if n0t else None}


def bootstrap_auc_ci(x, y, gap=G_EP, B=BOOT_B, seed=BOOT_SEED, return_values=False):
    from scipy.integrate import trapezoid
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    n = len(y)
    es, ee = episodes(y.astype(np.int8), gap)
    if len(es) < BOOT_MIN_EPISODES:
        return {"computed": False, "reason": f"{len(es)} episodios < {BOOT_MIN_EPISODES} (§3.7)"}
    bid = np.full(n, -1, np.int64)
    for i, (s, e) in enumerate(zip(es, ee)):
        bid[s:e] = i
    out_ep = bid < 0
    seg_s, seg_e, seg_v = runs(out_ep.astype(np.int8))
    segs = [(s, e) for s, e, v in zip(seg_s, seg_e, seg_v) if v == 1]
    for j, (s, e) in enumerate(segs):
        bid[s:e] = len(es) + j
    nep, nseg = len(es), len(segs)
    order = np.argsort(x, kind="mergesort")[::-1]
    xs = x[order]
    ys = y[order].astype(np.float64)
    bs = bid[order]
    tidx = np.r_[np.where(np.diff(xs))[0], n - 1]
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        me = np.bincount(rng.integers(0, nep, nep), minlength=nep) if nep else np.zeros(0, np.int64)
        ms = np.bincount(rng.integers(0, nseg, nseg), minlength=nseg) if nseg else np.zeros(0, np.int64)
        w = np.concatenate([me, ms]).astype(np.float64)[bs]
        tps = np.cumsum(ys * w)[tidx]
        fps = np.cumsum((1.0 - ys) * w)[tidx]
        if tps[-1] <= 0 or fps[-1] <= 0:
            vals[b] = np.nan
            continue
        tp0, fp0 = np.r_[0.0, tps], np.r_[0.0, fps]
        vals[b] = trapezoid(tp0 / tp0[-1], fp0 / fp0[-1])
    ok = np.isfinite(vals)
    out = {"computed": True, "B": B, "seed": seed, "n_episodes": nep, "n_benign_segments": nseg,
           "ci95_percentile": [float(np.percentile(vals[ok], 2.5)), float(np.percentile(vals[ok], 97.5))],
           "n_valid": int(ok.sum()), "rule": "MR-9; estratificado episodios/segmentos benignos; IC percentil 95 %",
           "tag": TAGS["expl"]}
    if return_values:
        out["values"] = vals
        out["blocks"] = {"episodes": list(zip(es.tolist(), ee.tolist())), "segments": [list(map(int, t)) for t in segs]}
    return out


def rows_of(E):
    """Filas (int64, ordenadas) de un intervalo [a, b), de una lista de segmentos o de un array."""
    if isinstance(E, np.ndarray):
        return E.astype(np.int64)
    if len(E) == 2 and all(isinstance(v, (int, np.integer)) for v in E):
        return np.arange(int(E[0]), int(E[1]), dtype=np.int64)
    parts = [np.arange(int(a), int(b), dtype=np.int64) for a, b in E]
    return np.concatenate(parts) if parts else np.array([], np.int64)


def segments_of(E):
    if len(E) == 2 and all(isinstance(v, (int, np.integer)) for v in E):
        return [[int(E[0]), int(E[1])]]
    return [[int(a), int(b)] for a, b in E]


def segments_subset(A, B):
    """Cada segmento [a, b) de A esta contenido en algun segmento de B (segmentos disjuntos y ordenados)."""
    return all(any(c <= a and b <= d for c, d in B) for a, b in A if b > a)


def get_regions(regions, candidate, extra_eval_exclude=None):
    """(calib, E, info) via common.check_protocol_regions; with_exclusions solo si la candidata declara exclusiones."""
    excl = C.eval_exclusions(regions, candidate)
    if extra_eval_exclude is not None:
        calib, ev = C.check_protocol_regions(regions, candidate, extra_eval_exclude=extra_eval_exclude)
        mode = "extra_eval_exclude"
    elif excl:
        calib, ev = C.check_protocol_regions(regions, candidate, with_exclusions=True)
        mode = "with_exclusions"
    else:
        calib, ev = C.check_protocol_regions(regions, candidate)
        mode = "plain"
    return calib, ev, {"candidate": candidate, "check": "common.check_protocol_regions", "mode": mode,
                       "calib": calib, "eval": ev, "eval_exclusions": excl,
                       "extra_eval_exclude": extra_eval_exclude}


_KNAD_DUP_MEMO = {}


def fit_set_mask(cfg, N, T, log=print):
    """Mascara de duplicados exactos de entrenamiento (plan §2.1; R-3). CIC/ACI: aux_dup_train de la cache con
    sha256 verificado. KNAD (v1 sin fit_set_exclusion, o v2 con mask=None): pipelines.train_duplicate_mask sobre x de
    la cache con warmup_audit.hash_matrix (fail-closed si no se puede)."""
    reg = cfg.get("regions") or {}
    fse = reg.get("fit_set_exclusion") or {}
    cache = cfg.get("cache") or {}
    if fse.get("path"):
        p = Path(fse["path"])
        sha = C.sha256_file(p)
        if fse.get("sha256") and sha != fse["sha256"]:
            raise RuntimeError(f"fit_set_exclusion: sha256 de {p} = {sha} != {fse['sha256']} (config.json)")
        m = np.load(p)
        if m.dtype != bool or m.shape != (N,):
            raise RuntimeError(f"fit_set_exclusion: {p} dtype/forma {m.dtype}/{m.shape} != bool/({N},)")
        if m[:T].any():
            raise RuntimeError("fit_set_exclusion: la mascara marca filas de entrenamiento")
        return m, {"mask_source": "cache aux_dup_train.npy (config.regions.fit_set_exclusion)", "path": str(p),
                   "sha256": sha, "n_marked_total": int(m.sum())}
    kind = ((cfg.get("spec") or {}).get("kind")) or ("knad" if str((cfg.get("manifest_row") or {}).get("dataset", "")).upper().startswith("KNAD:") else None)
    if kind != "knad":
        raise RuntimeError("fit_set_exclusion sin mascara en un stream no KNAD: fail-closed (plan §2.1, R-3)")
    cpath = Path(cache.get("path") or "")
    xsha = ((cache.get("x") or {}).get("sha256"))
    key = (str(cpath), xsha, int(T))
    if key in _KNAD_DUP_MEMO:
        m, info = _KNAD_DUP_MEMO[key]
        return m, dict(info, memo=True)
    xp = cpath / "x.npy"
    meta = C.read_json(cpath / "meta.json") or {}
    if not xp.exists() or ((meta.get("x") or {}).get("sha256")) != xsha:
        raise RuntimeError(f"KNAD: x.npy de la cache {cpath} ausente o con sha256 distinto al de config.cache "
                           "(fail-closed: no se puede calcular F_cal)")
    import pipelines as P
    t0 = time.time()
    xr = np.load(xp, mmap_mode="r")
    if xr.shape[0] != N:
        raise RuntimeError(f"KNAD: x de la cache con {xr.shape[0]} filas != N {N}")
    wa = P.t0_module(REPO)
    m, info = P.train_duplicate_mask(xr, T, wa.hash_matrix)
    del xr
    info = dict(info, mask_source="pipelines.train_duplicate_mask(x cache, T, warmup_audit.hash_matrix)",
                cache_x=str(xp), cache_x_sha256_meta=xsha, seconds=round(time.time() - t0, 1),
                sha256=hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest(),
                warmup_audit_sha256=C.sha256_file(REPO / "revision" / "analysis" / "warmup_audit.py"))
    log(f"  KNAD F_cal: mascara de duplicados calculada en {info['seconds']} s ({info.get('n')} filas)")
    _KNAD_DUP_MEMO[key] = (m, info)
    return m, info


_TWIN_HASH_MEMO = {}


def _rows_equal(xr, a_rows, b_rows):
    A, B = np.asarray(xr[a_rows], dtype=np.float64), np.asarray(xr[b_rows], dtype=np.float64)
    return ((A == B) | (np.isnan(A) & np.isnan(B))).all(axis=1)


def _twins_of(xr, h, ref_rows, query_rows, step=100_000):
    """Mascara sobre query_rows: el vector de x es IDENTICO al de alguna fila de ref_rows (hash + comparacion exacta)."""
    out = np.zeros(len(query_rows), bool)
    if not len(ref_rows) or not len(query_rows):
        return out
    hr = h[ref_rows]
    order = np.argsort(hr, kind="stable")
    sh = hr[order]
    hq = h[query_rows]
    k = np.searchsorted(sh, hq)
    kc = np.minimum(k, len(sh) - 1)
    hit = (k < len(sh)) & (sh[kc] == hq)
    qi = np.flatnonzero(hit)
    first = ref_rows[order[kc[hit]]]
    eq = np.zeros(len(qi), bool)
    for a in range(0, len(qi), step):
        eq[a:a + step] = _rows_equal(xr, query_rows[qi[a:a + step]], first[a:a + step])
    for j in np.flatnonzero(~eq):
        lo, hi = np.searchsorted(sh, hq[qi[j]], "left"), np.searchsorted(sh, hq[qi[j]], "right")
        cand = ref_rows[order[lo:hi]]
        eq[j] = bool(_rows_equal(xr, np.full(len(cand), query_rows[qi[j]]), cand).any())
    out[qi[eq]] = True
    return out


def calib_eval_twins(cfg, N, fcal, E_rows, log=print):
    """[EXPLORATORIO] S-DUPCAL (F-01, MR-11): mascara sobre E_rows de las filas cuyo vector de entrada del modelo (x.npy
    de la cache del stream) es identico al de alguna fila de F_cal; y recuento de filas de F_cal con gemelo en E.
    Solo features (sin etiquetas ni scores). Hash por fila = warmup_audit.hash_matrix (T0), confirmado por valores."""
    cache = cfg.get("cache") or {}
    cpath = Path(cache.get("path") or "")
    xsha = ((cache.get("x") or {}).get("sha256"))
    xp = cpath / "x.npy"
    meta = C.read_json(cpath / "meta.json") or {}
    if not xp.exists() or not xsha or ((meta.get("x") or {}).get("sha256")) != xsha:
        raise RuntimeError(f"x.npy de la cache {cpath} ausente o con sha256 distinto al de config.cache")
    xr = np.load(xp, mmap_mode="r")
    if xr.shape[0] != N:
        raise RuntimeError(f"x de la cache con {xr.shape[0]} filas != N {N}")
    t0 = time.time()
    key = (str(xp), xsha)
    h = _TWIN_HASH_MEMO.get(key)
    if h is None:
        import pipelines as P
        wa = P.t0_module(REPO)
        h = np.empty(N, np.uint64)
        for a in range(0, N, P.HASH_CHUNK):
            h[a:a + P.HASH_CHUNK] = wa.hash_matrix(xr[a:a + P.HASH_CHUNK])
        _TWIN_HASH_MEMO.clear()
        _TWIN_HASH_MEMO[key] = h
    frows = np.flatnonzero(fcal).astype(np.int64)
    erows = np.asarray(E_rows, dtype=np.int64)
    twE = _twins_of(xr, h, frows, erows)
    twF = _twins_of(xr, h, erows, frows)
    info = {"x_source": f"{xp} (entrada del modelo)", "x_sha256_meta": xsha,
            "n_F_cal_rows_with_twin_in_E": int(twF.sum()), "n_F_cal": int(len(frows)),
            "method": "warmup_audit.hash_matrix por fila + comparacion exacta de valores (-0 == +0, NaN == NaN)",
            "seconds": round(time.time() - t0, 1)}
    del xr
    return twE, info


def load_aux(cfg, N, T, labels, dataset):
    """Timestamps (s, float con NaN), dia (codigo entero + nombre) y tipo de ataque por fila; alineacion comprobada
    con y de la cache en [T, N) (en T4-E labels.npy lleva las filas inyectadas en el entrenamiento)."""
    cache = cfg.get("cache") or {}
    cpath = Path(cache.get("path") or "")
    meta = C.read_json(cpath / "meta.json") or {}
    out = {"ts": None, "day": None, "day_names": None, "types": None, "info": {"cache": str(cpath)}}
    ypath = cpath / "y.npy"
    if ypath.exists():
        yc = np.load(ypath, mmap_mode="r")
        if len(yc) != N or not np.array_equal(np.asarray(yc[T:]).astype(np.int64), labels[T:].astype(np.int64)):
            raise RuntimeError("labels.npy del job != y.npy de la cache en [T, N): auxiliares desalineados")
        out["info"]["cache_y_matches_labels_T_N"] = True
    else:
        raise RuntimeError(f"falta {ypath}: no se puede comprobar la alineacion de los auxiliares")
    aux = meta.get("aux") or {}

    def load(name):
        a = aux.get(name)
        if not a:
            return None
        p = cpath / a["file"]
        sha = C.sha256_file(p)
        if a.get("sha256") and sha != a["sha256"]:
            raise RuntimeError(f"aux {name}: sha256 {sha} != meta {a['sha256']}")
        arr = np.load(p)
        if len(arr) != N:
            raise RuntimeError(f"aux {name}: {len(arr)} filas != N {N}")
        out["info"][name] = {"path": str(p), "sha256": sha}
        return arr

    ds = str(dataset)
    if ds.upper().startswith("KNAD:"):
        attack = ds.split(":", 1)[1]
        pz = T0_CACHE / "pcap" / f"{attack}.npz"
        lz = T0_CACHE / "labels" / "knad" / f"{attack}.npz"
        try:
            with np.load(pz, allow_pickle=False) as z:
                ts = np.asarray(z["ts"], dtype=np.float64)
            with np.load(lz, allow_pickle=False) as z:
                codes = np.asarray(z["codes"])
            if len(codes) != N or not np.array_equal(codes[T:].astype(np.int64), labels[T:].astype(np.int64)):
                raise RuntimeError("etiquetas de la cache T0 != labels.npy en [T, N)")
            if len(ts) < N:
                ts = np.concatenate([ts, np.full(N - len(ts), np.nan)])
            out["ts"] = ts[:N]
            out["info"]["ts"] = {"path": str(pz), "sha256": C.sha256_file(pz), "source": "pcap (cache T0, solo lectura)",
                                 "labels_check": str(lz)}
        except Exception as e:
            out["info"]["ts"] = {"available": False, "reason": f"{type(e).__name__}: {e}"}
        out["types"] = np.where(labels.astype(bool), attack, "")
        out["info"]["types"] = {"source": "KNAD: un solo tipo por captura"}
        return out
    ts = load("ts")
    if ts is not None:
        tsf = ts.astype(np.float64)
        na = ts == TS_NA
        tsf[na] = np.nan
        out["ts"] = tsf
        dnum = np.where(na, -1, ts // 86400).astype(np.int64)
        uniq = np.unique(dnum)
        out["day"] = dnum
        out["day_names"] = {int(u): ("NA" if u < 0 else str(np.datetime64(int(u), "D"))) for u in uniq}
        out["info"]["day"] = {"source": "fecha del timestamp corregido (aux_ts)"}
    files = (((meta.get("extra") or {}).get("t0") or {}).get("files")) or []
    if ds == "CIC2017":
        sf = load("src_file")
        if sf is None:
            raise RuntimeError("CIC2017 sin aux_src_file: no hay dia por fila")
        names = []
        for f in files:
            pre = f.split("-")[0]
            if pre not in CIC2017_DAY_BY_PREFIX:
                raise RuntimeError(f"CIC2017: fichero sin dia reconocible {f!r}")
            names.append(CIC2017_DAY_BY_PREFIX[pre])
        order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        code_of_file = np.array([order.index(nm) for nm in names], dtype=np.int64)
        out["day"] = code_of_file[sf.astype(np.int64)]
        out["day_names"] = {i: nm for i, nm in enumerate(order)}
        out["info"]["day"] = {"source": "fichero CSV (aux_src_file -> t0.files)", "files": files}
    lc = load("label_code")
    vocab = (((meta.get("extra") or {}).get("labels") or {}).get("vocab"))
    if lc is not None and vocab:
        out["types"] = np.array(vocab, dtype=object)[lc.astype(np.int64)].astype(str)
        out["info"]["types"] = {"source": "aux_label_code + extra.labels.vocab"}
    return out


def day_guard(day, N):
    change = np.flatnonzero(day[1:] != day[:-1]) + 1
    last = np.full(N, -10 ** 12, np.int64)
    if len(change):
        idx = np.searchsorted(change, np.arange(N), side="right") - 1
        ok = idx >= 0
        last[ok] = change[idx[ok]]
    return (np.arange(N) - last) < DAY_GUARD


def sanitize_scores(s, start):
    """§12.2: NaN, +inf (y -inf, MR-7) -> +inf. Detectores: +inf -> maximo finito observado en [start, t] (causal).
    RAW: +inf -> maximo finito + 1 (mismos rangos). Devuelve (s_det, s_raw, info)."""
    s = np.asarray(s, dtype=np.float64)
    nonfin = ~np.isfinite(s)
    info = {"n_nan": int(np.isnan(s).sum()), "n_posinf": int(np.isposinf(s).sum()),
            "n_neginf": int(np.isneginf(s).sum())}
    if not nonfin.any():
        return s, s, dict(info, replaced=0)
    s_det = s.copy()
    tail = s[start:]
    fin = np.where(np.isfinite(tail), tail, -np.inf)
    runmax = np.maximum.accumulate(fin)
    runmax[~np.isfinite(runmax)] = 0.0
    bad = ~np.isfinite(tail)
    s_det[start:][bad] = runmax[bad]
    s_det[:start][~np.isfinite(s_det[:start])] = 0.0
    s_raw = s.copy()
    mx = float(np.max(s[np.isfinite(s)])) if np.isfinite(s).any() else 0.0
    s_raw[nonfin] = mx + 1.0
    return s_det, s_raw, dict(info, replaced=int(nonfin.sum()), rule="MR-7")


def job_descriptor(cfg):
    row = cfg.get("manifest_row") or {}
    ds = row.get("dataset")
    var = row.get("variant")
    ar = C.parse_bool(row.get("ar")) if row.get("ar") not in (None, "") else None
    arch_i = ARCH_ORDER.index(var) if var in ARCH_ORDER else None
    if arch_i is not None:
        cid = f"C{2 * arch_i + (1 if ar else 2)}"
    elif (var, ar) in CTX_CONFIG_IDS:
        cid = CTX_CONFIG_IDS[(var, ar)]
    else:
        cid = {"mlp": "C9", "original": "C10", "stat": "C11"}.get(var)
    block = row.get("block", "")
    rule = C.ABLATION_BLOCKS.get(block)
    ctx = C.CTX_BLOCKS.get(block)
    return {"job_id": cfg.get("job_id"), "block": block, "family": C.block_family(block), "dataset": ds,
            "dataset_id": DATASET_IDS.get(ds), "dataset_key": C.label_counts_key(ds),
            "corpus": "KNAD" if str(ds).upper().startswith("KNAD:") else ds,
            "variant": var, "ar": ar, "paradigm": row.get("paradigm"), "config_id": cid,
            "mlp_replica": ("AR" if ar else "TSR") if var == "mlp" else None,
            "deterministic": var in ("original", "stat", "original_eps"), "seed": row.get("seed"),
            "seed_int": C.SEEDS.get(row.get("seed")), "rep": row.get("rep"),
            "pipeline": row.get("pipeline"), "pipeline_version": row.get("pipeline_version"),
            "window": row.get("window"), "k_clusters": row.get("k_clusters"), "grouping": row.get("grouping"),
            "partition_seed": row.get("partition_seed"), "contamination": row.get("contamination"),
            "ablation": ({"factor": rule["factor"], "plan": rule["plan"],
                          "reference_block": C.ablation_reference_block(block, ds)} if rule else None),
            "exploratory_arm": bool(ctx),
            "exploratory_arm_rule": ({"decision": ctx["decision"], "note": ctx["note"],
                                  "config_name": CTX_CONFIG_NAMES.get(cid),
                                  "reference_block": f"T1-A-{'KNAD' if str(ds).upper().startswith('KNAD:') else {'ACI-IOT-2023': 'ACI'}.get(ds, ds)}",
                                  "scope": "descriptivo: fuera de H1-H4, de S*_d, de R1.4 y del abstract"}
                                 if ctx else None),
            "runner_tag": ((cfg.get("code") or {}).get("runner_tag"))}


def applicable(cfg, job_dir):
    row = cfg.get("manifest_row") or {}
    fam = C.block_family(row.get("block", ""))
    if fam not in ("T1", "T2", "T3", "T4", "T5", "T7", "T8"):
        raise NotApplicable(f"familia {fam} ({row.get('block')}): sin metricas del protocolo (X2 = regresion; S = "
                            "sondas)")
    if row.get("pipeline") != "corrected":
        raise NotApplicable(f"pipeline {row.get('pipeline')}: solo 'corrected' (el 'orig' es la fuga del envio)")
    for f in ("scores.npy", "labels.npy", "config.json", "DONE"):
        if not (Path(job_dir) / f).exists():
            raise NotApplicable(f"falta {f}")


class Ctx:
    """Estado de un job: stream, etiquetas, regiones y auxiliares (las etiquetas solo se usan al evaluar)."""


def _series_eval(ser, st, rows, y, xs, taus, ts=None, o2=False, wd=True, q_tag=None, triv=None):
    out = {"trivial": triv if triv is not None else trivial_refs(y)}
    for x in xs:
        xe = ser[x][rows - st]
        d = curve_metrics(xe, y, o2=(o2 and x in X_O2))
        d = _add_apn(d, out["trivial"])
        d.update(threshold_metrics(xe, y, taus[x], out["trivial"]))
        if "o2" in d:
            d["o2"]["gap_vs_calibrated_f1"] = d["o2"]["f1_max"] - d["f1"]
            d["o2"]["tag"] = TAGS["oracle"]
        if wd:
            d["windowdiff"] = wd_block(y.astype(np.int8), (xe > taus[x]).astype(np.int8))
        out[x] = d
    return out


def _thr(params, q=Q_MAIN):
    return params["thresholds"][str(q)]


def _fit_and_transform(ctx, fit_mask_abs, series_start, calib_end, s_det=None):
    """s_det: serie saneada de forma causal desde series_start (§12.2; F-06). Por defecto la del protocolo (desde T)."""
    s = ctx.s_det if s_det is None else s_det
    pre = slice(series_start, calib_end)
    params = fit(s[pre], fit_mask_abs[pre], series_start)
    ser = transform(params, s[series_start:])
    ser["RAW"] = ctx.s_raw[series_start:]
    return params, ser


def _params_summary(p):
    return {k: p[k] for k in ("series_start", "n_prefix", "n_fit", "mvd", "kmd", "fusion", "thresholds")}


def per_day_partition(day, day_names, N, E_rows, y):
    """Solo etiquetas (§3.5): por dia, filas de E del dia sin las 800 primeras tras cada cambio de dia del stream,
    episodios G = 10.000 recalculados en esas filas y elegibilidad del AUC por dia (>= 3 episodios, >= 1.000 benignas).
    Devuelve [(nombre, filas de E del dia, filas conservadas, dict de recuentos)]."""
    guard = day_guard(day, N)
    e_mask = np.zeros(N, bool)
    e_mask[E_rows] = True
    out = []
    for code in np.unique(day[E_rows]).tolist():
        d = day_names[int(code)]
        m = (day == code) & e_mask
        keep = np.flatnonzero(m & ~guard)
        yy = np.asarray(y)[keep].astype(np.int8)
        n0 = int((yy == 0).sum())
        eps_s, _ = episodes(yy, G_EP)
        rec = {"rows_E": int(m.sum()), "rows_after_guard": int(len(keep)), "attack_rows": int(yy.sum()),
               "benign_rows": n0, "episodes_G10000_within_day": int(len(eps_s)),
               "eligible_auc_per_day": bool(len(eps_s) >= 3 and n0 >= 1_000)}
        out.append((d, keep, yy, rec))
    return out


def section_per_day(ctx, ser, st, taus):
    if ctx.aux["day"] is None:
        return {"available": False, "reason": "sin dia por fila (KNAD: una captura continua; §3.5 solo CIC/ACI)"}
    fus = ser["FUS"]
    extras = (ctx.label_extras.get(ctx.desc["dataset_key"]) or {}).get("per_day") or {}
    out = {"available": True, "guard_rows": DAY_GUARD, "metric_series": "FUS", "days": {},
           "rule": "filas de E del dia sin las 800 primeras tras cada cambio de dia (§3.5)", "tag": TAGS["expl"]}
    mism = []
    for d, keep, yy, rec in per_day_partition(ctx.aux["day"], ctx.aux["day_names"], ctx.N, ctx.E_rows, ctx.y):
        xx = fus[keep - st]
        alarm = xx > taus["FUS"]
        n0 = rec["benign_rows"]
        eligible = rec["eligible_auc_per_day"]
        tsd = ctx.aux["ts"][keep] if ctx.aux["ts"] is not None else None
        rec = dict(rec, fpr=(int(np.count_nonzero(alarm & (yy == 0))) / n0) if n0 else None,
                   adjacent_to_calibration=bool(ctx.desc["dataset"] == "CIC2017" and d == "Mon"))
        op = operational(yy, alarm, G_EP, tsd, per_episode=not eligible)
        rec["false_alarm_episodes_per_1e6_benign"] = op["false_alarm_episodes_per_1e6_benign"]
        rec["n_false_alarm_episodes"] = op["n_false_alarm_episodes"]
        if eligible:
            rec["auroc_fus"] = auroc(xx, yy.astype(bool))
            rec["auc_t"] = auc_index(yy.astype(bool))
        else:
            rec["episodes_detection"] = {k: op[k] for k in ("n_episodes", "n_detected", "detection_rate",
                                                            "median_delay_rows", "median_delay_s", "episodes")}
        rec["windowdiff_fus"] = wd_block(yy, alarm.astype(np.int8)) if len(yy) > 1 else None
        ref = extras.get(d)
        if ref is not None:
            for k in ("rows_E", "rows_after_guard", "attack_rows", "benign_rows", "episodes_G10000_within_day",
                      "eligible_auc_per_day"):
                if ref.get(k) != rec[k]:
                    mism.append({"day": d, "key": k, "plan_support": ref.get(k), "obtained": rec[k]})
        out["days"][d] = rec
    out["label_extras_check"] = {"mismatches": mism, "matches": not mism, "n_days_plan_support": len(extras)}
    return out


def section_per_type(ctx, ser, st, taus):
    if ctx.aux["types"] is None:
        return {"available": False}
    types = ctx.aux["types"][ctx.E_rows]
    yb = ctx.yE
    out = {"available": True, "rule": "recall calibrado (X > tau) por tipo de ataque en E (§3.5, R2.4)",
           "tag": TAGS["expl"], "types": {}}
    for t in sorted(set(types[yb].tolist())):
        m = yb & (types == t)
        rec = {"n_rows": int(m.sum())}
        for x in X_MAIN:
            xe = ser[x][ctx.E_rows[m] - st]
            rec[f"recall_{x}"] = float(np.count_nonzero(xe > taus[x]) / len(xe)) if len(xe) else None
        out["types"][t] = rec
    return out


def compute_job(job_dir, opts, log=print):
    """Calcula todas las metricas de un job y devuelve el dict de metrics.json (sin escribirlo)."""
    t_all = time.time()
    tm = {}
    J = Path(job_dir)
    cfg = C.read_json(J / "config.json")
    if not cfg:
        raise RuntimeError(f"config.json ilegible en {J}")
    applicable(cfg, J)
    desc = job_descriptor(cfg)
    regions = cfg["regions"]
    ctx = Ctx()
    ctx.desc = desc
    t0 = time.time()
    s = np.load(J / "scores.npy")
    labels = np.load(J / "labels.npy")
    arrays = C.read_json(J / "arrays.json") or {}
    if s.dtype != np.float64 or s.ndim != 1:
        raise RuntimeError(f"scores.npy {s.dtype}/{s.shape}: se esperaba float64 (N,)")
    N = int(len(s))
    if labels.shape != (N,) or not set(np.unique(labels).tolist()) <= {0, 1}:
        raise RuntimeError("labels.npy con forma o valores inesperados")
    if int(regions.get("n_rows", -1)) != N:
        raise RuntimeError(f"regions.n_rows {regions.get('n_rows')} != len(scores) {N}")
    T = int(regions["train"][1])
    fm = int(regions["fm"][1])
    ctx.N, ctx.T, ctx.fm = N, T, fm
    ctx.y = labels.astype(bool)
    tm["load_s"] = round(time.time() - t0, 2)
    calib, E, rinfo = get_regions(regions, PRIMARY_CANDIDATE)
    ctx.calib = calib
    ctx.E_rows = rows_of(E)
    ctx.yE = ctx.y[ctx.E_rows]
    Cn = calib[1] - calib[0]
    if calib[0] != T:
        raise RuntimeError(f"calibracion primaria {calib} no empieza en T = {T}")
    lc = C.read_json(REPO / LABEL_COUNTS_REL) or {}
    ctx.label_extras = C.read_json(REPO / LABEL_EXTRAS_REL) or {}
    lce = lc.get(desc["dataset_key"]) or {}
    t0 = time.time()
    dup, fsinfo = fit_set_mask(cfg, N, T, log=log)
    fcal = np.zeros(N, bool)
    fcal[calib[0]:calib[1]] = True
    fcal &= ~dup
    ctx.dup, ctx.fcal = dup, fcal
    tm["fit_set_s"] = round(time.time() - t0, 2)
    n_fit = int(fcal.sum())
    fit_set = {"rule": "F_cal = calibracion sin duplicados exactos de [0, T) (plan §2.1); los duplicados siguen en "
                       "el stream, en las ventanas y en E", "n_calib": int(Cn), "n_dup_excluded": int(Cn - n_fit),
               "n_fit": n_fit, **fsinfo,
               "labels_reported_only": {"attack_rows_in_F_cal": int((ctx.y & fcal).sum()),
                                        "attack_frac_F_cal": float((ctx.y & fcal).sum() / n_fit),
                                        "attack_rows_in_calib": int(ctx.y[calib[0]:calib[1]].sum()),
                                        "note": "solo se reportan (eje x de R1.1); no intervienen en el ajuste"}}
    cons = label_consistency(ctx, lce, calib, E, fit_set)
    s_det, s_raw, nfinfo = sanitize_scores(s, T)
    ctx.s_det, ctx.s_raw = s_det, s_raw
    ex = s[T:]
    frac = float((~np.isfinite(ex)).mean()) if len(ex) else 0.0
    nonfinite = dict(nfinfo, arrays_json={"n_nan": (arrays.get("scores.npy") or {}).get("n_nan"),
                                          "n_inf": (arrays.get("scores.npy") or {}).get("n_inf")},
                     frac_nonfinite_exec=frac, divergent=bool(frac > DIVERGENCE_FRAC),
                     rule="§12.2: NaN y +inf -> +inf; > 1 % en [T, N) = ejecucion divergida")
    ctx.aux = load_aux(cfg, N, T, labels, desc["dataset"])
    t0 = time.time()
    params, ser = _fit_and_transform(ctx, fcal, T, calib[1])
    tm["fit_transform_main_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    taus = _thr(params)
    tsE = ctx.aux["ts"][ctx.E_rows] if ctx.aux["ts"] is not None else None
    main = _series_eval(ser, T, ctx.E_rows, ctx.yE, X_MAIN, taus, o2=True)
    triv = main["trivial"]
    triv["duration_hours"] = (float((np.nanmax(tsE) - np.nanmin(tsE)) / 3600.0)
                              if tsE is not None and np.isfinite(tsE).any() else None)
    from sklearn.metrics import roc_auc_score
    t1 = time.time()
    sk = float(roc_auc_score(ctx.yE, ser["FUS"][ctx.E_rows - T]))
    fast = main["FUS"]["auroc"]
    if fast is None or abs(sk - fast) > 1e-12:
        raise RuntimeError(f"replica de ROC-AUC {fast} != sklearn {sk} (MR-1)")
    primary = {"metric": "ROC-AUC de FUS en E (plan §3.1)", "value": sk, "source": "sklearn.metrics.roc_auc_score",
               "fast_replica": fast, "abs_diff": abs(sk - fast), "sklearn_seconds": round(time.time() - t1, 2),
               "auc_t": triv["auc_t"], "tag": TAGS["primary"]}
    if desc.get("exploratory_arm"):
        primary["tag"] = TAGS["expl"] + f" {desc['exploratory_arm_rule']['decision']}: fuera del plan congelado"
        primary["confirmatory"] = False
    tm["eval_main_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    sloc = {x: auc_loc(ser[x][ctx.E_rows - T], ctx.yE) for x in X_MAIN}
    primary["auc_loc_raw"] = sloc["RAW"]["auc_loc"]
    alarmE = ser["FUS"][ctx.E_rows - T] > taus["FUS"]
    yE8 = ctx.yE.astype(np.int8)
    op = {"G10000": operational(yE8, alarmE, G_EP, tsE), "G800": operational(yE8, alarmE, G_EP_SENS, tsE)}
    op["G10000"]["tag"] = TAGS["secondary"]
    op["G800"]["tag"] = TAGS["sens"] + " S-G800"
    ep_ref = np.zeros(len(yE8), np.int8)
    for a, b in zip(*episodes(yE8, G_EP)):
        ep_ref[a:b] = 1
    s_ep = {x: wd_block(ep_ref, (ser[x][ctx.E_rows - T] > taus[x]).astype(np.int8)) for x in X_MAIN}
    per_day = section_per_day(ctx, ser, T, taus)
    per_type = section_per_type(ctx, ser, T, taus)
    tm["loc_ops_days_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    sens = {"S-RAW": {"see": "E.RAW", "tag": TAGS["sens"]},
            "S-G800": {"see": "operational.G800", "tag": TAGS["sens"]},
            "S-LOC": dict(sloc, tag=TAGS["sens"], note="AUC_loc(RAW) entra en la regla de robustez §5.8; FUS, KMD_f y "
                                                      "MVD_f solo descriptivos (filtros > bloque)"),
            "S-EP": dict(s_ep, tag=TAGS["sens"], reference="indicador de episodios G = 10.000 en E; k por la misma regla")}
    for name, q in Q_SENS.items():
        tq = _thr(params, q)
        sens[name] = {"q": q, "tag": TAGS["sens"]}
        for x in X_MAIN:
            xe = ser[x][ctx.E_rows - T]
            d = threshold_metrics(xe, ctx.yE, tq[x], triv)
            d["windowdiff"] = wd_block(yE8, (xe > tq[x]).astype(np.int8))
            sens[name][x] = d
    sens["S-MAX"] = dict(_series_eval(ser, T, ctx.E_rows, ctx.yE, ("FUS_MAX",), taus, triv=triv),
                         tag=TAGS["sens"], rule="max(z_KMD_f, z_MVD_f); umbral P99 de F_cal")
    sens["S-NF"] = dict(_series_eval(ser, T, ctx.E_rows, ctx.yE, ("FUS_NF", "KMD_raw", "MVD_raw"), taus, triv=triv),
                        tag=TAGS["sens"], rule="detectores sin filtro; misma fusion (mediana/MAD de F_cal) y P99")
    del sens["S-MAX"]["trivial"], sens["S-NF"]["trivial"]
    dupE = dup[ctx.E_rows]
    if dupE.any():
        keep = ctx.E_rows[~dupE]
        sd = _series_eval(ser, T, keep, ctx.y[keep], X_MAIN, taus)
        sens["S-DUP"] = dict(sd, applicable=True, n_removed=int(dupE.sum()),
                             n_removed_attack=int((dupE & ctx.yE).sum()), tag=TAGS["sens"])
    else:
        sens["S-DUP"] = {"applicable": False, "n_removed": 0, "reason": "0 duplicados de entrenamiento en E",
                         "tag": TAGS["sens"]}
    tm["sens_main_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    try:
        twE, twinfo = calib_eval_twins(cfg, N, fcal, ctx.E_rows, log=log)
        blk = {"applicable": bool(twE.any()), "n_E_rows_with_twin_in_F_cal": int(twE.sum()),
               "n_E_attack_rows_with_twin": int((twE & ctx.yE).sum()), **twinfo, "tag": TAGS["dupcal"],
               "rule": "MR-11: metricas en E sin las filas cuyo vector de entrada (x.npy de la cache) es identico al de "
                       "alguna fila de F_cal; F_cal no se toca (quitar filas de F_cal por su gemelo en E miraria "
                       "hacia delante)"}
        if twE.any():
            keep = ctx.E_rows[~twE]
            blk.update(_series_eval(ser, T, keep, ctx.y[keep], X_MAIN, taus))
        else:
            blk["reason"] = "0 filas de E con gemelo exacto en F_cal"
        sens["S-DUPCAL"] = blk
    except Exception as e:
        sens["S-DUPCAL"] = {"error": f"{type(e).__name__}: {e}", "tag": TAGS["dupcal"]}
        section_errors_pre = [f"S-DUPCAL: {e}"]
    else:
        section_errors_pre = []
    tm["s_dupcal_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    full = np.zeros(N, bool)
    full[T + W_MVD + F_FILT - 2:] = True
    fps_mask = fcal & full
    if int(fps_mask.sum()) >= PS_MIN_ROWS:
        p_ps, ser_ps = _fit_and_transform(ctx, fps_mask, T, calib[1])
        sens["S-PS"] = dict(_series_eval(ser_ps, T, ctx.E_rows, ctx.yE, X_MAIN, _thr(p_ps), triv=triv),
                            applicable=True, n_fit=int(fps_mask.sum()), params=_params_summary(p_ps),
                            rule="MR-3: todos los objetos ajustados en F_cal ∩ {t >= T + W + F - 2}", tag=TAGS["sens"])
        del sens["S-PS"]["trivial"], ser_ps
    else:
        sens["S-PS"] = {"applicable": False, "n_fit_full_support": int(fps_mask.sum()),
                        "reason": f"< {PS_MIN_ROWS} filas de F_cal con soporte completo (NA; §2.9)", "tag": TAGS["sens"]}
    tm["s_ps_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    section_errors = list(section_errors_pre)
    try:
        c_ins, e_ins, info_ins = get_regions(regions, INS_CANDIDATE)
        if not segments_subset(segments_of(E), segments_of(e_ins)):
            raise RuntimeError("E primaria no es subconjunto de la eval de la candidata in-sample")
        m_ins = np.zeros(N, bool)
        m_ins[c_ins[0]:c_ins[1]] = True
        s_det_ins = sanitize_scores(s, fm)[0] if nfinfo.get("replaced") else ctx.s_det
        p_ins, ser_ins = _fit_and_transform(ctx, m_ins, fm, c_ins[1], s_det=s_det_ins)
        del s_det_ins
        sens["S-INS"] = dict(_series_eval(ser_ins, fm, ctx.E_rows, ctx.yE, X_MAIN, _thr(p_ins), triv=triv),
                             in_sample=True, regions=info_ins, series_start=fm, n_fit=int(m_ins.sum()),
                             params=_params_summary(p_ins), rule="MR-4", tag=TAGS["sens"] + " [IN-SAMPLE]")
        del sens["S-INS"]["trivial"], ser_ins
    except Exception as e:
        sens["S-INS"] = {"error": f"{type(e).__name__}: {e}", "in_sample": True}
        section_errors.append(f"S-INS: {e}")
    tm["s_ins_s"] = round(time.time() - t0, 2)
    t4p = {"applicable": desc["family"] == "T1"}
    if desc["family"] == "T1":
        t0 = time.time()
        for key, fn in (("a", section_t4pa), ("b", section_t4pb), ("c", section_t4pc)):
            try:
                t4p[key] = fn(ctx, regions, params, ser, lce, triv)
            except Exception as e:
                t4p[key] = {"error": f"{type(e).__name__}: {e}"}
                section_errors.append(f"t4p.{key}: {e}")
        tm["t4p_s"] = round(time.time() - t0, 2)
    t4e_ref = None
    if (desc["block"] == "T1-A-KNAD" and desc["dataset"] == "KNAD:Active_Wiretap" and desc["seed"] == "A" and
            ((desc["variant"], None) in T4E_VARIANTS or (desc["variant"], desc["ar"]) in T4E_VARIANTS)):
        try:
            ex = C.t4e_eval_exclusion(desc["dataset"], REPO)
            _, Er, info_r = get_regions(regions, PRIMARY_CANDIDATE, extra_eval_exclude=[ex["excluded_with_guards"]])
            rr = rows_of(Er)
            t4e_ref = dict(_series_eval(ser, T, rr, ctx.y[rr], X_MAIN, taus), regions=info_r, exclusion=ex,
                           tag=TAGS["t4"] + " T4-E c = 0")
        except Exception as e:
            t4e_ref = {"error": f"{type(e).__name__}: {e}"}
            section_errors.append(f"t4e_reference: {e}")
    expl = {}
    if desc["dataset"] == "CIC2017":
        ed = block_edges(len(ctx.yE))
        rows = []
        for a, b in zip(ed[:-1], ed[1:]):
            yy = ctx.yE[a:b]
            n0 = int((~yy).sum())
            rows.append([int(ctx.E_rows[a]), n0, (int(np.count_nonzero(alarmE[a:b] & ~yy)) / n0) if n0 else None])
        expl["fpr_by_block_fus"] = {"rows": rows, "columns": ["first_row", "n_benign", "fpr"], "b": B_LOC,
                                    "tag": TAGS["expl"], "rule": "§3.4: FPR de FUS frente al tiempo desde la calibracion"}
    if opts.get("bootstrap_ci"):
        t0 = time.time()
        expl["ci_bootstrap_auroc_fus"] = bootstrap_auc_ci(ser["FUS"][ctx.E_rows - T], ctx.yE)
        tm["bootstrap_s"] = round(time.time() - t0, 2)
    else:
        expl["ci_bootstrap_auroc_fus"] = {"computed": False, "reason": "opt-in (--bootstrap-ci), §3.7 opcional"}
    divergence = None
    if nonfinite["divergent"]:
        p = triv["prevalence"]
        divergence = {"imputed_itt": {"auroc_fus": 0.5, "auc_loc_raw": 0.5, "auprc_fus": p, "f1_fus": 0.0},
                      "rule": "§12.2: en los analisis confirmatorios se imputan estos valores; sensibilidad sin imputar"}
    tm["total_s"] = round(time.time() - t_all, 2)
    out = {
        "schema": SCHEMA, "status": "ok", "job_id": desc["job_id"], "job": desc,
        "regions": dict(rinfo, N=N, T=T, fm=fm, C=int(Cn), E_rows=int(len(ctx.E_rows)), E_segments=segments_of(E),
                        fit_set=fit_set),
        "label_consistency": cons, "nonfinite": nonfinite, "divergence": divergence,
        "params": _params_summary(params), "primary": primary,
        "E": dict(main, tag=(TAGS["expl"] if desc.get("exploratory_arm") else TAGS["secondary"]),
                  note="ROC-AUC de FUS = primaria (ver 'primary'); el resto secundarias"),
        "operational": op, "per_day": per_day, "per_attack_type": per_type, "sensitivities": sens,
        "t4p": t4p, "t4e_reference": t4e_ref,
        "oracles": {"O2": {x: main[x].get("o2") for x in X_O2},
                    "O1": {"file": "metrics_o1.json", "seed_A_only": True,
                           "how": "metrics_rev.py --o1 (con /usr/bin/python3; MR-8)"}},
        "exploratory": expl, "section_errors": section_errors, "timing": tm,
        "aux_inputs": ctx.aux["info"],
    }
    if desc.get("exploratory_arm"):
        out["exploratory_job"] = dict(desc["exploratory_arm_rule"], config_id=desc["config_id"], block=desc["block"],
                                      variant=desc["variant"], tag=TAGS["expl"])
    return out


def label_consistency(ctx, lce, calib, E, fit_set):
    """Comprobaciones con los insumos congelados del plan (solo etiquetas). Duras: regiones y ataque/prevalencia de E
    (una discrepancia = stream distinto del del plan). Blandas (se registran): F_cal, k, episodios, AUC_t, S-LOC."""
    out = {"source": LABEL_COUNTS_REL, "hard": {}, "soft": {}}
    if not lce:
        raise RuntimeError(f"{LABEL_COUNTS_REL} sin la entrada {ctx.desc['dataset_key']!r}")
    ev_full = [calib[1], ctx.N]
    if list(lce.get("calibration", [])) != list(calib) or list(lce.get("evaluation", [])) != ev_full:
        raise RuntimeError(f"regiones {calib}/{ev_full} != label_counts {lce.get('calibration')}/{lce.get('evaluation')}")
    yfull = ctx.y[calib[1]:ctx.N]
    n1 = int(yfull.sum())
    p = n1 / len(yfull)
    ev = lce.get("eval") or {}
    if n1 != int(ev.get("attack_rows", -1)) or abs(p - float(ev.get("prevalence", -1))) > 1e-12:
        raise RuntimeError(f"ataques en E {n1} / p {p} != label_counts {ev.get('attack_rows')} / {ev.get('prevalence')}")
    out["hard"] = {"regions": True, "attack_rows_E": n1, "prevalence_E": p}
    cal = lce.get("calib") or {}
    soft = {"n_fit": (fit_set["n_fit"], cal.get("fit_rows_dedup")),
            "k_windowdiff": (wd_k(yfull.astype(np.int8))[0], (ev.get("segments") or {}).get("k_windowdiff_standard")),
            "episodes_G10000": (len(episodes(yfull.astype(np.int8), G_EP)[0]), ((ev.get("episodes") or {}).get("G10000") or {}).get("n")),
            "episodes_G800": (len(episodes(yfull.astype(np.int8), G_EP_SENS)[0]), ((ev.get("episodes") or {}).get("G800") or {}).get("n"))}
    lx = ctx.label_extras.get(ctx.desc["dataset_key"]) or {}
    soft["auc_t"] = (auc_index(yfull), lx.get("AUC_t"))
    cov = loc_coverage(yfull)
    for k in ("blocks", "blocks_both_classes", "attack_rows_covered", "benign_rows_covered"):
        soft[f"S_LOC_{k}"] = (cov[k], (lx.get("S_LOC_cover") or {}).get(k))
    for k, (a, b) in soft.items():
        ok = (a == b) if not isinstance(a, float) else (b is not None and abs(a - b) <= 1e-12)
        out["soft"][k] = {"obtained": a, "plan_support": b, "match": bool(ok)}
    out["all_soft_match"] = all(v["match"] for v in out["soft"].values())
    return out


def _t4_eval(ser, st, rows, y, taus, triv=None):
    out = _series_eval(ser, st, rows, y, X_T4, taus, wd=False, triv=triv)
    for x in X_T4:
        out[x] = {k: out[x][k] for k in ("auroc", "auprc", "threshold", "f1", "precision", "recall", "fpr",
                                          "alarm_rate", "tp", "fp", "tn", "fn")}
    return out


def section_t4pa(ctx, regions, params, ser, lce, triv):
    calib = ctx.calib
    yc = ctx.y[calib[0]:calib[1]]
    contam = float(yc.mean())
    if contam < T4PA_MIN_CONTAM:
        return {"applicable": False, "natural_calib_contamination": contam,
                "reason": "< 1 % de ataque en la calibracion (§8.2)"}
    cy = np.concatenate([[0], np.cumsum(ctx.y, dtype=np.int64)])
    t = np.arange(ctx.N, dtype=np.int64)
    prev_excl = cy[t] - cy[np.maximum(0, t - L_CTX)]
    clean = ctx.fcal & (~ctx.y) & (prev_excl == 0)
    n_clean = int(clean.sum())
    base = {"applicable": True, "natural_calib_contamination": contam, "n_clean": n_clean,
            "rule": "F_cal con y = 0 y sin ataques en las 800 filas anteriores (contrafactual NO desplegable; usa "
                    "etiquetas)", "tag": TAGS["t4"] + " T4-P-a [NO DESPLEGABLE]",
            "plan_support_clean_rows": (lce.get("calib") or {}).get("clean_rows_guard800_dedup")}
    if n_clean < T4PA_MIN_CLEAN:
        return dict(base, estimable=False, substitute="S-INS (sensitivities.S-INS)",
                    reason=f"< {T4PA_MIN_CLEAN} filas limpias (§8.2)")
    p_cl, ser_cl = _fit_and_transform(ctx, clean, ctx.T, calib[1])
    nat = _t4_eval(ser, ctx.T, ctx.E_rows, ctx.yE, _thr(params), triv)
    cl = _t4_eval(ser_cl, ctx.T, ctx.E_rows, ctx.yE, _thr(p_cl), triv)
    diff = {x: {k: (nat[x][k] - cl[x][k]) if (nat[x][k] is not None and cl[x][k] is not None) else None
                for k in ("auroc", "auprc", "threshold", "f1", "precision", "recall", "fpr")} for x in X_T4}
    return dict(base, estimable=True, natural=nat, clean=cl, diff_natural_minus_clean=diff,
                params_clean=_params_summary(p_cl))


def section_t4pb(ctx, regions, params, ser, lce, triv):
    calib = ctx.calib
    if int(ctx.y[calib[0]:calib[1]].sum()) != 0:
        return {"applicable": False, "reason": "calibracion con ataques naturales (T4-P-b solo KNAD x6 y CIC-IDS2017)"}
    pool_ref = lce.get("t4p_donor_pool") or {}
    if not pool_ref.get("pool"):
        return {"applicable": False, "reason": "sin t4p_donor_pool en label_counts"}
    n_fit = int(ctx.fcal.sum())
    need = int(math.ceil(T4P_POOL_FRAC * n_fit))
    es, ee = calib[1], ctx.N
    att = es + np.flatnonzero(ctx.y[es:ee])
    if len(att) < need:
        raise RuntimeError(f"E con {len(att)} ataques < {need}")
    a, b = int(att[0]), int(att[need - 1]) + 1
    xs, xe = max(es, a - T4P_GUARD), min(ee, b + T4P_GUARD)
    if [a, b] != list(pool_ref["pool"]) or [xs, xe] != list(pool_ref["excluded_with_guards"]) or \
            need != int(pool_ref.get("attack_rows_needed", -1)):
        raise RuntimeError(f"pool recalculado {[a, b]}/{[xs, xe]}/{need} != label_counts {pool_ref}")
    _, E4, info4 = get_regions(regions, PRIMARY_CANDIDATE, extra_eval_exclude=[[xs, xe]])
    r4 = rows_of(E4)
    y4 = ctx.y[r4]
    triv4 = trivial_refs(y4)
    Cn = calib[1] - calib[0]
    donors = att[:need]
    levels = []
    for c in T4P_LEVELS:
        B = int(round(c * n_fit))
        off = min((Cn - B) // 2, Cn - REACH - B)
        if off < 0:
            raise RuntimeError(f"c={c}: desplazamiento negativo {off}")
        rec = {"c": c, "B_c": B, "offset": int(off), "injected_rows": [int(calib[0] + off), int(calib[0] + off + B)]}
        if B == 0:
            p_c, ser_c = params, ser
            inj_fit = 0
        else:
            sd_save, sr_save = ctx.s_det, ctx.s_raw
            sm = ctx.s_det.copy()
            sm[calib[0] + off:calib[0] + off + B] = ctx.s_det[donors[:B]]
            smr = ctx.s_raw.copy()
            smr[calib[0] + off:calib[0] + off + B] = ctx.s_raw[donors[:B]]
            ctx.s_det, ctx.s_raw = sm, smr
            try:
                p_c, ser_c = _fit_and_transform(ctx, ctx.fcal, ctx.T, calib[1])
            finally:
                ctx.s_det, ctx.s_raw = sd_save, sr_save
            inj = np.zeros(ctx.N, bool)
            inj[calib[0] + off:calib[0] + off + B] = True
            inj_fit = int((inj & ctx.fcal).sum())
            if calib[0] + off + B > calib[1] - REACH:
                raise RuntimeError("el bloque inyectado alcanzaria ventanas de E")
        nat_fit = int((ctx.y & ctx.fcal).sum())
        rec["effective_contamination"] = (nat_fit + inj_fit) / n_fit
        rec["injected_in_F_cal"] = inj_fit
        rec["metrics"] = _t4_eval(ser_c, ctx.T, r4, y4, _thr(p_c), triv4)
        rec["thresholds"] = {x: _thr(p_c)[x] for x in X_T4}
        levels.append(rec)
        if B:
            del ser_c
    return {"applicable": True, "n_fit": n_fit, "pool": [a, b], "excluded_with_guards": [xs, xe], "need": need,
            "E_T4": info4, "E_T4_rows": int(len(r4)), "trivial_E_T4": triv4, "levels": levels,
            "rule": "§8.3: bloque contiguo de B_c filas de calibracion en min(floor((C-B)/2), C-59999-B) sustituido por "
                    "los scores de las primeras B_c filas de ataque del pool; ventanas sobre la serie modificada",
            "tag": TAGS["t4"] + " T4-P-b"}


def section_t4pc(ctx, regions, params, ser, lce, triv):
    N, T = ctx.N, ctx.T
    start20 = T + int(math.floor(0.20 * (N - T)))
    out = {"common_eval": [start20, N], "tag": TAGS["t4"] + " T4-P-c", "fractions": {}}
    ref = (lce.get("warmup_fraction_common_eval") or {}).get("start")
    out["plan_support_start"] = ref
    if ref is not None and int(ref) != start20:
        raise RuntimeError(f"inicio de la evaluacion comun {start20} != label_counts {ref}")
    rows_c = np.arange(start20, N, dtype=np.int64)
    yc = ctx.y[rows_c]
    trivc = trivial_refs(yc)
    out["trivial_common_eval"] = trivc
    for f, cand in FRAC_CANDIDATES.items():
        cal_f, ev_f, info_f = get_regions(regions, cand)
        if not segments_subset([[start20, N]], segments_of(ev_f)):
            raise RuntimeError(f"{cand}: la evaluacion comun no es subconjunto de {ev_f}")
        m = np.zeros(N, bool)
        m[cal_f[0]:cal_f[1]] = True
        m &= ~ctx.dup
        if f == 10:
            p_f, ser_f = params, ser
        else:
            p_f, ser_f = _fit_and_transform(ctx, m, T, cal_f[1])
        n_fit = int(m.sum())
        out["fractions"][str(f)] = {"candidate": cand, "regions": info_f, "n_fit": n_fit,
                                    "effective_contamination": float((ctx.y & m).sum() / n_fit),
                                    "metrics": _t4_eval(ser_f, T, rows_c, yc, _thr(p_f), trivc),
                                    "thresholds": {x: _thr(p_f)[x] for x in X_T4}}
    return out


def _clean(o):
    """NaN/inf -> None (JSON estricto); numpy -> Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _iso_aware(t):
    """ISO -> datetime con zona (las ingenuas son hora local)."""
    d = dt.datetime.fromisoformat(str(t))
    return d if d.tzinfo else d.astimezone()


def read_chronology(out_root):
    """F-10 (MR-12): registro append-only de la cronologia. Devuelve (primer registro | None, entradas del log)."""
    d = Path(out_root) / CHRONO_DIR
    first = C.read_json(d / CHRONO_FIRST) if (d / CHRONO_FIRST).exists() else None
    log = []
    if (d / CHRONO_LOG).exists():
        for ln in (d / CHRONO_LOG).read_text(encoding="utf-8").splitlines():
            try:
                log.append(json.loads(ln))
            except ValueError:
                continue
    return first, log


def record_chronology(out_root, entry):
    """F-10 (MR-12): primer metrics.json con O_EXCL (nunca se reescribe) + una linea por escritura en el log."""
    d = Path(out_root) / CHRONO_DIR
    d.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(str(d / CHRONO_LOG), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        fd = os.open(str(d / CHRONO_FIRST), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        return False
    try:
        os.write(fd, (json.dumps(dict(entry, rule="MR-12: primer metrics.json de esta raiz (O_EXCL; no se reescribe)"),
                                 sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def first_metrics_created(out_root, this_created):
    """min(registro append-only, entradas del log, mtimes de los metrics.json actuales, esta escritura) (F-10)."""
    cands = [this_created]
    first, log = read_chronology(out_root)
    if first and first.get("created"):
        cands.append(str(first["created"]))
    cands += [str(e["created"]) for e in log if e.get("created")]
    for p in Path(out_root).glob("*/metrics.json"):
        try:
            cands.append(dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"))
        except OSError:
            pass
    return min(cands, key=_iso_aware)


class DirLock:
    def __init__(self, d):
        self.d = Path(d)
        self.fd = None

    def __enter__(self):
        self.fd = os.open(str(self.d), os.O_RDONLY)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            return False
        return True

    def __exit__(self, *a):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)


def run_one(job_dir, out_dir, opts, prov, force=False, log=print, out_root_for_first=None):
    job_dir, out_dir = Path(job_dir), Path(out_dir)
    C.check_not_forbidden(out_dir, "--out-dir")
    applicable(C.read_json(job_dir / "config.json") or {}, job_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "metrics.json"
    vkey = version_key(opts)
    if target.exists() and not force:
        old = C.read_json(target) or {}
        op = old.get("provenance") or {}
        oi = op.get("inputs") or {}
        same_inputs = (oi.get("config_sha256") == C.sha256_file(job_dir / "config.json") and
                       oi.get("scores_sha256_arrays_json") ==
                       ((C.read_json(job_dir / "arrays.json") or {}).get("scores.npy") or {}).get("sha256"))
        if op.get("version_key") == vkey and same_inputs and not old.get("section_errors"):
            return "skip", None
    with DirLock(out_dir) as got:
        if not got:
            return "locked", None
        res = compute_job(job_dir, opts, log=log)
        created = C.now_iso()
        res["created"] = created
        pv = dict(prov)
        pv.update(version_key=vkey, options=opts,
                  inputs={"job_dir": str(job_dir), "scores_sha256": C.sha256_file(job_dir / "scores.npy"),
                          "scores_sha256_arrays_json": ((C.read_json(job_dir / "arrays.json") or {}).get("scores.npy") or {}).get("sha256"),
                          "labels_sha256": C.sha256_file(job_dir / "labels.npy"),
                          "config_sha256": C.sha256_file(job_dir / "config.json"),
                          "label_counts_sha256": C.sha256_file(REPO / LABEL_COUNTS_REL),
                          "label_extras_sha256": C.sha256_file(REPO / LABEL_EXTRAS_REL),
                          "aux": "ver aux_inputs"},
                  first_metrics_json_created=first_metrics_created(out_root_for_first or out_dir.parent, created),
                  deviations_proposed=DEVIATIONS_PROPOSED)
        if pv["inputs"]["scores_sha256_arrays_json"] and pv["inputs"]["scores_sha256_arrays_json"] != pv["inputs"]["scores_sha256"]:
            raise RuntimeError("sha256 de scores.npy != arrays.json")
        res["provenance"] = pv
        res = _clean(res)
        C.write_json_atomic(target, res)
        record_chronology(out_root_for_first or out_dir.parent,
                          {"created": created, "job_id": res.get("job_id"), "dataset": (res.get("job") or {}).get("dataset"),
                           "dataset_id": (res.get("job") or {}).get("dataset_id"), "metrics_json": str(target),
                           "family": (res.get("job") or {}).get("family"),
                           "exploratory_arm": bool((res.get("job") or {}).get("exploratory_arm")),
                           "metrics_json_sha256": C.sha256_file(target), "metrics_rev_sha256": pv.get("metrics_rev_sha256")})
    return "ok", res


def base_provenance():
    prov = {"metrics_rev_version": METRICS_REV_VERSION, "metrics_rev_sha256": C.sha256_file(Path(__file__).resolve()),
            "metrics_rev_path": str(Path(__file__).resolve()), "helpers_sha256": helper_hashes(),
            "plan": plan_guard(), "versions": versions(), "env": env_guard(), "host": C.hostname(),
            "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    return prov


def list_done_jobs(results_root, blocks=None):
    out = []
    for d in sorted(Path(results_root).glob("*/DONE")):
        J = d.parent
        if J.name.startswith("."):
            continue
        cfg = C.read_json(J / "config.json") or {}
        row = cfg.get("manifest_row") or {}
        if blocks and not C.block_selected(row.get("block", ""), blocks):
            continue
        out.append((row.get("dataset", ""), J.name, J, cfg))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--job-dir")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--results-root", default=str(C.RESULTS_ROOT))
    ap.add_argument("--out-dir", default=None, help="(--job-dir) escribe metrics.json aqui en vez de en el job")
    ap.add_argument("--out-root", default=None, help="(--all) escribe <out-root>/<job_id>/metrics.json")
    ap.add_argument("--blocks", default=None, help="filtro de bloques (prefijo con guion, como worker.py)")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bootstrap-ci", action="store_true", help="§3.7 IC bootstrap por bloques (exploratorio)")
    ap.add_argument("--o1", action="store_true", help="cota O1 as-submitted (metrics_o1.json); /usr/bin/python3")
    ap.add_argument("--o1-wd-replica", action="store_true", help="O1 con WindowDiff/Pk uint8 replicado (MR-8)")
    ap.add_argument("--o1-min-mem-gib", type=float, default=26.0)
    ap.add_argument("--tmp-dir", default=None)
    a = ap.parse_args(argv)
    for pth in (a.job_dir, a.results_root, a.out_dir, a.out_root):
        if pth is not None and not os.path.isabs(pth):
            print(f"ERROR: rutas absolutas: {pth}")
            return 2
    blocks = [b.strip() for b in a.blocks.split(",")] if a.blocks else None
    if a.o1:
        return main_o1(a, blocks)
    try:
        prov = base_provenance()
    except (PlanGuardError, EnvGuardError) as e:
        print(f"ABORT: {e}")
        return 3
    opts = {"bootstrap_ci": bool(a.bootstrap_ci)}
    if a.job_dir:
        jobs = [(None, Path(a.job_dir).name, Path(a.job_dir), None)]
    else:
        jobs = list_done_jobs(a.results_root, blocks)
    n_ok = n_skip = n_na = n_fail = 0
    t_all = time.time()
    done = 0
    for _ds, name, J, _cfg in jobs:
        if a.max_jobs is not None and done >= a.max_jobs:
            break
        out_dir = Path(a.out_dir) if a.out_dir else (Path(a.out_root) / name if a.out_root else J)
        root_first = Path(a.out_root) if a.out_root else (Path(a.out_dir).parent if a.out_dir else Path(a.results_root))
        t0 = time.time()
        try:
            st, res = run_one(J, out_dir, opts, prov, force=a.force, out_root_for_first=root_first)
        except NotApplicable as e:
            n_na += 1
            print(f"[{C.now_iso()}] NA    {name}: {e}", flush=True)
            continue
        except Exception as e:
            n_fail += 1
            print(f"[{C.now_iso()}] FAIL  {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        if st == "skip":
            n_skip += 1
            print(f"[{C.now_iso()}] SKIP  {name} (metrics.json de esta version)", flush=True)
            continue
        if st == "locked":
            n_skip += 1
            print(f"[{C.now_iso()}] LOCK  {name} (otro proceso)", flush=True)
            continue
        n_ok += 1
        done += 1
        tmm = res.get("timing") or {}
        print(f"[{C.now_iso()}] OK    {name} {time.time() - t0:.1f} s "
              f"(fit {tmm.get('fit_transform_main_s')} s, eval {tmm.get('eval_main_s')} s, t4p {tmm.get('t4p_s')} s; "
              f"errores de seccion {len(res.get('section_errors') or [])}) -> {out_dir / 'metrics.json'}", flush=True)
    print(f"[{C.now_iso()}] FIN ok={n_ok} skip={n_skip} na={n_na} fail={n_fail} en {time.time() - t_all:.1f} s",
          flush=True)
    return 1 if n_fail else 0


def _mem_available_gib():
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def _load_cpr():
    p = HERE / "tests" / "check_postproc_regression.py"
    spec = importlib.util.spec_from_file_location("_jisa_cpr", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, C.sha256_file(p)


def compute_o1(job_dir, wd_replica=False, tmp_dir=None, min_mem_gib=26.0, log=print, _test_any_job=False):
    """O1 de un job T1 con semilla A. _test_any_job (SOLO tests: test_h h12) permite el canario X2 de OS Scan, cuyos
    scores son s0 bit a bit, para cotejar esta ruta con los JSON enviados; nunca se usa en produccion."""
    J = Path(job_dir)
    cfg = C.read_json(J / "config.json") or {}
    if not _test_any_job:
        applicable(cfg, J)
    desc = job_descriptor(cfg)
    if not _test_any_job and (desc["family"] != "T1" or desc["seed"] != "A"):
        raise NotApplicable("O1 solo en la rejilla principal con semilla A (plan §3.6)")
    frame = np.load(J / "scores.npy")
    target = np.load(J / "labels.npy").astype(int)
    N = len(frame)
    if len(target) != N:
        raise RuntimeError("scores y labels con longitudes distintas")
    k_wd = max(min(N // 10, 20_000_000_000 // N), 3)
    need_gib = N * k_wd / 1024 ** 3 * 1.1 + 2
    mem = _mem_available_gib()
    if not wd_replica and (mem is None or mem < max(min_mem_gib, need_gib)):
        raise RuntimeError(f"O1 con torch necesita ~{need_gib:.1f} GiB (N x k uint8) y hay {mem} GiB disponibles; "
                           "espera a tener RAM o usa --o1-wd-replica (MR-8)")
    CPR, cpr_sha = _load_cpr()
    d25 = REPO / "postprocess" / "detectors_2025"
    ok, hashes = CPR.verify_d25(d25)
    if not ok:
        raise RuntimeError(f"detectors_2025 no coincide con su SHA256SUMS: {hashes}")
    E = CPR.load_d25(d25)
    impl = "torch (detectors_2025/src/detectors/metrics.py sin cambios)"
    if wd_replica:
        E.wd_pk = lambda ref, hyp, k=None: wd_pk_uint8(ref, hyp, k)[:2]
        impl = "replica numpy wd_pk_uint8 (MR-8; validada por R-2 frente a los JSON enviados)"
    import torch
    t0 = time.time()
    target_smooth = E.label_smoother(target, CPR.W_LABEL)
    res = {}
    base_tmp = Path(tmp_dir) if tmp_dir else None
    for det in ("KMD", "MVD"):
        td = time.time()
        fn = E.stat_detector if det == "MVD" else E.kmeans_detector
        sm_fn = E.median_smoothing if det == "MVD" else E.mean_smoothing
        d = fn(frame, window_length=CPR.W_DET)
        s = sm_fn(d, window_length=CPR.W_SMOOTH)
        th_raw = E.compute_scores(d, target)[4]
        th_sm = E.compute_scores(s, target_smooth)[4]
        with tempfile.TemporaryDirectory(prefix="jisa_o1_", dir=base_tmp) as tmp:
            m = E.compute_metrics(d, s, target, target_smooth, th_raw, th_sm, tmp, f"o1_{det}")
        res[det] = dict({k: m[k] for k in CPR.KEYS}, seconds=round(time.time() - td, 1))
        log(f"  O1 {det}: {res[det]['seconds']} s")
        del d, s
    sel = "KMD" if res["KMD"]["Window Diff Smooth"] <= res["MVD"]["Window Diff Smooth"] else "MVD"
    import scipy
    import sklearn
    import torch as _t
    out = {"schema": SCHEMA_O1, "job_id": desc["job_id"], "job": desc, "tag": TAGS["oracle1"],
           "label": "as-submitted oracle bound (threshold and detector selected with evaluation labels)",
           "wd_impl": impl, "k_wd": int(k_wd), "N": int(N),
           "protocol": "detectors_2025 sin cambios: MVD stat_detector(frame, 10000) + median_smoothing(50000); KMD "
                       "kmeans_detector + mean_smoothing(50000); label_smoother(10000); umbral compute_scores "
                       "(TNR/TPR, linspace(0, 0.3, 10000)) sobre [0, N); compute_metrics (WD/Pk uint8)",
           "MVD": res["MVD"], "KMD": res["KMD"],
           "selected": {"detector": sel, "F1-Score Smooth": res[sel]["F1-Score Smooth"],
                        "Window Diff Smooth": res[sel]["Window Diff Smooth"],
                        "rule": "min 'Window Diff Smooth'; empate -> KMD (hotplot.ipynb: experiments_kmeans primero, "
                                "idxmin toma el primero)"},
           "notes": (["ACI: la referencia frame[100k:200k) cae en la ejecucion (T = 72.264), declarado (§3.6)"]
                     if desc["dataset"] == "ACI-IOT-2023" else []),
           "provenance": {"metrics_rev_sha256": C.sha256_file(Path(__file__).resolve()),
                          "check_postproc_regression_sha256": cpr_sha, "d25_code": hashes,
                          "plan": plan_guard(), "test_any_job": bool(_test_any_job),
                          "versions": {"python": sys.version.split()[0], "executable": sys.executable,
                                       "numpy": np.__version__, "scipy": scipy.__version__,
                                       "sklearn": sklearn.__version__, "torch": _t.__version__},
                          "inputs": {"scores_sha256": C.sha256_file(J / "scores.npy"),
                                     "labels_sha256": C.sha256_file(J / "labels.npy")},
                          "mem_available_gib_at_start": mem, "seconds": round(time.time() - t0, 1),
                          "deviations_proposed": [DEVIATIONS_PROPOSED[7]]}}
    return _clean(out)


def main_o1(a, blocks):
    try:
        plan_guard()
    except PlanGuardError as e:
        print(f"ABORT: {e}")
        return 3
    if a.job_dir:
        jobs = [(None, Path(a.job_dir).name, Path(a.job_dir), None)]
    else:
        jobs = list_done_jobs(a.results_root, blocks or ["T1-A"])
    n_fail = 0
    for _ds, name, J, _cfg in jobs:
        out_dir = Path(a.out_dir) if a.out_dir else (Path(a.out_root) / name if a.out_root else J)
        target = out_dir / "metrics_o1.json"
        if target.exists() and not a.force:
            print(f"SKIP {name} (metrics_o1.json existe)")
            continue
        t0 = time.time()
        try:
            C.check_not_forbidden(out_dir, "--out-dir")
            out_dir.mkdir(parents=True, exist_ok=True)
            res = compute_o1(J, wd_replica=a.o1_wd_replica, tmp_dir=a.tmp_dir, min_mem_gib=a.o1_min_mem_gib)
            res["created"] = C.now_iso()
            C.write_json_atomic(target, res)
            print(f"[{C.now_iso()}] OK O1 {name} {time.time() - t0:.1f} s -> {target}", flush=True)
        except NotApplicable as e:
            print(f"[{C.now_iso()}] NA O1 {name}: {e}", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[{C.now_iso()}] FAIL O1 {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
