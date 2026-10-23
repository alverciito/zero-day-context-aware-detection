#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baselines_t5.py - lineas base de T5 (analysis_plan.md §7.5, descriptivas, fuera de las familias) en CPU.

Produce SOLO scores por fila, con el mismo contrato de salida que el runner (scores.npy, labels.npy, arrays.json,
config.json, env.json, timing.json, DONE), para que metrics_rev.py aplique EL MISMO post-proceso sin oraculo
(detectores KMD/MVD revisados, fusion, umbral P99 de F_cal, metricas en E). Aqui no se calcula ninguna metrica de
rendimiento y no se miran etiquetas en ningun momento del ajuste: labels.npy se copia de y.npy de la cache tal cual,
igual que hace el runner, y solo lo lee metrics_rev al evaluar.

Variantes (plan §7.5):
  iforest            sklearn.ensemble.IsolationForest(n_estimators=100, max_samples='auto', contamination='auto',
                     max_features=1.0, bootstrap=False, random_state in {1234, 5678, 9012}); ajuste SOLO con las filas
                     [0, T) de la misma matriz de features que recibe el modelo (x.npy de la cache del stream);
                     score = -score_samples (mayor = mas anomalo), calculado para TODAS las filas. 9 datasets x 3
                     semillas.
  kitnet_published   KitNET publicado (github.com/ymirsky/Kitsune-py, MIT), importado SIN modificar de
                     external/Kitsune-py; parametros publicados max_autoencoder_size = 10, learning_rate = 0,1,
                     hidden_ratio = 0,75; graces alineados con [0, fm) y [fm, T) del protocolo; score =
                     log10(1 + RMSE) como en el codigo del envio (kitnet.py:323). Solo los 6 KNAD.

Regiones: las emite run_job.regions_for SIN tocar (mismo contrato de regiones que los jobs T1-T4; metrics_rev las lee
con common.check_protocol_regions). Ninguna region se recalcula aqui.

Diferencias inevitables y decisiones declaradas: DIFFS_DECLARED (abajo); van integras en config.json de cada job.

Uso (entorno del plan, RF-2: envs/postproc; nice y OMP_NUM_THREADS=4):
    R=<raiz del repositorio>
    P="nice -n 19 env OMP_NUM_THREADS=4 $R/envs/postproc/bin/python $R/revision/experiments/baselines_t5.py"
    # coste y plan, sin escribir nada
    $P --plan
    # un job real
    $P --variant iforest --dataset ACI-IOT-2023 --seed A
    # todos (desacoplado)
    nohup $P --variant iforest --dataset ALL --seeds A,B,C > $R/logs/t5_iforest.log 2>&1 &
    nohup $P --variant kitnet_published --dataset KNAD > $R/logs/t5_kitnet.log 2>&1 &
    # humo: stream truncado, fuera de results/revision y SIN DONE (no es un resultado)
    $P --variant kitnet_published --dataset KNAD:OS_Scan --max-rows 260000 --out-root /tmp/t5_smoke --smoke
    # compatibilidad de un job con metrics_rev (estructural: descriptor, regiones, F_cal, auxiliares; sin metricas)
    $P --check-metrics-rev --job-dir $R/results/revision/T5-IF__aci-iot-2023__iforest__corrected.v1__sA__r1

Salida: 0 OK; 1 algun job fallo; 2 uso; 3 guarda de entorno o del plan (no se escribe nada).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import shutil
import socket
import sys
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
KITSUNE_DIR = REPO / "external" / "Kitsune-py"
SCHEMA = "jisa-baselines-t5/1"
BASELINES_T5_VERSION = "1.1.0"
FM_ROWS = 50_000
CHUNK_ROWS = 200_000
KNAD_DATASETS = [f"KNAD:{a}" for a in ("Active_Wiretap", "Fuzzing", "OS_Scan", "SSDP_Flood", "SSL_Renegotiation",
                                       "Video_Injection")]
ALL_DATASETS = KNAD_DATASETS + ["ACI-IOT-2023", "CIC2017", "CIC2018"]
VARIANT_BLOCK = {"iforest": "T5-IF", "kitnet_published": "T5-KITNET"}
VARIANT_DATASETS = {"iforest": ALL_DATASETS, "kitnet_published": KNAD_DATASETS}
IF_PARAMS = {"n_estimators": 100, "max_samples": "auto", "contamination": "auto", "max_features": 1.0,
             "bootstrap": False, "n_jobs": 1}
KITNET_PARAMS = {"max_autoencoder_size": 10, "learning_rate": 0.1, "hidden_ratio": 0.75}

DIFFS_DECLARED = {
    "common": {
        "T5-1 features": "Ninguna linea base re-extrae paquetes: ambas consumen la matriz de features de la cache del "
                         "stream (x.npy), que es EXACTAMENTE la que recibe el modelo del articulo. En KNAD son las "
                         "115 features AfterImage del CSV publicado con el dataset Kitsune (sin preprocesar: KitNET "
                         "normaliza 0-1 por su cuenta); en CIC/ACI son las features ya preprocesadas por "
                         "pipelines.corrected_preprocess (fillna 0, +-inf -> maximo finito de [0, T), z-score con "
                         "media y desviacion de [0, T)), es decir, ajustadas SOLO con el tramo de entrenamiento.",
        "T5-2 protocolo": "Mismas regiones que el runner (run_job.regions_for sin modificar): entrenamiento [0, T) = "
                          "FM [0, fm) + AE [fm, T); calibracion y evaluacion las fija metrics_rev con "
                          "check_protocol_regions sobre la candidata del plan. Ningun objeto de estas lineas base ve "
                          "etiquetas ni filas posteriores a T.",
        "T5-3 post-proceso": "KMD/MVD/fusion/P99/metricas NO se calculan aqui: los aplica metrics_rev.py sobre "
                             "scores.npy, igual que en T1-T4.",
        "T5-4 gate": "metrics_rev.applicable() lists the families whose protocol metrics are computed; T5 is one of "
                     "them.",
        "T5-5 S-INS asimetrica": "La sensibilidad S-INS (calibrar en [fm + ad/3, T), filas in-sample) NO significa lo "
                                 "mismo en T5-IF que en el resto: en T1-T4 y en T5-KitNET esas filas llevan el error "
                                 "prequencial del modelo que se esta entrenando en ese momento, mientras que el "
                                 "Isolation Forest las puntua con el bosque YA AJUSTADO sobre [0, T) completo. La "
                                 "distribucion de calibracion de T5-IF es por construccion mas estrecha y optimista, "
                                 "baja el umbral P99 y cambia el recall. Consecuencia operativa: el S-INS de T5-IF "
                                 "solo vale como diagnostico interno de esa linea base; NO se compara con el S-INS de "
                                 "T1-T4 ni de T5-KitNET, y cualquier tabla que los ponga juntos tiene que decirlo en "
                                 "el pie. Producir un score out-of-fold sobre [fm, T) para T5-IF seria la alternativa, "
                                 "y no esta en el plan §7.5.",
        "T5-6 etiqueta": "Una linea base descriptiva del plan §7.5 no es confirmatoria: esta fuera de H1-H4, de "
                         "S*_d, de R1.4 y del abstract.",
    },
    "iforest": {
        "T5-IF-1 ajuste": "IsolationForest se ajusta con las filas [0, T) completas; max_samples='auto' = min(256, "
                          "n) submuestras por arbol, como en el plan. Sin etiquetas.",
        "T5-IF-2 score": "score = -score_samples(x) para TODAS las filas del stream (plan §7.5.1). No se aplica "
                         "log10(1 + s): esa transformacion pertenece al pipeline de KitNET (kitnet.py:323) y el plan "
                         "no la pide para IF. Las filas [0, T) se puntuan tambien (in-sample, declarado): el "
                         "protocolo principal solo usa [T, N), y la sensibilidad S-INS del plan necesita [fm, T).",
        "T5-IF-3 no finitos": "IsolationForest no admite NaN/inf. Se reproduce el ORDEN exacto de "
                              "pipelines.corrected_preprocess, ajustado solo con [0, T): primero NaN -> 0 y DESPUES "
                              "el maximo finito de la columna en [0, T) (que por tanto incluye esos ceros), que es "
                              "el relleno de +-inf; si tras eso la columna no tiene ningun valor finito en [0, T), "
                              "el relleno es 0.0. El orden importa: con el orden inverso, una columna cuyos unicos "
                              "valores finitos de [0, T) fueran negativos y que ademas tuviera NaN daria un relleno "
                              "negativo en vez de 0.0. Los recuentos van en config.json (nonfinite_input). En los 9 "
                              "streams medidos es un no-op (n_nan = n_inf = 0).",
        "T5-IF-4 determinismo": "n_jobs = 1 y random_state = semilla del plan (A 1234, B 5678, C 9012): el score no "
                                "depende del numero de threads ni del tamano de chunk (se comprueba en --selfcheck).",
        "T5-IF-5 sin ventana": "IF puntua cada fila por separado: no hay ventana L ni paradigma AR/TSR. En "
                               "manifest_row, window y k_clusters valen 'NA' y ar queda vacio.",
    },
    "kitnet_published": {
        "T5-KN-1 codigo": "KitNET/dA/corClust se importan SIN modificar de external/Kitsune-py (MIT, commit fijado en "
                          "config.json). El repositorio queda de solo lectura.",
        "T5-KN-2 features": "El KitNET publicado se alimenta normalmente de su propio extractor AfterImage sobre el "
                            "pcap (FeatureExtractor.py + netStat.py). Aqui se alimenta de las 115 features AfterImage "
                            "del CSV que los autores del dataset publican junto a cada captura, que es la entrada del "
                            "modelo del articulo. No se re-extraen paquetes, asi que cualquier diferencia entre el "
                            "extractor publicado y el CSV publicado queda fuera de esta comparacion.",
        "T5-KN-3 graces": "Los graces publicados del demo (FM 5.000 / AD 50.000) se sustituyen por los del protocolo: "
                          "el mapa de features se entrena con [0, fm) y los AEs con [fm, T). En el codigo publicado "
                          "el corte es inclusivo (train mientras n_trained <= FM+AD), asi que se pasa "
                          "FM_grace_period = fm - 1 y AD_grace_period = T - fm; se comprueba que el mapa se crea "
                          "exactamente en la fila fm - 1 y que la primera fila ejecutada es la T. Sin ese -1, el "
                          "modelo entrenaria ademas con la primera fila de calibracion.",
        "T5-KN-4 normalizacion": "La normalizacion 0-1 de cada dA usa el minimo y el maximo vistos durante el "
                                 "entrenamiento y NO se actualiza al ejecutar (dA.train vs dA.execute): no hay fuga "
                                 "hacia [T, N).",
        "T5-KN-5 score": "score = log10(1 + RMSE) como en el envio (kitnet.py:323-324). Filas [0, fm) = 0.0 (el "
                         "codigo original tampoco produce error en el mapa de features); filas [fm, T) = RMSE de "
                         "entrenamiento de la capa de salida, que es lo que guarda kitnet.py:185.",
        "T5-KN-6 captura del RMSE de entrenamiento": "KitNET.train() descarta el RMSE de entrenamiento y process() "
                                                     "devuelve 0.0 durante todo el entrenamiento. Para guardar las "
                                                     "filas [fm, T) (necesarias para la sensibilidad S-INS) se "
                                                     "reproduce literalmente la rama de entrenamiento de "
                                                     "KitNET.train() llamando a los mismos objetos publicados "
                                                     "(ensembleLayer[a].train, outputLayer.train) y se recoge el "
                                                     "valor devuelto. El estado resultante es identico: "
                                                     "--selfcheck compara los scores de ejecucion con los de "
                                                     "process() sin tocar.",
        "T5-KN-7 ensemble": "El tamano del ensemble lo decide el clustering por correlacion del propio KitNET "
                            "(corClust con maxClust = 10): en KNAD salen ~15 AEs, no los 4 del articulo. Es la "
                            "configuracion publicada, y esa es la diferencia que la linea base mide.",
        "T5-KN-8 determinismo": "Sin semillas: los pesos iniciales de dA usan RandomState(1234) fijo dentro del "
                                "codigo publicado y no hay otra fuente de azar. Se ejecuta una sola vez; la etiqueta "
                                "de semilla del job_id es 'A' por convenio del runner y no significa replica.",
        "T5-KN-9 alcance": "Solo los 6 KNAD (plan §7.5.2). En CIC/ACI las filas son flujos y las features no son "
                           "AfterImage: KitNET publicado no es comparable ahi.",
    },
}

DEV_ROWS_PROPOSED = [
    {"id": "T5-N1", "que": "metrics_rev.applicable() admite tambien la familia T5 (lineas base del plan §7.5)",
     "detalle": "The T5 jobs meet the contract of metrics_rev.py (scores.npy float64 (N,), labels.npy int8, "
                "config.json with the regions of run_job.regions_for, cache with x/y and their sha256). A "
                "descriptive baseline of plan §7.5 is not confirmatory.",
     "efecto": "ninguno sobre los numeros ya calculados si el recalculo se hace en un lote"},
    {"id": "T5-N2", "que": "KitNET publicado: FM_grace_period = fm - 1 (49.999) en vez de fm",
     "detalle": "El corte del codigo publicado es inclusivo; con fm el modelo entrenaria con 50.001/200.001 filas e "
                "invadiria la primera fila de calibracion. Con fm - 1 las regiones coinciden EXACTAMENTE con "
                "[0, fm) y [fm, T) del runner.",
     "efecto": "elimina una fila de entrenamiento fuera de [0, T); sin efecto sobre la comparabilidad"},
    {"id": "T5-N3", "que": "KitNET publicado: se recoge el RMSE de entrenamiento reproduciendo la rama de "
                              "KitNET.train() (T5-KN-6)",
     "detalle": "process() devuelve 0.0 durante el entrenamiento y dejaria [fm, T) constante, con lo que la "
                "sensibilidad S-INS del plan seria degenerada. La equivalencia del estado se comprueba con "
                "--selfcheck (scores de ejecucion identicos bit a bit frente a process()).",
     "efecto": "ninguno sobre [T, N)"},
    {"id": "T5-N4", "que": "IsolationForest se ajusta sobre la matriz de features de la cache tal como la recibe "
                              "el modelo (KNAD: AfterImage sin z-score; CIC/ACI: z-score de [0, T))",
     "detalle": "El plan §7.5.1 dice 'features preprocesadas de [0, T)'. La unica matriz preprocesada que existe es "
                "la del pipeline corregido, y en KNAD el pipeline no preprocesa nada porque KitNET normaliza por su "
                "cuenta. IF es invariante a reescalados afines por columna salvo redondeo, asi que la eleccion no "
                "cambia el ranking; se declara.",
     "efecto": "ninguno esperado; declarado por transparencia"},
    {"id": "T5-N5", "que": "external/Kitsune-py clonado en el arbol principal (no versionado)",
     "detalle": "commit 7c8ea51e89268d9a20399dca4bf32d663e638eb4 (24 mar 2026), licencia MIT; el directorio se deja "
                "de solo lectura y solo se importa. El clon incluye mirai.zip (7,8 MB) y 'Kitsune paper.pdf' (4,4 "
                "MB), que no estan cubiertos por la licencia MIT del codigo y no se usan. La licencia MIT se cita en "
                "el manuscrito al reportar la linea base.",
     "efecto": "ninguno sobre resultados"},
    {"id": "T5-N6", "que": "S-INS de T5-IF no es comparable con el de T1-T4 ni con el de T5-KitNET",
     "detalle": "En T1-T4 y en T5-KitNET las filas [fm + ad/3, T) llevan el error prequencial (el modelo se esta "
                "entrenando en ese momento); en T5-IF las puntua el bosque ya ajustado sobre [0, T), asi que su "
                "distribucion de calibracion es sistematicamente mas estrecha, baja el umbral P99 y cambia el "
                "recall. Declarado en differences_declared['common']['T5-5 S-INS asimetrica'].",
     "efecto": "el S-INS de T5-IF se reporta solo como diagnostico interno de esa linea base; ninguna tabla "
               "compara S-INS entre T5-IF y las demas familias sin decir esto en el pie"},
    {"id": "T5-N7", "que": "Regla de no finitos de T5-IF reordenada para ser bit a bit la del pipeline",
     "detalle": "`nonfinite_fill_from_train` calculaba el maximo finito sobre la columna cruda; "
                "pipelines.corrected_preprocess sustituye NaN por 0 ANTES de ese maximo (pipelines.py:484-491). En "
                "una columna con NaN cuyos unicos valores finitos de [0, T) fueran negativos, el relleno de +-inf "
                "difiere (0.0 frente al maximo negativo). Corregido en baselines_t5 1.1.0.",
     "efecto": "no-op en los 5 jobs T5-IF ya escritos (n_nan = n_inf = 0 en su config.json): sus scores no "
               "cambian y no hay que re-ejecutarlos"},
]


class BaselineError(RuntimeError):
    pass


def log_line(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_peak_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def guards(skip_plan=False):
    """RF-2 (entorno del lock) y R-7 (plan congelado), reutilizando las mismas funciones que metrics_rev."""
    import metrics_rev as MR
    env = MR.env_guard(REPO)
    plan = None if skip_plan else MR.plan_guard(REPO)
    return {"env": env, "plan": plan, "versions": MR.versions(),
            "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                           "MKL_NUM_THREADS")}}


class SpecShim:
    """Lo minimo que run_job.regions_for y run_job._fit_set_exclusion leen de una fila del manifiesto."""

    def __init__(self, dataset, pipeline="corrected", pipeline_version="v1"):
        self.dataset = dataset
        self.pipeline = pipeline
        self.pipeline_version = pipeline_version
        if str(dataset).upper().startswith("KNAD:"):
            self.kind = "knad"
            self.attack = dataset.split(":", 1)[1]
            self.attack_dir = C.KNAD_ATTACKS.get(self.attack)
        else:
            self.kind = {"CIC2017": "cic2017", "CIC2018": "cic2018", "ACI-IOT-2023": "aci"}.get(dataset)
            self.attack = self.attack_dir = None
        if self.kind is None:
            raise BaselineError(f"dataset desconocido: {dataset!r}")


def cache_dir(dataset):
    return C.CACHE_ROOT / f"{C.ds_slug(dataset)}__corrected__v1"


def reference_fm_ad(dataset, results_root, required=True):
    """fm y ad del dataset, cotejados con un job T1 **DONE** del mismo dataset (T5-3).

    El corte de entrenamiento de la linea base no puede salir de un job abortado, a medio escribir o de otro
    runner: se exige `DONE` y se prefiere el bloque T1-A (el confirmatorio completo en los 9 datasets). Si no hay
    ninguno se aborta, salvo `required=False` (humo), donde se avisa por stderr y se usan las constantes.
    """
    T = int(C.corrected_train_rows(dataset))
    fm, ad, ref = FM_ROWS, T - FM_ROWS, None
    slug = C.ds_slug(dataset)
    roots, seen = [], set()
    for r in (Path(results_root), Path(C.RESULTS_ROOT)):
        if r.is_dir() and str(r.resolve()) not in seen:
            seen.add(str(r.resolve()))
            roots.append(r)
    cands = []
    for r in roots:
        cands += [d for d in r.glob(f"T1-*__{slug}__*") if (d / "DONE").exists()]
    cands.sort(key=lambda d: (0 if d.name.startswith("T1-A") else 1, d.name))
    for d in cands:
        hp = (C.read_json(d / "config.json") or {}).get("hyperparameters_frozen") or {}
        if hp.get("fm") and hp.get("ad"):
            ref = {"job": d.name, "fm": int(hp["fm"]), "ad": int(hp["ad"]), "done": True}
            break
    if ref is None:
        msg = (f"{dataset}: no hay ningun job T1 con DONE del que cotejar fm/ad en "
               f"{', '.join(str(r) for r in roots)}")
        if required:
            raise BaselineError(msg + " (usa --allow-missing-reference para aceptar las constantes)")
        print(f"AVISO: {msg}; se usan las constantes fm={fm}, ad={ad}", file=sys.stderr)
    if ref and (ref["fm"], ref["ad"]) != (fm, ad):
        raise BaselineError(f"{dataset}: fm/ad {fm}/{ad} != los del job de referencia {ref}")
    return fm, ad, {"fm": fm, "ad": ad, "T": T, "source": "common.CORRECTED_TRAIN_ROWS + fm congelado (50.000)",
                    "reference_job": ref, "checked_against_reference": ref is not None,
                    "rule": "job T1 con DONE del mismo dataset, T1-A preferido (T5-3)"}


def cache_meta_public(meta, p, verify_sha):
    """Misma vista de la cache que entrega pipelines._verify_and_load al runner (rutas absolutas en 'aux'): es la que
    espera run_job.regions_for para apuntar fit_set_exclusion a aux_dup_train.npy."""
    keys = ("key", "pipeline", "pipeline_version", "semantics", "builder", "test_mode", "data_root", "sources",
            "x", "y", "created", "extra")
    pub = {k: meta[k] for k in keys if k in meta}
    pub.update(cached=True, path=str(p), y_path=str(p / "y.npy"), y_sha256=(meta.get("y") or {}).get("sha256"))
    aux = {}
    for name, a in (meta.get("aux") or {}).items():
        f = p / a["file"]
        if not f.is_file():
            raise BaselineError(f"cache {p}: falta {a['file']}")
        if verify_sha and C.sha256_file(f) != a["sha256"]:
            raise BaselineError(f"cache {p}: sha256 de {a['file']} no coincide")
        aux[name] = dict(a, path=str(f))
    if aux:
        pub["aux"] = aux
    return pub


def load_cache(dataset, verify_sha=True, max_rows=None):
    """x (mmap, solo lectura), y (int8), meta (vista publica) y la ficha de la cache que va a config.json."""
    p = cache_dir(dataset)
    if not (p / "DONE").exists():
        raise BaselineError(f"cache sin DONE: {p}")
    meta = C.read_json(p / "meta.json") or {}
    x = np.load(p / "x.npy", mmap_mode="r")
    y = np.load(p / "y.npy")
    if len(x) != len(y):
        raise BaselineError(f"{p}: len(x) {len(x)} != len(y) {len(y)}")
    if int(meta.get("expected_rows", len(y))) != len(y):
        raise BaselineError(f"{p}: expected_rows {meta.get('expected_rows')} != {len(y)}")
    info = {"key": meta.get("key"), "path": str(p), "pipeline": meta.get("pipeline"),
            "pipeline_version": meta.get("pipeline_version"), "semantics": meta.get("semantics"),
            "x": meta.get("x"), "y": meta.get("y"), "sha256_verified": False,
            "read_only": "la cache es de solo lectura; baselines_t5.py no escribe nada en ella"}
    t0 = time.time()
    if verify_sha:
        for name in ("x.npy", "y.npy"):
            exp = (meta.get(name.split(".")[0]) or {}).get("sha256")
            got = C.sha256_file(p / name)
            if exp and got != exp:
                raise BaselineError(f"{p / name}: sha256 {got} != meta.json {exp}")
        info["sha256_verified"] = True
    pub = cache_meta_public(meta, p, verify_sha)
    info["sha256_verify_seconds"] = round(time.time() - t0, 1)
    info["aux"] = {k: {"path": v["path"], "sha256": v.get("sha256")} for k, v in (pub.get("aux") or {}).items()}
    if max_rows:
        x, y = x[:max_rows], y[:max_rows]
        info["truncated_to_rows"] = int(max_rows)
    return x, y.astype(np.int8), pub, p, info


def build_regions(dataset, n, meta, fm, ad):
    """Regiones IDENTICAS a las de un job T1: run_job.regions_for sin modificar."""
    import run_job as RJ
    spec = SpecShim(dataset)
    reg = RJ.regions_for(spec, int(n), {"fm": int(fm), "ad": int(ad)}, meta)
    for k in reg.get("detector_protocol_candidates", {}):
        if not k.startswith("_"):
            C.check_protocol_regions(reg, k, with_exclusions=True)
    return reg


def job_id_for(variant, dataset, seed):
    return f"{VARIANT_BLOCK[variant]}__{C.ds_slug(dataset)}__{variant}__corrected.v1__s{seed}__r1"


def manifest_row_for(variant, dataset, seed, job_id):
    """Fila 'como si' del manifiesto: la leen metrics_rev.job_descriptor y aggregate. window/k_clusters = NA porque
    estas lineas base no tienen ventana ni numero de AEs fijado por el manifiesto."""
    return {"job_id": job_id, "block": VARIANT_BLOCK[variant], "priority": "NA", "dataset": dataset,
            "variant": variant, "ar": "", "paradigm": "NA", "pipeline": "corrected", "pipeline_version": "v1",
            "window": "NA", "k_clusters": "NA", "grouping": "NA", "partition_seed": "NA", "contamination": "0",
            "seed": seed, "rep": "1", "save_ckpt": "False", "timing_ref": "False",
            "notes": "linea base T5 (analysis_plan.md §7.5): NO esta en manifest.csv; la produce baselines_t5.py"}


def nonfinite_fill_from_train(xtr):
    """Valores de relleno por columna a partir de [0, T), con el ORDEN de pipelines.corrected_preprocess:
    primero NaN -> 0 y despues el maximo finito de la columna (0.0 si tras eso no queda ninguno).

    pipelines.py:484-491 hace `col[m] = 0.0` (NaN) ANTES de `inf_fill[j] = tr[fin].max()`, asi que esos ceros
    entran en el maximo. Calcular el maximo sobre la columna cruda daria otro relleno en una columna cuyos unicos
    valores finitos de [0, T) fueran negativos y que ademas tuviera NaN (T5-5).
    """
    fill = np.zeros(xtr.shape[1], np.float64)
    cols_sin_finito = []
    for j in range(xtr.shape[1]):
        col = xtr[:, j]
        col = np.where(np.isnan(col), 0.0, col)
        fin = np.isfinite(col)
        if fin.any():
            fill[j] = float(col[fin].max())
        else:
            cols_sin_finito.append(int(j))
    return fill, cols_sin_finito


def sanitize(block, fill, counters):
    """NaN -> 0; +-inf -> fill de la columna. Devuelve el bloque saneado (copia) y actualiza los recuentos."""
    nan = np.isnan(block)
    inf = np.isinf(block)
    if not nan.any() and not inf.any():
        return block
    out = np.array(block, dtype=np.float64, copy=True)
    counters["n_nan"] += int(nan.sum())
    counters["n_inf"] += int(inf.sum())
    out[nan] = 0.0
    if inf.any():
        idx = np.nonzero(inf)
        out[idx] = fill[idx[1]]
    return out


def iter_chunks(x, start, stop, chunk=CHUNK_ROWS):
    for a in range(start, stop, chunk):
        b = min(a + chunk, stop)
        yield a, b, np.ascontiguousarray(x[a:b], dtype=np.float64)


def run_iforest(x, T, seed_int, log=log_line):
    from sklearn.ensemble import IsolationForest
    n = len(x)
    phases = {}
    c_fit = {"n_nan": 0, "n_inf": 0}
    c_all = {"n_nan": 0, "n_inf": 0}
    t0 = time.time()
    xtr = np.ascontiguousarray(x[:T], dtype=np.float64)
    phases["read_train_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    fill, cols_sin_finito = nonfinite_fill_from_train(xtr)
    xtr = sanitize(xtr, fill, c_fit)
    phases["sanitize_train_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    clf = IsolationForest(random_state=int(seed_int), **IF_PARAMS)
    clf.fit(xtr)
    phases["fit_s"] = round(time.time() - t0, 2)
    log(f"  IF ajustado con [0, {T}) en {phases['fit_s']} s (max_samples_ = {clf.max_samples_})")
    del xtr
    scores = np.empty(n, np.float64)
    t0 = time.time()
    for a, b, blk in iter_chunks(x, 0, n):
        scores[a:b] = -clf.score_samples(sanitize(blk, fill, c_all))
    phases["score_s"] = round(time.time() - t0, 2)
    model = {"estimator": "sklearn.ensemble.IsolationForest", "params": dict(IF_PARAMS, random_state=int(seed_int)),
             "max_samples_": int(clf.max_samples_), "n_features_in_": int(clf.n_features_in_),
             "offset_": float(clf.offset_), "fit_rows": [0, int(T)],
             "score": "-score_samples(x) (mayor = mas anomalo; plan §7.5.1)",
             "chunk_rows": CHUNK_ROWS,
             "nonfinite_input": {"rule": "NaN -> 0; +-inf -> maximo finito de la columna en [0, T) (0.0 si no hay "
                                         "ninguno), como pipelines.corrected_preprocess",
                                 "fit_rows_0_T": dict(c_fit), "scored_rows_0_N": dict(c_all),
                                 "columns_without_finite_train_values": cols_sin_finito,
                                 "no_op": c_all["n_nan"] == 0 and c_all["n_inf"] == 0}}
    return scores, model, phases


def import_published_kitnet():
    if not (KITSUNE_DIR / "KitNET" / "KitNET.py").exists():
        raise BaselineError(f"falta el KitNET publicado en {KITSUNE_DIR} (git clone https://github.com/ymirsky/"
                            "Kitsune-py.git)")
    if str(KITSUNE_DIR) not in sys.path:
        sys.path.insert(0, str(KITSUNE_DIR))
    import KitNET.KitNET as kit
    if not C.is_under(kit.__file__, KITSUNE_DIR):
        raise BaselineError(f"KitNET importado desde {kit.__file__}, no desde {KITSUNE_DIR}")
    rc, head, _ = C.git(KITSUNE_DIR, "rev-parse", "HEAD")
    prov = {"repo": "https://github.com/ymirsky/Kitsune-py", "license": "MIT", "path": str(KITSUNE_DIR),
            "commit": head if rc == 0 else None,
            "files_sha256": {f: C.sha256_file(KITSUNE_DIR / "KitNET" / f)
                             for f in ("KitNET.py", "dA.py", "corClust.py", "utils.py")},
            "modified": False, "import": "sin modificar (solo se importa)"}
    return kit, prov


def _train_step_capture(kn, row):
    """Training branch of KitNET.train() reproduced to collect the RMSE it discards; the model state stays identical."""
    s_l1 = np.zeros(len(kn.ensembleLayer))
    for a in range(len(kn.ensembleLayer)):
        s_l1[a] = kn.ensembleLayer[a].train(row[kn.v[a]])
    rmse = kn.outputLayer.train(s_l1)
    kn.n_trained += 1
    return float(rmse)


def run_kitnet(x, T, fm, log=log_line, pure_process=False):
    kit, prov = import_published_kitnet()
    n, d = len(x), x.shape[1]
    phases = {}
    kn = kit.KitNET(d, max_autoencoder_size=KITNET_PARAMS["max_autoencoder_size"],
                    FM_grace_period=fm - 1, AD_grace_period=T - fm,
                    learning_rate=KITNET_PARAMS["learning_rate"], hidden_ratio=KITNET_PARAMS["hidden_ratio"])
    rmse = np.zeros(n, np.float64)
    t0 = time.time()
    for a, b, blk in iter_chunks(x, 0, min(fm, n)):
        for i in range(b - a):
            kn.train(blk[i])
    phases["fm_s"] = round(time.time() - t0, 2)
    if kn.v is None:
        raise BaselineError(f"el mapa de features no se creo en la fila {fm - 1} (n_trained = {kn.n_trained})")
    if kn.n_trained != min(fm, n):
        raise BaselineError(f"tras el mapa de features n_trained = {kn.n_trained} != fm = {fm}")
    fmap = [list(map(int, v)) for v in kn.v]
    log(f"  KitNET: mapa de {d} features en {len(fmap)} AEs (tamanos {[len(v) for v in fmap]}) en {phases['fm_s']} s")
    t0 = time.time()
    for a, b, blk in iter_chunks(x, min(fm, n), min(T, n)):
        for i in range(b - a):
            rmse[a + i] = 0.0 if pure_process else _train_step_capture(kn, blk[i])
            if pure_process:
                kn.train(blk[i])
    phases["train_s"] = round(time.time() - t0, 2)
    if kn.n_trained != min(T, n):
        raise BaselineError(f"tras el entrenamiento n_trained = {kn.n_trained} != T = {T}")
    t0 = time.time()
    for a, b, blk in iter_chunks(x, min(T, n), n):
        for i in range(b - a):
            rmse[a + i] = kn.execute(blk[i])
    phases["exec_s"] = round(time.time() - t0, 2)
    if kn.n_executed != max(n - T, 0):
        raise BaselineError(f"n_executed = {kn.n_executed} != N - T = {n - T}")
    scores = np.log10(rmse + 1.0)
    model = {"implementation": "KitNET publicado (Kitsune-py), sin modificar", "provenance": prov,
             "params": dict(KITNET_PARAMS, FM_grace_period=int(fm - 1), AD_grace_period=int(T - fm),
                            n_features=int(d)),
             "grace_alignment": {"fm_rows": [0, int(fm)], "ae_train_rows": [int(fm), int(T)],
                                 "first_executed_row": int(T), "n_executed": int(kn.n_executed),
                                 "note": DIFFS_DECLARED["kitnet_published"]["T5-KN-3 graces"]},
             "feature_map": {"n_autoencoders": len(fmap), "sizes": [len(v) for v in fmap], "map": fmap,
                             "source": "corClust del codigo publicado (maxClust = 10) sobre [0, fm)"},
             "score": "log10(1 + RMSE) (kitnet.py:323-324); filas [0, fm) = 0.0",
             "train_rmse_capture": ("process() puro: [fm, T) = 0.0" if pure_process
                                    else DIFFS_DECLARED["kitnet_published"]["T5-KN-6 captura del RMSE de entrenamiento"]),
             "rmse_summary": {"train_max": float(rmse[fm:T].max()) if T > fm else None,
                              "exec_max": float(rmse[T:].max()) if n > T else None}}
    return scores, model, phases


def write_job(out_dir, job_id, variant, dataset, seed, scores, labels, model, phases, wall_s, regions, cache_info,
              fmad, prov, smoke=False, extra=None, yes_delete=False):
    out_dir = Path(out_dir)
    C.check_not_forbidden(out_dir, "--out-root")
    tmp_root = out_dir.parent / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = tmp_root / f"{out_dir.name}.{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n = len(scores)
    np.save(tmp / "scores.npy", np.asarray(scores, np.float64))
    np.save(tmp / "labels.npy", np.asarray(labels, np.int8))
    arrays = {"scores.npy": {"path": "scores.npy", "shape": [n], "dtype": "float64",
                             "sha256": C.sha256_file(tmp / "scores.npy"),
                             "sha256_array": sha256_array(np.asarray(scores, np.float64)),
                             "n_nan": int(np.isnan(scores).sum()), "n_inf": int(np.isinf(scores).sum())},
              "labels.npy": {"path": "labels.npy", "shape": [n], "dtype": "int8",
                             "sha256": C.sha256_file(tmp / "labels.npy"),
                             "sha256_array": sha256_array(np.asarray(labels, np.int8)),
                             "values": sorted(set(np.unique(labels).tolist())),
                             "alignment": "y[i] = etiqueta de la fila i del stream (copiada de y.npy de la cache)"}}
    C.write_json_atomic(tmp / "arrays.json", arrays)
    cfg = {"schema": SCHEMA, "baselines_t5_version": BASELINES_T5_VERSION, "job_id": job_id,
           "manifest_row": manifest_row_for(variant, dataset, seed, job_id),
           "manifest_path": "NA (linea base T5: no esta en manifest.csv)",
           "spec": {"job_id": job_id, "block": VARIANT_BLOCK[variant], "family": "T5", "dataset": dataset,
                    "kind": SpecShim(dataset).kind, "variant": variant, "pipeline": "corrected",
                    "pipeline_version": "v1", "seed": seed, "seed_int": C.SEEDS.get(seed), "rep": 1,
                    "mode": "smoke" if smoke else "full"},
           "mode": "smoke" if smoke else "full", "test_mode": bool(smoke), "smoke": bool(smoke),
           "seed_label": seed, "seed_int": C.SEEDS.get(seed),
           "code": {"script": str(Path(__file__).resolve()), "sha256": C.sha256_file(Path(__file__).resolve()),
                    "common_sha256": C.sha256_file(HERE / "common.py"),
                    "run_job_sha256": C.sha256_file(HERE / "run_job.py"),
                    "main_repo": str(REPO), "head": C.head_commit(REPO), "tags_at_head": C.tags_at_head(REPO),
                    "runner_tag": C.runner_tag_at_head(REPO),
                    "note": "las regiones vienen de run_job.regions_for sin modificar; el modelo NO es el del "
                            "articulo: es una linea base de T5"},
           "results_root": str(out_dir.parent), "started": prov.get("started"), "host": socket.gethostname(),
           "pid": os.getpid(), "device": "/CPU:0",
           "hyperparameters_frozen": {"fm": int(fmad[0]), "ad": int(fmad[1]), "L": None, "k": None,
                                      "source": fmad[2]},
           "model": model, "cache": cache_info, "regions": regions,
           "score_row_semantics": "score[i] corresponde a la fila i del stream; ninguna linea base de T5 usa ventana",
           "scores_summary": {"n": n, "n_nan": arrays["scores.npy"]["n_nan"],
                              "n_inf": arrays["scores.npy"]["n_inf"], "nan_is_result": True},
           "labels_note": "labels.npy se copia de y.npy de la cache SIN usarla: ningun objeto de este job se ajusta "
                          "con etiquetas (las lee solo metrics_rev al evaluar)",
           "plan": {"section": "analysis_plan.md §7.5", "role": "linea base descriptiva, fuera de las familias y de "
                                                                "H1-H4", "tag": "[SENSIBILIDAD plan §7.5]"},
           "differences_declared": {"common": DIFFS_DECLARED["common"], variant: DIFFS_DECLARED[variant]},
           "deviations_proposed": DEV_ROWS_PROPOSED,
           "provenance": dict(prov, versions=prov.get("versions")),
           "outputs": ["arrays.json", "config.json", "DONE", "env.json", "labels.npy", "scores.npy", "timing.json"]}
    if extra:
        cfg.update(extra)
    C.write_json_atomic(tmp / "config.json", cfg)
    C.write_json_atomic(tmp / "env.json", {"versions": prov.get("versions"), "threads_env": prov.get("threads_env"),
                                           "env_lock": prov.get("env"), "host": socket.gethostname(),
                                           "device": "/CPU:0", "python": sys.executable,
                                           "nice": os.nice(0), "cpu_count": os.cpu_count()})
    timing = {"wall_seconds": round(wall_s, 2), "host": socket.gethostname(), "device": "/CPU:0", "phases": phases,
              "rows": n, "rows_per_s": round(n / wall_s, 1) if wall_s > 0 else None,
              "rss_peak_bytes": rss_peak_bytes(), "timing_ref": 0,
              "note": "CPU con nice -n 19 y 4 threads, con otros trabajos en la maquina: NO es una medida de coste "
                      "para R1.3 (plan §9 exige exclusividad)"}
    C.write_json_atomic(tmp / "timing.json", timing)
    if not smoke:
        (tmp / "DONE").write_text(C.now_iso() + "\n", encoding="utf-8")
    C.fsync_tree(tmp)
    with C.host_claim(out_dir.parent, timeout_s=600.0) as got:
        if not got:
            shutil.rmtree(tmp, ignore_errors=True)
            raise BaselineError(f"{out_dir}: no se pudo obtener el flock de host en 600 s (hay un worker "
                                f"escribiendo); no se toca nada")
        if out_dir.exists():
            prev = (C.read_json(out_dir / "config.json") or {}).get("schema")
            if prev != SCHEMA or not out_dir.name.startswith("T5-"):
                shutil.rmtree(tmp, ignore_errors=True)
                raise BaselineError(f"{out_dir} existe y no es un job T5 de baselines_t5.py (schema {prev!r}): "
                                    f"no se toca")
            if not (smoke or yes_delete):
                shutil.rmtree(tmp, ignore_errors=True)
                raise BaselineError(f"{out_dir} ya existe: --force solo sustituye con --yes-delete (el job anterior "
                                    "se conserva como .trash-...)")
            trash = out_dir.parent / f".trash-{out_dir.name}.{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
            out_dir.rename(trash)
            print(f"AVISO: job anterior apartado en {trash} (no se ha borrado)", file=sys.stderr)
        tmp.rename(out_dir)
        C.fsync_dir(out_dir.parent)
    return timing


def run_job(variant, dataset, seed, results_root, max_rows=None, smoke=False, force=False, verify_sha=True,
            pure_process=False, allow_missing_reference=False, yes_delete=False, log=log_line):
    job_id = job_id_for(variant, dataset, seed)
    out_dir = Path(results_root) / job_id
    if (out_dir / "DONE").exists() and not force:
        log(f"SKIP {job_id}: ya tiene DONE")
        return "skip", None
    t_all = time.time()
    prov = guards(skip_plan=smoke)
    prov["started"] = C.now_iso()
    log(f"== {job_id}")
    x, y, meta, cpath, cache_info = load_cache(dataset, verify_sha=verify_sha, max_rows=max_rows)
    fm, ad, fmad_info = reference_fm_ad(dataset, results_root,
                                        required=not (smoke or allow_missing_reference))
    T = fm + ad
    n = len(y)
    if n <= T:
        raise BaselineError(f"{dataset}: {n} filas <= T = {T}; con --max-rows usa al menos T + algo")
    if smoke:
        regions = {"units": "indices de fila del stream (0-based, semiabiertos)", "n_rows": int(n),
                   "fm": [0, fm], "ae_train": [fm, T], "execution": [T, n], "train": [0, T], "post_train": [T, n],
                   "smoke": "stream TRUNCADO: no hay regiones de evaluacion validas ni candidatas (no es un resultado)"}
    else:
        regions = build_regions(dataset, n, meta, fm, ad)
    log(f"  stream {n} filas x {x.shape[1]} features; T = {T} (fm {fm} + ad {ad})")
    if variant == "iforest":
        scores, model, phases = run_iforest(x, T, C.SEEDS[seed], log=log)
    elif variant == "kitnet_published":
        scores, model, phases = run_kitnet(x, T, fm, log=log, pure_process=pure_process)
    else:
        raise BaselineError(f"variante desconocida: {variant}")
    wall = time.time() - t_all
    timing = write_job(out_dir, job_id, variant, dataset, seed, scores, y, model, phases, wall, regions, cache_info,
                       (fm, ad, fmad_info), prov, smoke=smoke, yes_delete=yes_delete)
    log(f"  OK {job_id}: {wall:.1f} s ({timing['rows_per_s']} filas/s), fases {phases} -> {out_dir}")
    return "ok", out_dir


def check_metrics_rev(job_dir, log=log_line):
    """Structural compatibility of a T5 job with metrics_rev.py (descriptor, regions, F_cal, auxiliary arrays, formats); it computes no metric."""
    import metrics_rev as MR
    J = Path(job_dir)
    cfg = C.read_json(J / "config.json") or {}
    out = {"job_dir": str(J), "job_id": cfg.get("job_id"), "checks": {}, "blocking": []}
    try:
        MR.applicable(cfg, J)
        out["checks"]["applicable"] = "OK (la guarda ya admite T5)"
    except MR.NotApplicable as e:
        out["checks"]["applicable"] = f"NotApplicable: {e}"
        out["blocking"].append("metrics_rev.applicable() no admite la familia T5")
    desc = MR.job_descriptor(cfg)
    out["checks"]["job_descriptor"] = {k: desc[k] for k in ("job_id", "block", "family", "dataset", "dataset_id",
                                                            "dataset_key", "corpus", "variant", "config_id", "seed")}
    s = np.load(J / "scores.npy", mmap_mode="r")
    y = np.load(J / "labels.npy", mmap_mode="r")
    N = len(s)
    out["checks"]["arrays"] = {"n": N, "scores_dtype": str(s.dtype), "labels_dtype": str(y.dtype),
                               "scores_1d_float64": s.dtype == np.float64 and s.ndim == 1,
                               "labels_ok": y.shape == (N,) and set(np.unique(y).tolist()) <= {0, 1},
                               "n_nan": int(np.isnan(s).sum()), "n_inf": int(np.isinf(s).sum())}
    regions = cfg["regions"]
    if int(regions.get("n_rows", -1)) != N:
        out["blocking"].append(f"regions.n_rows {regions.get('n_rows')} != len(scores) {N}")
    cands = {}
    for k in regions.get("detector_protocol_candidates", {}):
        if k.startswith("_"):
            continue
        try:
            calib, ev = C.check_protocol_regions(regions, k, with_exclusions=True)
            cands[k] = {"calib": list(calib), "eval": ev}
        except C.ProtocolRegionError as e:
            cands[k] = f"ProtocolRegionError: {e}"
            if k == MR.PRIMARY_CANDIDATE:
                out["blocking"].append(f"candidata primaria invalida: {e}")
    out["checks"]["candidates"] = cands
    T = int(regions["train"][1])
    t0 = time.time()
    dup, fsinfo = MR.fit_set_mask(cfg, N, T, log=lambda *_: None)
    out["checks"]["fit_set_mask"] = {"n_marked": int(dup.sum()), "source": fsinfo.get("mask_source"),
                                     "seconds": round(time.time() - t0, 1)}
    aux = MR.load_aux(cfg, N, T, np.asarray(y), desc["dataset"])
    out["checks"]["load_aux"] = {"ts": aux["ts"] is not None, "day": aux["day"] is not None,
                                 "types": aux["types"] is not None, "info": list((aux.get("info") or {}).keys())}
    lc = (C.read_json(REPO / MR.LABEL_COUNTS_REL) or {}).get(desc["dataset_key"]) or {}
    calib, ev, _ = MR.get_regions(regions, MR.PRIMARY_CANDIDATE)
    ref = (lc.get("calib") or {})
    out["checks"]["label_counts_regions"] = {
        "calib_job": list(calib), "calib_plan_support": ref.get("rows") or ref.get("calib"),
        "eval_job": ev, "note": "metrics_rev vuelve a comprobarlo con sus comprobaciones duras"}
    out["compatible"] = not out["blocking"]
    log(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return out


def selfcheck(dataset="KNAD:OS_Scan", rows=None, log=log_line):
    """Cheap consistency checks on a short stretch of a stream; no job is written."""
    from sklearn.ensemble import IsolationForest
    res = {}
    x, y, meta, p, _ = load_cache(dataset, verify_sha=False)
    fm_s, T_s = 2_000, 8_000
    n_s = rows or 12_000
    X = np.ascontiguousarray(x[:n_s], dtype=np.float64)
    clf = IsolationForest(random_state=1234, **IF_PARAMS).fit(X[:T_s])
    a = -clf.score_samples(X)
    b = np.concatenate([-clf.score_samples(X[i:i + 997]) for i in range(0, n_s, 997)])
    clf4 = IsolationForest(random_state=1234, **dict(IF_PARAMS, n_jobs=4)).fit(X[:T_s])
    c = -clf4.score_samples(X)
    res["s1_if_chunk_and_threads"] = {"chunking_identical": bool(np.array_equal(a, b)),
                                      "n_jobs_max_abs_diff": float(np.abs(a - c).max())}
    Xp = X.copy()
    Xp[T_s:] = Xp[T_s:] * 1e3 + 7.0
    clfp = IsolationForest(random_state=1234, **IF_PARAMS).fit(Xp[:T_s])
    res["s2_if_no_lookahead"] = {"scores_0_T_identical":
                                 bool(np.array_equal(a[:T_s], (-clfp.score_samples(Xp[:T_s]))))}
    s_cap, m_cap, _ = run_kitnet(X, T_s, fm_s, log=lambda *_: None, pure_process=False)
    s_pur, _, _ = run_kitnet(X, T_s, fm_s, log=lambda *_: None, pure_process=True)
    res["s3_kitnet_state_identical"] = {
        "exec_scores_bitwise_equal": bool(np.array_equal(s_cap[T_s:], s_pur[T_s:])),
        "train_rows_zero_in_pure": bool(np.array_equal(s_pur[fm_s:T_s], np.zeros(T_s - fm_s))),
        "train_rows_nonzero_in_capture": bool((s_cap[fm_s:T_s] > 0).any())}
    res["s4_kitnet_regions"] = {"fm_rows_zero": bool(np.array_equal(s_cap[:fm_s], np.zeros(fm_s))),
                                "n_autoencoders": m_cap["feature_map"]["n_autoencoders"],
                                "first_executed_row": m_cap["grace_alignment"]["first_executed_row"],
                                "n_executed": m_cap["grace_alignment"]["n_executed"],
                                "expected_n_executed": n_s - T_s}
    res["ok"] = (res["s1_if_chunk_and_threads"]["chunking_identical"]
                 and res["s2_if_no_lookahead"]["scores_0_T_identical"]
                 and res["s3_kitnet_state_identical"]["exec_scores_bitwise_equal"]
                 and res["s4_kitnet_regions"]["n_executed"] == res["s4_kitnet_regions"]["expected_n_executed"])
    log(json.dumps(res, indent=2, ensure_ascii=False))
    return res


RATES = {"iforest": {"score_rows_per_s": 3.0e5, "read_bytes_per_s": 1.0e9},
         "kitnet_published": {"train_rows_per_s": 3.07e3, "exec_rows_per_s": 6.09e3, "read_bytes_per_s": 1.0e9}}


def cost_plan(results_root, log=log_line):
    rows = []
    for variant in ("iforest", "kitnet_published"):
        for ds in VARIANT_DATASETS[variant]:
            p = cache_dir(ds)
            meta = C.read_json(p / "meta.json") or {}
            n = int(meta.get("expected_rows") or 0)
            T = int(C.corrected_train_rows(ds))
            xb = (p / "x.npy").stat().st_size if (p / "x.npy").exists() else 0
            r = RATES[variant]
            done = sum(1 for sd in ("A", "B", "C")
                       if (Path(results_root) / job_id_for(variant, ds, sd) / "DONE").exists())
            if variant == "iforest":
                s = (xb * (1 + T / max(n, 1))) / r["read_bytes_per_s"] + (n + T) / r["score_rows_per_s"]
                seeds = 3
            else:
                s = xb / r["read_bytes_per_s"] + T / r["train_rows_per_s"] + max(n - T, 0) / r["exec_rows_per_s"]
                seeds = 1
            rows.append({"variant": variant, "dataset": ds, "n_rows": n, "T": T,
                         "seconds_per_seed": round(s), "seeds": seeds, "done": done,
                         "seconds_pending": round(s * (seeds - done)), "seconds_total": round(s * seeds),
                         "gb_x": round(xb / 2**30, 2)})
    tot, pend = {}, {}
    for r in rows:
        tot[r["variant"]] = tot.get(r["variant"], 0) + r["seconds_total"]
        pend[r["variant"]] = pend.get(r["variant"], 0) + r["seconds_pending"]
    out = {"note": "estimacion con las tasas medidas el 24 sep en esta maquina (nice -n 19, 4 threads, con otros "
                   "trabajos activos); solo CPU, sin GPU", "rates": RATES, "per_job": rows,
           "total_seconds_by_variant": tot, "pending_seconds_by_variant": pend,
           "total_hours": round(sum(tot.values()) / 3600.0, 2),
           "pending_hours": round(sum(pend.values()) / 3600.0, 2),
           "disk_bytes_scores_labels": sum(r["n_rows"] * 9 * r["seeds"] for r in rows)}
    log(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANT_BLOCK))
    ap.add_argument("--dataset", help="nombre exacto (KNAD:OS_Scan, CIC2017, ...), 'KNAD' o 'ALL'")
    ap.add_argument("--seed", default="A", choices=sorted(C.SEEDS))
    ap.add_argument("--seeds", default=None, help="lista separada por comas (iforest); por defecto, --seed")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--out-root", default=None, help="alias de --results-root (humo fuera de results/revision)")
    ap.add_argument("--max-rows", type=int, default=None, help="trunca el stream (solo con --smoke)")
    ap.add_argument("--smoke", action="store_true", help="prueba: no escribe DONE y salta la guarda del plan")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-verify-sha", action="store_true", help="no recalcula el sha256 de x.npy/y.npy de la cache")
    ap.add_argument("--kitnet-pure-process", action="store_true", help="KitNET con process() puro ([fm, T) = 0.0)")
    ap.add_argument("--check-metrics-rev", action="store_true")
    ap.add_argument("--job-dir", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--plan", action="store_true", help="estimacion de coste; no escribe nada")
    ap.add_argument("--allow-missing-reference", action="store_true",
                    help="acepta fm/ad de las constantes si no hay ningun job T1 con DONE del dataset (T5-3)")
    ap.add_argument("--yes-delete", action="store_true",
                    help="con --force: autoriza mover el job anterior a .trash-<job>.<ts> sin preguntar")
    a = ap.parse_args(argv)
    results_root = Path(a.out_root or a.results_root or C.RESULTS_ROOT)

    try:
        if a.plan:
            cost_plan(results_root)
            return 0
        if a.check_metrics_rev:
            if not a.job_dir:
                ap.error("--check-metrics-rev necesita --job-dir")
            return 0 if check_metrics_rev(a.job_dir)["compatible"] else 1
        if a.selfcheck:
            guards(skip_plan=True)
            return 0 if selfcheck()["ok"] else 1
        if not a.variant or not a.dataset:
            ap.error("hacen falta --variant y --dataset (o --plan / --selfcheck / --check-metrics-rev)")
        if a.max_rows and not a.smoke:
            ap.error("--max-rows solo con --smoke (un stream truncado no es un resultado)")
        if a.dataset == "ALL":
            datasets = VARIANT_DATASETS[a.variant]
        elif a.dataset == "KNAD":
            datasets = KNAD_DATASETS
        else:
            datasets = [a.dataset]
        bad = [d for d in datasets if d not in VARIANT_DATASETS[a.variant]]
        if bad:
            ap.error(f"{a.variant}: {bad} fuera del alcance del plan §7.5 ({VARIANT_DATASETS[a.variant]})")
        seeds = [s.strip() for s in a.seeds.split(",")] if a.seeds else [a.seed]
        if a.variant == "kitnet_published" and len(seeds) > 1:
            ap.error("kitnet_published es determinista: una sola ejecucion (T5-KN-8)")
    except SystemExit:
        raise
    except (C.EnvError, BaselineError) as e:
        print(f"GUARDA: {e}", file=sys.stderr)
        return 3

    n_fail = 0
    for ds in datasets:
        for sd in seeds:
            try:
                run_job(a.variant, ds, sd, results_root, max_rows=a.max_rows, smoke=a.smoke, force=a.force,
                        verify_sha=not a.no_verify_sha, pure_process=a.kitnet_pure_process,
                        allow_missing_reference=a.allow_missing_reference, yes_delete=a.yes_delete)
            except Exception:
                n_fail += 1
                log_line(f"FALLO {a.variant} {ds} s{sd}:\n{traceback.format_exc()}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
