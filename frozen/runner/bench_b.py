#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_b.py - cost benchmark of the analysis plan (§9.4): latency, throughput, memory and parameter counts per architecture."""
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common as C

REPO = C.MAIN_REPO
PY_TF = REPO / "envs" / "tf215" / "bin" / "python"
PY_POSTPROC = REPO / "envs" / "postproc" / "bin" / "python"
OUT_DEFAULT = C.RESULTS_ROOT / "bench"
OUT_DRYRUN = C.RESULTS_ROOT / "bench" / "dryrun"

DATASET = "KNAD:Active_Wiretap"
JOB_GLOB = "T1-A-KNAD__knad-active_wiretap__{variant}__ar{ar}__*__sA__r1"
FM_SIZES = [82, 15, 8, 10]
L_WINDOW = 800
HIDDEN_RATIO = 0.22
LR = 0.002
EXECUTION_BS = 512
ELM_GRACE = 10_000
ARCHS = ["stat", "original", "mlp", "mlp_ctx", "lstm", "conv2d", "mha", "conv1d"]
WINDOWED = {"conv1d", "conv2d", "lstm", "mha", "mlp_ctx"}
KERAS = {"conv1d", "conv2d", "lstm", "mha", "mlp", "mlp_ctx"}
NUMPY_ARCHS = {"original", "stat"}
AR_OF = {a: "T" for a in ARCHS}

PLAN_TAG = {"graph_b1": "[CONFIRMATORIO plan §9.4]", "eager_b1": "[SENSIBILIDAD plan §9.4]",
            "eager_b512": "[SENSIBILIDAD plan §9.4]", "predict_b512": "[CONFIRMATORIO plan §9.4]",
            "numpy_b1": "[CONFIRMATORIO plan §9.4]", "numpy_b512": "[CONFIRMATORIO plan §9.4]",
            "postproc": "[CONFIRMATORIO plan §9.4]", "params": "[CONFIRMATORIO plan §9.4]"}

CONFIGS = ("gpu", "cpu1", "cpucnt", "postproc")


def n_hidden_of(n_visible, hidden_ratio=HIDDEN_RATIO):
    """n_hidden = ceil(hidden_ratio * n_visible) (multilayer.py:31 y equivalentes)."""
    import math
    return int(math.ceil(hidden_ratio * n_visible))


def macs_per_call(arch, b, l, nv, nh):
    """MAC del pase hacia delante de UN AE, contados sobre la definicion de src/models/networks/*.py."""
    dec = b * (nh * nh + nh * nv) if arch in WINDOWED else b * nh * nv
    if arch == "stat":
        return b * nv
    if arch == "original":
        return b * (nv * nh + nh * nv)
    if arch == "mlp":
        return b * nv * nh + b * nh * nv
    if arch == "mlp_ctx":
        return b * (l * nv) * nh + b * nh * nv
    if arch == "conv1d":
        k = max(1, l // 4)
        c1 = b * l * nv * k * nv
        c2 = b * (l // 2) * nh * k * nv
        return c1 + c2 + b * nh * nh + dec
    if arch == "conv2d":
        c1 = b * l * nv * 32 * 9
        c2 = b * (l // 2) * ((nv + 1) // 2) * nh * 9 * 32
        return c1 + c2 + b * nh * nh + dec
    if arch == "lstm":
        return b * l * 4 * (nv * nh + nh * nh) + dec
    if arch == "mha":
        proj = 4 * b * l * nv * (nh * nv)
        att = 2 * b * nh * l * l * nv
        return proj + att + b * l * nv * nv + b * nv * nh + dec
    raise ValueError(arch)


PREDICT_INNER_BS = 32


def effective_batch(mode, batch):
    """Batch that actually reaches the graph: model.predict re-splits into batches of 32, the other modes pass it whole."""
    return min(batch, PREDICT_INNER_BS) if mode == "predict" else batch


def activation_bytes(arch, b, l, nv, nh, itemsize=4):
    """Bytes de la activacion intermedia mayor (fp32). Cota gruesa para decidir si la celda cabe."""
    win = b * l * nv * itemsize
    if arch in ("stat", "original", "mlp"):
        return b * nv * itemsize * 4
    if arch == "mlp_ctx":
        return win * 2
    if arch == "conv1d":
        return b * l * max(1, l // 4) * nv * itemsize + win * 3
    if arch == "conv2d":
        return b * l * nv * 32 * itemsize * 2 + b * l * nv * 9 * itemsize + win
    if arch == "lstm":
        return win * 2
    if arch == "mha":
        return b * nh * l * l * itemsize + b * l * nh * nv * itemsize * 3 + win
    raise ValueError(arch)


def time_calls(fn, warmup, iters, budget_s):
    """Ejecuta fn() warmup veces y luego hasta `iters` medidas o hasta agotar `budget_s`.

    fn DEBE forzar la sincronizacion (`.numpy()`); si no, en GPU se medirian encolados.
    :return: dict con latencias en segundos y el estado del presupuesto.
    """
    t_w0 = time.perf_counter()
    fn()
    first_s = time.perf_counter() - t_w0
    extra = warmup - 1 if first_s * warmup < 0.4 * budget_s else max(0, int(0.4 * budget_s / max(first_s, 1e-9)) - 1)
    for _ in range(max(0, min(warmup - 1, extra))):
        fn()
    warm_s = time.perf_counter() - t_w0
    warmup = 1 + max(0, min(warmup - 1, extra))
    lat = []
    t0 = time.perf_counter()
    budget_hit = False
    while len(lat) < iters:
        a = time.perf_counter()
        fn()
        lat.append(time.perf_counter() - a)
        if time.perf_counter() - t0 > budget_s:
            budget_hit = len(lat) < iters
            break
    return {"latencies_s": lat, "n_measured": len(lat), "n_requested": iters, "warmup": warmup,
            "warmup_s": warm_s, "measure_s": time.perf_counter() - t0, "budget_hit": budget_hit}


def summarize(tc, batch):
    """p50/p99 de latencia y de throughput a partir de las latencias crudas."""
    import statistics
    lat = sorted(tc["latencies_s"])
    if not lat:
        return {"n_measured": 0, "error": "sin medidas"}

    def q(p):
        if len(lat) == 1:
            return lat[0]
        h = (len(lat) - 1) * p
        lo, hi = int(h), min(int(h) + 1, len(lat) - 1)
        return lat[lo] + (h - lo) * (lat[hi] - lat[lo])

    p50, p99 = q(0.50), q(0.99)
    out = {"n_measured": tc["n_measured"], "n_requested": tc["n_requested"], "warmup": tc["warmup"],
           "budget_hit": tc["budget_hit"], "warmup_s": round(tc["warmup_s"], 6),
           "measure_s": round(tc["measure_s"], 6), "batch": batch,
           "latency_s": {"p50": p50, "p99": p99, "min": lat[0], "max": lat[-1],
                         "mean": statistics.fmean(lat),
                         "stdev": (statistics.stdev(lat) if len(lat) > 1 else 0.0)},
           "latency_per_window_s": {"p50": p50 / batch, "p99": p99 / batch},
           "throughput_windows_per_s": {"p50": batch / p50 if p50 > 0 else None,
                                        "p99": batch / p99 if p99 > 0 else None}}
    return out


def rss_peak_bytes():
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def rss_now_bytes():
    try:
        for ln in Path("/proc/self/status").read_text().splitlines():
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) * 1024
    except OSError:
        pass
    return None


def mem_available_bytes():
    """MemAvailable de /proc/meminfo: lo que hay AHORA, con los jobs confirmatorios que esten corriendo."""
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) * 1024
    except OSError:
        pass
    return None


def gpu_free_bytes():
    """memory.total - memory.used de la GPU 0 (MiB -> bytes). None si no hay nvidia-smi."""
    import shutil
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits",
                            "-i", "0"], capture_output=True, text=True, timeout=30)
        tot, used = [float(x) for x in p.stdout.strip().splitlines()[0].split(",")]
        return int((tot - used) * (1 << 20))
    except Exception:
        return None


def nvidia_smi():
    """Estado de la GPU (plan §9.4: `nvidia-smi` antes y despues)."""
    import shutil
    if shutil.which("nvidia-smi") is None:
        return {"error": "nvidia-smi no disponible"}
    out = {"at": C.now_iso()}
    for key, q in (("gpu", "index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu"),):
        try:
            p = subprocess.run(["nvidia-smi", f"--query-{key}={q}", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30)
            out[key] = p.stdout.strip().splitlines()
        except Exception as e:
            out[key] = f"{type(e).__name__}: {e}"
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        out["compute_apps"] = p.stdout.strip().splitlines()
    except Exception as e:
        out["compute_apps"] = f"{type(e).__name__}: {e}"
    return out


def find_job(arch, results_root=None):
    """Job T1-A de Active Wiretap, semilla A, paradigma AR, de esta arquitectura. None si no existe o no esta DONE."""
    root = Path(results_root or C.RESULTS_ROOT)
    for d in sorted(root.glob(JOB_GLOB.format(variant=arch, ar=AR_OF.get(arch, "T")))):
        if (d / "DONE").exists():
            return d
    return None


def checkpoint_info(arch, results_root=None):
    """Que pesos entrenados hay para esta arquitectura y de donde salen (procedencia del JSON)."""
    job = find_job(arch, results_root)
    if job is None:
        return {"available": False, "reason": "sin job T1-A DONE de esta arquitectura", "job_id": None}
    if arch in KERAS:
        aes = [job / "ckpt" / f"ae{i}" / "model" for i in range(len(FM_SIZES))]
        ok = all(p.exists() for p in aes)
        return {"available": ok, "job_id": job.name, "kind": "keras_savedmodel",
                "paths": [str(p) for p in aes], "reason": None if ok else "faltan SavedModel en ckpt/ae*/model"}
    if arch == "original":
        p = job / "ckpt" / "kitnet_model.pkl"
        return {"available": p.exists(), "job_id": job.name, "kind": "kitnet_pickle", "paths": [str(p)],
                "reason": None if p.exists() else "falta ckpt/kitnet_model.pkl"}
    p = job / "model_stats.npz"
    return {"available": p.exists(), "job_id": job.name, "kind": "model_stats_npz", "paths": [str(p)],
            "reason": None if p.exists() else "falta model_stats.npz"}


def ae_classes(code_root):
    """Autoencoder classes per architecture; mlp_ctx comes from networks_ctx.py without mutating NETWORK_TYPES."""
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from src.models.oopsie import NETWORK_TYPES
    import networks_ctx
    cls = dict(NETWORK_TYPES)
    cls.update(networks_ctx.CONTEXT_NETWORK_TYPES)
    return cls


def build_random_ae(arch, nv, classes):
    """AE nuevo con los mismos argumentos que kitnet.__build_daes__ (kitnet.py:295-303)."""
    ae = classes[arch](n_visible=nv, sequence_length=L_WINDOW, lr=LR, hidden_ratio=HIDDEN_RATIO,
                       haste=False, ar=True, name=f"bench {arch} [{nv}]")
    prepare_numpy_ae(arch, ae, nv)
    return ae


def prepare_numpy_ae(arch, ae, nv):
    """`original` y `stat` arrancan con norm_min/max degenerados (+-inf o ceros): sin esto `execute` daria NaN y el
    tiempo no seria el del camino real. Solo afecta a las medidas con pesos ALEATORIOS."""
    import numpy as np
    if arch == "original":
        if not np.isfinite(np.asarray(ae.norm_min)).all():
            ae.norm_min = np.zeros(nv)
            ae.norm_max = np.ones(nv)
        ae.n = max(int(getattr(ae, "n", 0)), ELM_GRACE)
    elif arch == "stat":
        ae.norm_min = np.zeros(nv)
        ae.norm_max = np.ones(nv)
        ae.mean = np.zeros(nv)
        ae.std = np.ones(nv)
    elif arch in KERAS:
        ae.norm_min = 0.0
        ae.norm_max = 1.0


def load_trained(arch, classes, info, log=print):
    """Pesos entrenados de la semilla A de Active Wiretap. Devuelve (lista de objetos por AE, nota)."""
    import numpy as np
    if not info.get("available"):
        return None, info.get("reason") or "sin checkpoint"
    if arch in KERAS:
        import tensorflow as tf
        models = []
        for p in info["paths"]:
            models.append(tf.keras.models.load_model(p, compile=False))
        return models, None
    if arch == "original":
        import pickle
        with open(info["paths"][0], "rb") as f:
            kn = pickle.load(f)
        aes = list(kn.autoencoders)
        if len(aes) != len(FM_SIZES):
            return None, f"el checkpoint trae {len(aes)} AEs y no {len(FM_SIZES)}"
        for ae in aes:
            ae.n = max(int(getattr(ae, "n", 0)), ELM_GRACE)
        return aes, None
    z = np.load(info["paths"][0])
    aes = []
    for i, nv in enumerate(FM_SIZES):
        ae = classes["stat"](n_visible=nv)
        ae.norm_min = np.asarray(z[f"ae{i}_norm_min"], dtype=np.float64)
        ae.norm_max = np.asarray(z[f"ae{i}_norm_max"], dtype=np.float64)
        ae.mean = np.asarray(z[f"ae{i}_mean"], dtype=np.float64)
        ae.std = np.asarray(z[f"ae{i}_std"], dtype=np.float64)
        aes.append(ae)
    return aes, None


def count_params(arch, obj, nv):
    """Parametros por AE (plan §9.4 punto 6). Keras: count_params(); numpy: recuento explicito."""
    import numpy as np
    if arch in KERAS:
        model = obj if hasattr(obj, "count_params") else obj.model
        det = []
        try:
            det = [{"name": l.name, "class": type(l).__name__, "params": int(l.count_params())}
                   for l in model.layers]
        except Exception:
            det = []
        return {"params": int(model.count_params()), "layers": det, "source": "keras count_params()"}
    if arch == "original":
        w = np.asarray(obj.W)
        n = int(w.size + np.asarray(obj.h_bias).size + np.asarray(obj.v_bias).size)
        return {"params": n, "source": "W (%dx%d) + h_bias + v_bias; W_prime es la transpuesta (pesos atados)"
                                       % w.shape, "layers": []}
    n = 4 * int(nv)
    return {"params": n, "source": "estado (mean, std, norm_min, norm_max); StatisticalAnomaly no entrena pesos",
            "trainable": 0, "layers": []}


def setup_device_env(cfg, threads):
    """Variables que TF lee ANTES de crear los pools. Hay que llamarla antes de importar tensorflow."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["LD_LIBRARY_PATH"] = ""
    if cfg == "gpu":
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(max(1, min(2, threads)))
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)


def probe_gpu(tf):
    """Check that relu and sign really execute on /GPU:0; the GPU row of the benchmark is valid only then (plan §9.4)."""
    logical = [d.name for d in tf.config.list_logical_devices("GPU")]
    if not any(n.endswith("GPU:0") for n in logical):
        return {"ok": False, "reason": f"TensorFlow no ve /GPU:0 (logical={logical})", "logical": logical}
    try:
        with tf.device("/GPU:0"):
            t = tf.constant([-1.5, 0.0, 2.0], dtype=tf.float32)
            r_t, s_t = tf.nn.relu(t), tf.sign(t)
        r, s = [float(v) for v in r_t.numpy()], [float(v) for v in s_t.numpy()]
        devs = [str(getattr(r_t, "device", "")), str(getattr(s_t, "device", ""))]
    except Exception as e:
        return {"ok": False, "reason": f"kernels GPU: {type(e).__name__}: {str(e)[:400]}", "logical": logical}
    if r != [0.0, 0.0, 2.0] or s != [-1.0, 0.0, 1.0]:
        return {"ok": False, "reason": f"relu/sign devuelven {r}/{s}", "logical": logical}
    if not all(d.upper().endswith("GPU:0") for d in devs):
        return {"ok": False, "reason": f"colocacion blanda a CPU (dispositivos {devs})", "logical": logical}
    return {"ok": True, "reason": None, "logical": logical, "devices": devs}


def calibrate_flops(device, cfg, n=1024, reps=3):
    """GFLOP/s del dispositivo con un matmul fp32. Solo alimenta al estimador de coste (no va a la tabla)."""
    import numpy as np
    best = None
    if cfg == "postproc":
        a = np.random.default_rng(0).random((n, n), dtype=np.float32)
        for _ in range(reps):
            t0 = time.perf_counter()
            (a @ a).sum()
            best = min(best or 1e9, time.perf_counter() - t0)
    else:
        import tensorflow as tf
        with tf.device(device):
            a = tf.random.uniform((n, n), dtype=tf.float32)
            for _ in range(reps):
                t0 = time.perf_counter()
                (a @ a).numpy()
                best = min(best or 1e9, time.perf_counter() - t0)
    return {"matmul_n": n, "best_s": best, "flops_per_s": (2.0 * n ** 3) / best if best else None}


def make_input(arch, batch, nv, rng, tf=None, device=None):
    """Entrada de un pase: (B, L, nv) con ventana, (B, nv) sin ella. Valores en [0, 1] (tras la min-max del AE)."""
    import numpy as np
    shape = (batch, L_WINDOW, nv) if arch in WINDOWED else (batch, nv)
    x = rng.random(shape, dtype=np.float32)
    if tf is not None:
        with tf.device(device):
            return tf.constant(x)
    return x.astype(np.float64) if arch in NUMPY_ARCHS else x


def cell_key(mode, batch):
    return f"{mode}_b{batch}"


def cell_mem_cap(cfg, declared_gib, opts):
    """Memory cap of one benchmark cell, in bytes; on CPU it is recomputed from MemAvailable for every cell."""
    cap = declared_gib * (1 << 30)
    if cfg != "gpu":
        avail = mem_available_bytes()
        if avail:
            cap = min(cap, opts["mem_frac"] * avail)
    return cap


def measure_keras_cell(model, x, mode, batch, opts, tf, device):
    """Una celda Keras: eager `model(x)`, `graph` (tf.function) o `predict` (camino implementado)."""
    if mode == "eager":
        def fn():
            model(x, training=False).numpy()
    elif mode == "graph":
        f = tf.function(lambda t: model(t, training=False),
                        input_signature=[tf.TensorSpec(x.shape, tf.float32)])

        def fn():
            f(x).numpy()
    elif mode == "predict":
        import numpy as np
        xn = np.asarray(x)

        def fn():
            model.predict(xn, verbose=0)
    else:
        raise ValueError(mode)
    with tf.device(device):
        return time_calls(fn, opts["warmup"], opts["iters"], opts["cell_budget_s"])


def measure_numpy_cell(ae, x, opts):
    def fn():
        ae.execute(x)
    return time_calls(fn, opts["warmup"], opts["iters"], opts["cell_budget_s"])


def gpu_mem_reset(tf, cfg):
    if cfg != "gpu":
        return None
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        return True
    except Exception:
        return False


def gpu_mem_peak(tf, cfg):
    if cfg != "gpu":
        return None
    try:
        return int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except Exception:
        return None


def run_device_config(cfg, opts, log=print):
    """Mide todas las celdas de una configuracion de dispositivo. Se ejecuta en un proceso propio."""
    threads = 1 if cfg == "cpu1" else opts["cpu_threads"]
    setup_device_env(cfg, threads)
    import numpy as np
    import tensorflow as tf
    import logging as _logging
    tf.get_logger().setLevel(_logging.ERROR)
    if cfg == "cpu1" or cfg == "cpucnt":
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(max(1, min(2, threads)))
        except RuntimeError as e:
            log(f"[bench_b] aviso: no se pudo fijar intra/inter_op ({e}); se usan las variables de entorno")
    res = {"config": cfg, "requested_threads": threads, "started": C.now_iso(),
           "nvidia_smi_before": nvidia_smi(), "device": None, "gpu_probe": None, "cells": [], "params": {},
           "weights": {}, "notes": []}
    if cfg == "gpu":
        pr = probe_gpu(tf)
        res["gpu_probe"] = pr
        if not pr["ok"]:
            res["skipped"] = True
            res["skip_reason"] = pr["reason"]
            res["notes"].append("plan §9.4: la fila GPU se omite y se declara la causa")
            res["nvidia_smi_after"] = nvidia_smi()
            res["finished"] = C.now_iso()
            return res
        device = "/GPU:0"
    else:
        device = "/CPU:0"
    res["device"] = device
    try:
        import keras as _keras
        keras_v = _keras.__version__
    except Exception:
        keras_v = getattr(tf.keras, "__version__", None)
    res["tf"] = {"version": tf.__version__, "keras": keras_v,
                 "intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
                 "inter_op": tf.config.threading.get_inter_op_parallelism_threads(),
                 "env": {k: os.environ.get(k) for k in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                                                        "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
    res["calibration"] = calibrate_flops(device, cfg)
    flops = res["calibration"]["flops_per_s"] or 1e9
    mem_cap = opts["gpu_mem_cap_gib"] if cfg == "gpu" else opts["mem_cap_gib"]
    if cfg == "gpu":
        free = gpu_free_bytes()
        if free:
            mem_cap = min(mem_cap, opts["mem_frac"] * free / (1 << 30))
    res["mem_cap_gib"] = mem_cap
    res["mem_safety_factor"] = opts["mem_safety_factor"]
    res["mem_available_at_start_gib"] = (mem_available_bytes() or 0) / (1 << 30)
    if opts.get("mem_rlimit_gib"):
        import resource
        lim = int(opts["mem_rlimit_gib"] * (1 << 30))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            res["rlimit_as_gib"] = opts["mem_rlimit_gib"]
        except (ValueError, OSError) as e:
            res["rlimit_as_gib"] = f"no aplicado: {e}"
    classes = ae_classes(opts["code_root"])
    rng = np.random.default_rng(20261001)
    t_global0 = time.perf_counter()
    global_budget_s = opts["global_budget_h"] * 3600.0

    for arch in opts["archs"]:
        info = checkpoint_info(arch, opts["results_root"])
        res["weights"][arch] = info
        sources = []
        if info.get("available"):
            sources.append("trained")
        sources.append("random")
        if arch in NUMPY_ARCHS and cfg == "gpu":
            res["notes"].append(f"{arch}: implementado en numpy (src/models/networks/"
                                f"{'original_kitsune' if arch == 'original' else 'stdev'}.py); no hay fila GPU")
            continue
        trained_objs = None
        if "trained" in sources:
            try:
                trained_objs, err = load_trained(arch, classes, info, log)
                if trained_objs is None:
                    res["weights"][arch]["load_error"] = err
                    sources.remove("trained")
            except Exception as e:
                res["weights"][arch]["load_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                sources.remove("trained")
        random_objs = [build_random_ae(arch, nv, classes) for nv in FM_SIZES]
        res["params"][arch] = []
        for i, nv in enumerate(FM_SIZES):
            obj = (trained_objs[i] if trained_objs is not None else random_objs[i])
            o = obj if arch not in KERAS else (obj if hasattr(obj, "count_params") else obj.model)
            res["params"][arch].append(dict(count_params(arch, o, nv), ae=i, n_visible=nv,
                                            n_hidden=n_hidden_of(nv),
                                            weights="trained" if trained_objs is not None else "random",
                                            plan=PLAN_TAG["params"]))

        for source in sources:
            objs = trained_objs if source == "trained" else random_objs
            iters = opts["iters"] if source == "trained" or "trained" not in sources else opts["compare_iters"]
            for batch in (1, EXECUTION_BS):
                modes = (["numpy"] if arch in NUMPY_ARCHS else
                         (["eager", "graph"] if batch == 1 else ["eager", "predict"]))
                for mode in modes:
                    if time.perf_counter() - t_global0 > global_budget_s:
                        res["notes"].append("presupuesto global agotado: celdas restantes sin medir")
                        res["global_budget_hit"] = True
                        res["nvidia_smi_after"] = nvidia_smi()
                        res["finished"] = C.now_iso()
                        return res
                    for i, nv in enumerate(FM_SIZES):
                        nh = n_hidden_of(nv)
                        macs = macs_per_call(arch, batch, L_WINDOW, nv, nh)
                        abytes = activation_bytes(arch, effective_batch(mode, batch), L_WINDOW, nv, nh)
                        est_s = 2.0 * macs / flops
                        base = {"config": cfg, "device": device, "arch": arch, "ae": i, "n_visible": nv,
                                "n_hidden": nh, "batch": batch, "mode": mode, "weights": source,
                                "plan": PLAN_TAG.get(cell_key(mode, batch), "[EXPLORATORIO]"),
                                "est_macs": macs, "est_activation_bytes": abytes, "est_call_s": est_s}
                        need = abytes * opts["mem_safety_factor"]
                        base["est_bytes_with_safety"] = need
                        cap_now = cell_mem_cap(cfg, mem_cap, opts)
                        base["mem_cap_bytes_now"] = cap_now
                        if need > cap_now:
                            res["cells"].append(dict(base, skipped="est_too_big"))
                            continue
                        if est_s > opts["max_call_s"]:
                            res["cells"].append(dict(base, skipped="est_too_slow"))
                            continue
                        warm = opts["warmup"]
                        if est_s > 0 and est_s * warm > 0.5 * opts["cell_budget_s"]:
                            warm = max(1, int(0.5 * opts["cell_budget_s"] / est_s))
                        cell_opts = dict(opts, iters=iters, warmup=warm)
                        rss0 = rss_now_bytes()
                        gpu_mem_reset(tf, cfg)
                        try:
                            if arch in NUMPY_ARCHS:
                                x = make_input(arch, batch, nv, rng)
                                tc = measure_numpy_cell(objs[i], x, cell_opts)
                            else:
                                model = objs[i] if hasattr(objs[i], "predict") else objs[i].model
                                x = make_input(arch, batch, nv, rng, tf=tf, device=device)
                                tc = measure_keras_cell(model, x, mode, batch, cell_opts, tf, device)
                        except Exception as e:
                            kind = "oom" if C.looks_like_oom(e) else "error"
                            res["cells"].append(dict(base, skipped=kind,
                                                     error=f"{type(e).__name__}: {str(e)[:300]}"))
                            continue
                        cell = dict(base, **summarize(tc, batch))
                        cell["memory"] = {"gpu_peak_bytes": gpu_mem_peak(tf, cfg),
                                          "rss_before_bytes": rss0, "rss_after_bytes": rss_now_bytes(),
                                          "rss_high_water_bytes": rss_peak_bytes()}
                        res["cells"].append(cell)
                        log(f"[bench_b] {cfg} {arch} ae{i} b{batch} {mode} {source}: "
                            f"p50 {cell['latency_s']['p50'] * 1e3:.3f} ms n={cell['n_measured']}")
        for source in sources:
            objs = trained_objs if source == "trained" else random_objs
            for batch in (1, EXECUTION_BS):
                mode = "numpy" if arch in NUMPY_ARCHS else ("graph" if batch == 1 else "predict")
                if time.perf_counter() - t_global0 > global_budget_s:
                    break
                cell = measure_ensemble(arch, objs, classes, batch, mode, opts, tf, device, cfg, rng, flops,
                                        mem_cap)
                cell.update(config=cfg, device=device, weights=source, unit="ensemble",
                            plan=PLAN_TAG.get(cell_key(mode, batch), "[EXPLORATORIO]"))
                res["cells"].append(cell)
        del trained_objs, random_objs
        gc.collect()
        if arch in KERAS:
            tf.keras.backend.clear_session()
    res["nvidia_smi_after"] = nvidia_smi()
    res["finished"] = C.now_iso()
    res["wall_s"] = time.perf_counter() - t_global0
    return res


def measure_ensemble(arch, objs, classes, batch, mode, opts, tf, device, cfg, rng, flops, mem_cap):
    """Los 4 AEs en secuencia + la capa de salida `stat` sobre los 4 errores: el coste de una ventana completa."""
    import numpy as np
    eb = effective_batch(mode, batch)
    base = {"arch": arch, "ae": None, "batch": batch, "mode": mode, "unit": "ensemble",
            "est_macs": sum(macs_per_call(arch, batch, L_WINDOW, nv, n_hidden_of(nv)) for nv in FM_SIZES),
            "est_activation_bytes": max(activation_bytes(arch, eb, L_WINDOW, nv, n_hidden_of(nv))
                                        for nv in FM_SIZES)}
    base["est_call_s"] = 2.0 * base["est_macs"] / flops
    base["est_bytes_with_safety"] = base["est_activation_bytes"] * opts["mem_safety_factor"]
    if base["est_bytes_with_safety"] > cell_mem_cap(cfg, mem_cap, opts):
        return dict(base, skipped="est_too_big")
    if base["est_call_s"] > opts["max_call_s"]:
        return dict(base, skipped="est_too_slow")
    out_ae = classes["stat"](n_visible=len(FM_SIZES))
    prepare_numpy_ae("stat", out_ae, len(FM_SIZES))
    xs, models = [], []
    for i, nv in enumerate(FM_SIZES):
        if arch in NUMPY_ARCHS:
            xs.append(make_input(arch, batch, nv, rng))
            models.append(objs[i])
        else:
            xs.append(make_input(arch, batch, nv, rng, tf=tf, device=device))
            models.append(objs[i] if hasattr(objs[i], "predict") else objs[i].model)
    graphs = None
    if mode == "graph":
        graphs = [tf.function(lambda t, m=m: m(t, training=False),
                              input_signature=[tf.TensorSpec(x.shape, tf.float32)])
                  for m, x in zip(models, xs)]
    xs_np = [np.asarray(x) for x in xs] if mode == "predict" else None

    targets = [np.asarray(x)[:, -1, :] if arch in WINDOWED else np.asarray(x) for x in xs]

    def fn():
        err = np.zeros((batch, len(FM_SIZES)), dtype=np.float64)
        for j in range(len(FM_SIZES)):
            if arch in NUMPY_ARCHS:
                err[:, j] = np.asarray(models[j].execute(xs[j]))
                continue
            z = graphs[j](xs[j]).numpy() if mode == "graph" else models[j].predict(xs_np[j], verbose=0)
            err[:, j] = np.sqrt(np.mean((targets[j] - z) ** 2, axis=1))
        out_ae.execute(err)

    try:
        with tf.device(device):
            tc = time_calls(fn, max(1, opts["warmup"] // 5), max(1, opts["ensemble_iters"]),
                            opts["cell_budget_s"])
    except Exception as e:
        return dict(base, skipped="oom" if C.looks_like_oom(e) else "error",
                    error=f"{type(e).__name__}: {str(e)[:300]}")
    cell = dict(base, **summarize(tc, batch))
    cell["memory"] = {"gpu_peak_bytes": gpu_mem_peak(tf, cfg), "rss_high_water_bytes": rss_peak_bytes()}
    cell["note"] = ("4 AEs en secuencia + capa de salida StatisticalAnomaly sobre los 4 errores; el error por fila "
                    "se calcula con numpy para no medir el bucle Python de execute() dos veces")
    return cell


def run_postproc_config(opts, log=print):
    """Rows per second of KMD, MVD, the causal filters, the fusion and the threshold (plan §9.4, item 5); timings only."""
    os.environ.setdefault("OMP_NUM_THREADS", str(opts["postproc_threads"]))
    import numpy as np
    sys.path.insert(0, str(HERE))
    import metrics_rev as M
    res = {"config": "postproc", "started": C.now_iso(), "cells": [],
           "versions": M.versions(), "notes": [], "plan": PLAN_TAG["postproc"]}
    try:
        res["env_guard"] = M.env_guard()
    except Exception as e:
        res["env_guard"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        res["notes"].append("las versiones no coinciden con el lock: los tiempos valen, la reproducibilidad no")
    src = opts["postproc_scores"]
    job = find_job("original", opts["results_root"])
    if src == "auto" and job is not None and (job / "scores.npy").exists():
        s_full = np.load(job / "scores.npy").astype(np.float64)
        res["scores_source"] = {"kind": "job", "job_id": job.name, "path": str(job / "scores.npy"),
                                "n_rows": int(s_full.size)}
    elif src not in ("auto", "synthetic"):
        s_full = np.load(src).astype(np.float64)
        res["scores_source"] = {"kind": "file", "path": src, "n_rows": int(s_full.size)}
    else:
        rng = np.random.default_rng(20261001)
        n = 2_278_689
        s_full = np.abs(rng.standard_normal(n)) * 0.01
        res["scores_source"] = {"kind": "synthetic", "n_rows": n, "seed": 20261001,
                                "note": "sin job DONE: los tiempos de KMeans dependen algo de los datos"}
    t_train = 200_000
    n_full = int(s_full.size)
    if opts["postproc_max_rows"]:
        s_full = s_full[:min(len(s_full), t_train + int(opts["postproc_max_rows"]))]
    n = int(s_full.size)
    calib_end = t_train + int(0.10 * (n - t_train))
    s_det, s_raw, san_info = M.sanitize_scores(s_full, t_train)
    res["sanitize"] = san_info
    s_prefix = s_det[t_train:calib_end]
    s_exec = s_det[t_train:]
    fit_mask = np.ones(s_prefix.shape, dtype=bool)
    fcal_info = {"mask": "todo True (sin exclusion de duplicados F_cal)",
                 "warning": "el tiempo de `fit` NO es el del protocolo implementado"}
    if (res.get("scores_source", {}).get("kind") == "job" and job is not None
            and (job / "config.json").exists()):
        try:
            t_m = time.perf_counter()
            dup, minfo = M.fit_set_mask(C.read_json(job / "config.json"), n_full, t_train, log=log)
            fit_mask = ~np.asarray(dup[t_train:calib_end], dtype=bool)
            fcal_info = {"mask": "F_cal real (plan §2.1): calibracion sin duplicados exactos de [0, T)",
                         "n_fit": int(fit_mask.sum()), "n_calib": int(fit_mask.size),
                         "mask_seconds": round(time.perf_counter() - t_m, 1),
                         "source": minfo.get("mask_source"), "job_id": job.name}
        except Exception as e:
            fcal_info = dict(fcal_info, error=f"{type(e).__name__}: {str(e)[:300]}")
    res["fit_set"] = fcal_info
    res["regions"] = {"N": n, "N_full": n_full, "T": t_train, "calib_end": calib_end,
                      "n_calib": int(s_prefix.size), "n_fit": int(fit_mask.sum()),
                      "n_from_T": int(s_exec.size),
                      "truncated": bool(opts["postproc_max_rows"])}
    res["constants"] = {"W_MVD": M.W_MVD, "F_FILT": M.F_FILT, "K_KMD": M.K_KMD, "Q_MAIN": M.Q_MAIN}

    def cell(name, fn, rows, iters, tag="[CONFIRMATORIO plan §9.4]"):
        t0 = time.perf_counter()
        try:
            tc = time_calls(fn, 1 if iters > 1 else 0, iters, opts["cell_budget_s"])
        except Exception as e:
            return {"name": name, "rows": rows, "error": f"{type(e).__name__}: {str(e)[:300]}", "plan": tag}
        sm = summarize(tc, 1)
        p50 = sm["latency_s"]["p50"]
        out = {"name": name, "rows": int(rows), "plan": tag, "rows_per_s": {
            "p50": rows / p50 if p50 else None,
            "p99": rows / sm["latency_s"]["p99"] if sm["latency_s"]["p99"] else None}}
        out.update({k: v for k, v in sm.items()
                    if k not in ("batch", "latency_per_window_s", "throughput_windows_per_s")})
        out["cell_wall_s"] = time.perf_counter() - t0
        log(f"[bench_b] postproc {name}: {out['rows_per_s']['p50']:,.0f} filas/s (n={sm['n_measured']})")
        return out

    it = opts["postproc_iters"]
    res["cells"].append(cell("fit_total (KMeans K=10 + series + escalas robustas + cuantiles)",
                             lambda: M.fit(s_prefix, fit_mask, t_train), s_prefix.size, max(1, it // 2)))
    params = M.fit(s_prefix, fit_mask, t_train)
    mu, sd = params["mvd"]["mu_ref"], params["mvd"]["sd_ref"]
    cents = np.asarray(params["kmd"]["centroids"], dtype=np.float64)
    res["cells"].append(cell("fit_kmeans (K=10, n_init=10, 1 thread OpenMP)",
                             lambda: M.fit_kmeans(s_prefix[fit_mask]), int(fit_mask.sum()), max(1, it // 2)))
    res["cells"].append(cell("transform (KMD + MVD + filtros + fusion, causal)",
                             lambda: M.transform(params, s_exec), s_exec.size, it))
    res["cells"].append(cell("detector_series (KMD_raw, MVD_raw, KMD_f, MVD_f)",
                             lambda: M.detector_series(s_exec, mu, sd, cents), s_exec.size, it))
    res["cells"].append(cell("MVD: media y desviacion moviles W=10.000",
                             lambda: (M._rolling(s_exec, M.W_MVD).mean().to_numpy(),
                                      M._rolling(s_exec, M.W_MVD).std(ddof=0).to_numpy()), s_exec.size, it))
    det = M.detector_series(s_exec, mu, sd, cents)
    kmd_raw, mvd_raw = det["KMD_raw"], det["MVD_raw"]
    res["cells"].append(cell("KMD: distancia al centroide mas cercano",
                             lambda: M.nearest_centroid_dist(s_exec, cents), s_exec.size, it))
    res["cells"].append(cell("filtro KMD: media movil causal F=50.001",
                             lambda: M._rolling(kmd_raw, M.F_FILT).mean().to_numpy(), s_exec.size, it))
    res["cells"].append(cell("filtro MVD: mediana movil causal F=50.001",
                             lambda: M._rolling(mvd_raw, M.F_FILT).median().to_numpy(), s_exec.size, it))
    res["cells"].append(cell("fusion (z robusto + media)",
                             lambda: M._fused(det, params["fusion"]["robust"]), s_exec.size, it))
    fus = M._fused(det, params["fusion"]["robust"])["FUS"]
    tau = params["thresholds"][str(M.Q_MAIN)]["FUS"]
    res["cells"].append(cell("umbral (alarma FUS > tau)", lambda: (fus > tau).sum(), s_exec.size, it))
    res["cells"].append(cell("sanitize_scores (no finitos causales, §12.2)",
                             lambda: M.sanitize_scores(s_full, t_train), n, max(1, it // 2)))
    res["helper_hashes"] = M.helper_hashes()
    res["finished"] = C.now_iso()
    return res


POSTPROC_EN = {
    "fit_total": "Calibration (K-means, detector series, robust scales, quantiles)",
    "fit_kmeans": "Calibration: K-means ($K=10$)",
    "transform": "Detectors + causal filters + fusion (full pass)",
    "detector_series": "Detector series (KMD, MVD, raw and filtered)",
    "MVD:": "MVD: causal rolling mean and s.d. ($W=10{,}000$)",
    "KMD:": "KMD: nearest-centroid distance",
    "filtro KMD": "KMD filter: causal moving average ($F=50{,}001$)",
    "filtro MVD": "MVD filter: causal moving median ($F=50{,}001$)",
    "fusion": "Fusion (robust $z$-scores, mean)",
    "umbral": "Thresholding (alarm if FUS $>$ tau)",
    "sanitize_scores": "Non-finite score handling",
}


def postproc_label(name):
    for k, v in POSTPROC_EN.items():
        if name.startswith(k):
            return v
    return tex_escape(name)


def tex_escape(t):
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#"),
                 ("\u00a7", "\\S{}")):
        t = t.replace(a, b)
    return t


def _fmt(v, nd=2):
    """Formato sin paquetes LaTeX externos (nada de siunitx): separador de millar con \\,."""
    if v is None or v != v:
        return "--"
    if nd == 0:
        return f"{round(v):,}".replace(",", "\\,")
    if v != 0 and (abs(v) >= 1e6 or abs(v) < 1e-3):
        m, e = f"{v:.{nd}e}".split("e")
        return f"${m}\\times 10^{{{int(e)}}}$"
    return f"{v:,.{nd}f}".replace(",", "\\,")


def pick(cells, prefer_weights="trained", **kw):
    """Celda que se publica. F-07: `weights` y `mode` tienen que ir RESTRINGIDOS, no salir por el orden en que se
    anadieron las celdas. Se prefieren los pesos entrenados; si no hay, se devuelve la de pesos aleatorios, que el
    llamante marca en la tabla (la comparacion entrenados/aleatorios de este mismo JSON llega al 42 % de
    diferencia relativa en alguna celda, asi que no se puede afirmar que el tiempo no depende de los pesos)."""
    cand = [c for c in cells
            if all(c.get(k) == v for k, v in kw.items()) and "skipped" not in c and c.get("n_measured")]
    for w in ([prefer_weights] if prefer_weights else []) + ["trained", "random"]:
        for c in cand:
            if c.get("weights") == w:
                return c
    return cand[0] if cand else None


MAIN_POSTPROC_STAGES = ("fit_total", "fit_kmeans", "transform")
PLAN_ITERS = 1000
PLAN_CPU_THREADS = 24


def table_is_preliminary(merged):
    """(preliminar, detalle). Una tabla medida en seco, con celdas truncadas por presupuesto o con el stream de
    post-proceso recortado NO puede salir con el mismo aspecto que la definitiva (BENCH-1)."""
    opts = (merged.get("provenance") or {}).get("options") or {}
    cells = [c for cfg in merged.get("configs", {}).values() for c in cfg.get("cells", [])]
    ens = [c for c in cells if c.get("unit") == "ensemble" and c.get("n_measured")]
    short = [c for c in ens if (c.get("n_requested") or 0) > (c.get("n_measured") or 0)]
    det = {"dry_run": bool(opts.get("dry_run")), "cells_short": len(short),
           "n_measured_min": min([c.get("n_measured") or 0 for c in ens] or [0]),
           "n_requested_max": max([c.get("n_requested") or 0 for c in ens] or [0]),
           "cpu_threads": opts.get("cpu_threads"), "postproc_max_rows": opts.get("postproc_max_rows"),
           "untrained": sorted({c.get("arch") for c in ens if c.get("weights") != "trained"})}
    det["below_plan_iters"] = det["n_requested_max"] < PLAN_ITERS
    det["cpu_threads_off_plan"] = bool(det["cpu_threads"]) and det["cpu_threads"] != PLAN_CPU_THREADS
    return bool(det["dry_run"] or short or det["postproc_max_rows"] or det["below_plan_iters"]), det


def tex_filename(merged):
    return "bench_b.DRYRUN-NO-CITAR.tex" if table_is_preliminary(merged)[0] else "bench_b.tex"


def latex_table(merged):
    """Tabla de coste por arquitectura (R1.3). Sin AUC ni F1 (plan §9.4 punto 10).

    BENCH-1: una prueba en seco (pocas medidas, sin GPU, con el stream truncado o con otros jobs vivos) produce un
    .tex identico al definitivo. Se marca en el fichero y EN EL PIE, para que no pueda pasar por un resultado.
    """
    cells = [c for cfg in merged["configs"].values() for c in cfg.get("cells", [])]
    n_threads = (merged["configs"].get("cpucnt") or {}).get("requested_threads", 24)
    opts = (merged["provenance"].get("options") or {})
    warn, det = table_is_preliminary(merged)
    dry, short = det["dry_run"], det["cells_short"]
    untrained, n_min, n_req = det["untrained"], det["n_measured_min"], det["n_requested_max"]
    head = []
    if warn:
        head = ["% ============================================================================",
                "% DRY RUN / INCOMPLETE MEASUREMENT: not the table of the analysis plan",
                f"%   dry_run = {dry}; celdas del conjunto con n_measured < n_requested: {short};",
                f"%   n_measured minimo = {n_min} de {n_req} solicitadas; threads CPU = {opts.get('cpu_threads')};",
                f"%   postproc_max_rows = {opts.get('postproc_max_rows')}",
                "% ============================================================================"]
    dry_txt = ""
    if warn:
        dry_txt = (" \\textbf{Dry-run measurement:} "
                   + (f"this table was produced by a dry run: {n_min} timed calls per cell were measured "
                      f"({n_req} requested) against the {PLAN_ITERS} the analysis plan asks for, on "
                      f"{opts.get('cpu_threads')} CPU threads instead of the {PLAN_CPU_THREADS} of the plan"
                      if dry else f"at least one cell has {n_min} of the {n_req} requested measurements, against "
                                  f"the {PLAN_ITERS} of the analysis plan")
                   + (", and the post-processing stream was truncated"
                      if opts.get("postproc_max_rows") else "") + ".")
    body, untrained_pub = [], []
    for arch in ARCHS:
        ps = merged["params_by_arch"].get(arch)
        lat1 = pick(cells, config="cpu1", arch=arch, unit="ensemble", batch=1)
        mark = ""
        if lat1 and lat1.get("weights") != "trained":
            mark = "$^{r}$"
            untrained_pub.append(arch)
        row = [tex_escape(arch) + mark, _fmt(ps, 0) if ps is not None else "--",
               str(lat1.get("n_measured")) if lat1 else "--"]
        for cfgname in ("gpu", "cpu1", "cpucnt"):
            lat = pick(cells, config=cfgname, arch=arch, unit="ensemble", batch=1)
            thr = pick(cells, config=cfgname, arch=arch, unit="ensemble", batch=EXECUTION_BS)
            lat_ms = lat["latency_s"]["p50"] * 1e3 if lat else None
            p99_ms = lat["latency_s"]["p99"] * 1e3 if lat else None
            p99_txt = (_fmt(p99_ms, 3) + ("$^{m}$" if (lat.get("n_measured") or 0) < 100 else "")) if lat else "--"
            wps = thr["throughput_windows_per_s"]["p50"] if thr else None
            if cfgname == "cpucnt":
                row.append(_fmt(wps, 0))
            else:
                row += [_fmt(lat_ms, 3), p99_txt, _fmt(wps, 0)]
        body.append(" & ".join(row) + " \\\\")
    untrained = untrained_pub
    lines = ["% Generado por revision/experiments/bench_b.py (plan congelado §9.4; R1.3, R2.2).",
             "% Sin metricas de rendimiento predictivo (plan §9.4 punto 10).",
             f"% Fuente: {merged['provenance']['output_json']}",
             f"% sha256(bench_b.py) = {merged['provenance']['bench_b_sha256']}"] + head
    lines += ["\\begin{table}[t]", "\\centering",
              "\\caption{Per-architecture inference cost of the ensemble (four autoencoders over the Active "
              "Wiretap feature map, $k=4$, $L=800$). Latency is the forward pass of one window at batch~1, the "
              "per-decision cost an operator waits for; throughput is measured with "
              "the batch size used by the implementation ($\\mathrm{EXECUTION\\_BS}=512$, internally re-batched "
              "to 32 by \\texttt{model.predict}) and is an amortised figure, so it is not the reciprocal of the "
              "latency column. The end-to-end throughput of a full run of each variant, including feature "
              "extraction and the per-row Python loop, is reported in Table~\\ref{tab:cost}; the two tables "
              "measure different things and their figures are not comparable directly. "
              "Parameters are summed over the four autoencoders. $n$ is the number of timed calls behind each "
              "percentile; a $p_{99}$ obtained from fewer than 100 calls is the maximum of those $n$ calls and is "
              "marked $^{m}$. Dashes mark configurations that were not executable (see notes)."
              + (" Architectures marked $^{r}$ were measured with untrained weights because no checkpoint was "
                 "available: " + tex_escape(", ".join(a for a in untrained if a)) + "." if untrained else "")
              + dry_txt + "}",
              "\\label{tab:cost-per-architecture}",
              "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
              f"Architecture & Params & $n$ & \\multicolumn{{3}}{{c}}{{GPU}} "
              f"& \\multicolumn{{3}}{{c}}{{CPU (1 thread)}} & CPU ({n_threads} threads) \\\\",
              "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-10}",
              " & & & $p_{50}$ (ms) & $p_{99}$ (ms) & win/s & $p_{50}$ (ms) & $p_{99}$ (ms) & win/s & win/s \\\\",
              "\\midrule"]
    lines += body
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    pp = merged["configs"].get("postproc", {})
    pp_rows = [c for c in pp.get("cells", []) if not c.get("error") and c.get("rows_per_s")]
    main_rows = [c for c in pp_rows if c["name"].startswith(MAIN_POSTPROC_STAGES)]
    stage_rows = [c for c in pp_rows if c not in main_rows]
    reg = pp.get("regions") or {}
    n_calib = reg.get("n_calib")
    n_from_T = reg.get("n_from_T")
    pp_cap = ("Post-processing throughput on CPU (detectors, causal filters, fusion and threshold), measured on "
              + (f"{_fmt(n_from_T, 0)} rows of the evaluation stream of Active Wiretap, with a calibration window "
                 f"of {_fmt(n_calib, 0)} rows" if n_from_T else "the evaluation stream of Active Wiretap")
              + (", i.e. a truncated stream (\\texttt{--postproc-max-rows}), not the full one"
                 if opts.get("postproc_max_rows") else "")
              + ". The calibration row times a K-means fit whose cost grows with the size of that window. "
                "Per-stage rates are reported in the appendix: they are not additive, because each one measures "
                "an isolated vectorised operation over an in-memory array." + dry_txt + "}")
    lines += ["\\begin{table}[t]", "\\centering",
              "\\caption{" + pp_cap,
              "\\label{tab:cost-postprocessing}", "\\begin{tabular}{lrr}", "\\toprule",
              "Stage & Rows & Rows/s ($p_{50}$) \\\\", "\\midrule"]
    for c in main_rows:
        lines.append(f"{postproc_label(c['name'])} & {_fmt(c['rows'], 0)} & "
                     f"{_fmt(c['rows_per_s']['p50'], 0)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    if stage_rows:
        lines += ["% --- apendice: desglose por etapa (F-12). NO son aditivas: cada una mide una expresion "
                  "vectorizada aislada sobre un array en memoria, y a estas magnitudes entran los efectos de "
                  "cache y la resolucion del reloj.",
                  "\\begin{table}[t]", "\\centering",
                  "\\caption{Appendix: per-stage post-processing rates. These rates are not additive: each one "
                  "measures an isolated vectorised operation over an in-memory array, so they bound no "
                  "end-to-end cost; the operational figures are in "
                  "Table~\\ref{tab:cost-postprocessing}." + dry_txt + "}",
                  "\\label{tab:cost-postprocessing-stages}", "\\begin{tabular}{lrr}", "\\toprule",
                  "Stage & Rows & Rows/s ($p_{50}$) \\\\", "\\midrule"]
        for c in stage_rows:
            lines.append(f"{postproc_label(c['name'])} & {_fmt(c['rows'], 0)} & "
                         f"{_fmt(c['rows_per_s']['p50'], 0)} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    lines.append("% --- notas de trazabilidad (no van al manuscrito tal cual) ---")
    gp = merged["configs"].get("gpu", {})
    lines.append(f"% GPU: {'medida' if not gp.get('skipped') and gp else 'NO medida'}"
                 f"{' - ' + str(gp.get('skip_reason'))[:200] if gp.get('skipped') else ''}")
    for c in cells:
        if c.get("skipped"):
            lines.append(f"%   no ejecutada: {c.get('config')} {c.get('arch')} ae={c.get('ae')} "
                         f"b={c.get('batch')} {c.get('mode')} {c.get('weights')} -> {c['skipped']} "
                         f"(est {c.get('est_call_s', 0):.2f} s/llamada, "
                         f"{c.get('est_activation_bytes', 0) / (1 << 30):.2f} GiB)")
    for c in cells:
        if c.get("budget_hit"):
            lines.append(f"%   presupuesto: {c.get('config')} {c.get('arch')} ae={c.get('ae')} "
                         f"b={c.get('batch')} {c.get('mode')} {c.get('weights')} -> "
                         f"{c.get('n_measured')}/{c.get('n_requested')} medidas")
    lines.append(f"% comparacion pesos entrenados/aleatorios: max |dif| relativa = "
                 f"{merged['weights_comparison']['max_rel_diff']:.3f}")
    return "\n".join(lines) + "\n"


def provenance(opts, out_json):
    files = {"bench_b.py": Path(__file__).resolve(), "common.py": HERE / "common.py",
             "metrics_rev.py": HERE / "metrics_rev.py", "networks_ctx.py": HERE / "networks_ctx.py"}
    for rel in ("src/models/kitnet.py", "src/models/oopsie.py", "src/models/__special__.py",
                "src/models/networks/conv1d.py", "src/models/networks/conv2d.py", "src/models/networks/lstm.py",
                "src/models/networks/mha.py", "src/models/networks/multilayer.py",
                "src/models/networks/original_kitsune.py", "src/models/networks/stdev.py",
                "src/models/utils/sequential.py"):
        files[rel] = Path(opts["code_root"]) / rel
    hashes = {}
    for k, p in files.items():
        try:
            hashes[k] = C.sha256_file(p)
        except OSError as e:
            hashes[k] = f"error: {e}"
    return {"script": str(Path(__file__).resolve()), "bench_b_sha256": hashes["bench_b.py"],
            "sha256": hashes, "output_json": str(out_json), "generated": C.now_iso(),
            "host": C.hostname(), "platform": platform.platform(), "cpu": cpu_model(),
            "git": git_state(opts["code_root"]), "options": {k: (str(v) if isinstance(v, Path) else v)
                                                             for k, v in opts.items()},
            "plan": {"section": "§9.4 (R1.3, R2.2)", "frozen_tag": "analysis-plan-frozen",
                     "rule": "sin metricas de rendimiento predictivo (plan §9.4 punto 10)"},
            "model_config": {"dataset": DATASET, "feature_map_sizes": FM_SIZES, "L": L_WINDOW,
                             "hidden_ratio": HIDDEN_RATIO, "lr": LR, "EXECUTION_BS": EXECUTION_BS,
                             "k_clusters": len(FM_SIZES)}}


def cpu_model():
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def git_state(root):
    out = {}
    for name, args in (("commit", ["rev-parse", "HEAD"]), ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
                       ("describe", ["describe", "--tags", "--always"]), ("status", ["status", "--porcelain"])):
        try:
            p = subprocess.run(["git", "-C", str(root)] + args, capture_output=True, text=True, timeout=30)
            out[name] = p.stdout.strip()[:2000]
        except Exception as e:
            out[name] = f"{type(e).__name__}: {e}"
    return out


def params_by_arch(merged):
    """Parametros totales (suma de los 4 AEs) por arquitectura, del primer config que los tenga."""
    out = {}
    for cfg in merged["configs"].values():
        for arch, rows in (cfg.get("params") or {}).items():
            if arch in out:
                continue
            tot = sum(r["params"] for r in rows if isinstance(r.get("params"), int))
            out[arch] = tot
    return out


def summary_table(merged):
    """Resumen legible: una fila por (arquitectura, config) con la celda del conjunto."""
    cells = [c for cfg in merged["configs"].values() for c in cfg.get("cells", [])]
    rows = []
    for arch in ARCHS:
        for cfgname in ("gpu", "cpu1", "cpucnt"):
            lat = pick(cells, config=cfgname, arch=arch, unit="ensemble", batch=1)
            thr = pick(cells, config=cfgname, arch=arch, unit="ensemble", batch=EXECUTION_BS)
            if lat is None and thr is None:
                continue
            n_b1 = lat["n_measured"] if lat else None
            rows.append({"arch": arch, "config": cfgname,
                         "params_total": merged["params_by_arch"].get(arch),
                         "latency_ms_p50_b1": (lat["latency_s"]["p50"] * 1e3) if lat else None,
                         "latency_ms_p99_b1": (lat["latency_s"]["p99"] * 1e3) if lat else None,
                         "latency_p99_b1_is_max_of_n": (n_b1 is not None and n_b1 < 100),
                         "budget_hit_b1": bool(lat.get("budget_hit")) if lat else None,
                         "budget_hit_b512": bool(thr.get("budget_hit")) if thr else None,
                         "n_requested_b1": lat.get("n_requested") if lat else None,
                         "windows_per_s_p50_b512": (thr["throughput_windows_per_s"]["p50"] if thr else None),
                         "n_measured_b512": thr["n_measured"] if thr else None,
                         "mode_b1": lat.get("mode") if lat else None,
                         "mode_b512": thr.get("mode") if thr else None,
                         "plan_b1": lat.get("plan") if lat else None,
                         "plan_b512": thr.get("plan") if thr else None,
                         "weights": (lat or thr).get("weights"),
                         "weights_b1": lat.get("weights") if lat else None,
                         "weights_b512": thr.get("weights") if thr else None,
                         "n_measured_b1": n_b1,
                         "gpu_peak_bytes": ((lat or thr).get("memory") or {}).get("gpu_peak_bytes")})
    return rows


def weights_comparison(merged):
    """Plan §9.4: entrenados frente a aleatorios; si difieren > 10 %, la tabla usa los entrenados."""
    cells = [c for cfg in merged["configs"].values() for c in cfg.get("cells", [])]
    idx = {}
    for c in cells:
        if "skipped" in c or not c.get("n_measured") or "arch" not in c:
            continue
        key = (c.get("config"), c["arch"], c.get("ae"), c.get("batch"), c.get("mode"), c.get("unit", "ae"))
        idx.setdefault(key, {})[c.get("weights")] = c["latency_s"]["p50"]
    out, worst = [], 0.0
    for key, d in sorted(idx.items(), key=lambda kv: str(kv[0])):
        if "trained" in d and "random" in d and d["random"]:
            rel = abs(d["trained"] - d["random"]) / d["random"]
            worst = max(worst, rel)
            out.append({"config": key[0], "arch": key[1], "ae": key[2], "batch": key[3], "mode": key[4],
                        "unit": key[5], "p50_trained_s": d["trained"], "p50_random_s": d["random"],
                        "rel_diff": rel})
    return {"pairs": out, "max_rel_diff": worst,
            "rule": "plan §9.4: si difieren > 10 % se usan los entrenados (la tabla los usa siempre que existen)",
            "exceeds_10pct": worst > 0.10}


def merge(partials, opts, out_json):
    merged = {"schema": "bench_b/1", "plan_section": "§9.4", "generated": C.now_iso(),
              "provenance": provenance(opts, out_json), "configs": {}, "notes": []}
    for name, obj in partials.items():
        merged["configs"][name] = obj
    merged["provenance"]["versions"] = {
        "parent_python": sys.version.split()[0],
        **{f"{name}_tf": cfg.get("tf") for name, cfg in merged["configs"].items() if cfg.get("tf")},
        **({"postproc": merged["configs"]["postproc"].get("versions")} if "postproc" in merged["configs"] else {})}
    merged["params_by_arch"] = params_by_arch(merged)
    merged["summary"] = summary_table(merged)
    merged["weights_comparison"] = weights_comparison(merged)
    merged["deviations_proposed"] = deviations(merged)
    return merged


def deviations(merged):
    """Deviation records built from what actually happened in this run of the benchmark."""
    rows = []
    cells = [c for cfg in merged["configs"].values() for c in cfg.get("cells", [])]
    short = [c for c in cells if c.get("budget_hit")]
    if short:
        rows.append({"id": "B-N1", "plan": "§9.4: 50 de calentamiento y 1.000 medidas",
                     "done": f"{len(short)} celdas con menos de 1.000 medidas por el presupuesto por celda",
                     "why": "conv1d (kernel = L/4 = 200) y mha (atencion L x L) no caben en 1.000 llamadas",
                     "impact": "p50 estable; el p99 de esas celdas tiene menos soporte (n_measured en el JSON)"})
    skipped = [c for c in cells if c.get("skipped")]
    if skipped:
        kinds = sorted({c["skipped"] for c in skipped})
        rows.append({"id": "B-N2", "plan": "§9.4: lote 1 y lote 512 en las 8 arquitecturas",
                     "done": f"{len(skipped)} celdas no ejecutadas ({', '.join(kinds)})",
                     "why": "estimador analitico de MAC y de bytes de activacion antes de medir",
                     "impact": "las celdas quedan en el JSON con su estimacion; la tabla pone --"})
    gpu = merged["configs"].get("gpu", {})
    if gpu.get("skipped"):
        rows.append({"id": "B-N3", "plan": "§9.4: fila GPU si relu/sign ejecutan en /GPU:0",
                     "done": "fila GPU omitida", "why": gpu.get("skip_reason", "")[:300],
                     "impact": "la tabla de coste se reporta solo en CPU y lo declara"})
    if any(c.get("mode") == "predict" for c in cells):
        rows.append({"id": "B-N4", "plan": "§9.4: pase hacia delante con lote 1 y lote 512 (EXECUTION_BS)",
                     "done": "se miden cuatro modos: `graph` (lote 1), `eager` (pase puro), `predict` (el camino "
                             "implementado, que Keras re-trocea en lotes de 32) y `numpy`. La tabla cita `graph` "
                             "en lote 1 y `predict` en lote 512, y esos son los CONFIRMATORIOS",
                     "why": "model.predict re-trocea, asi que un lote 512 real no existe en el pipeline; y el "
                            "sobrecoste eager de Keras domina el lote 1 (LSTM 215 ms eager frente a 6,5 ms en grafo, "
                            "CPU 1 thread)",
                     "impact": "ninguno numerico: se publican los dos. `eager` queda como [SENSIBILIDAD plan "
                               "§9.4] y `graph`/`predict`/`numpy` como [CONFIRMATORIO plan §9.4], que es lo que "
                               "mide el codigo (F-06)"})
    miss = sorted({a for cfg in merged["configs"].values()
                   for a, w in (cfg.get("weights") or {}).items()
                   if not w.get("available") or w.get("load_error")})
    if miss:
        wc = merged.get("weights_comparison") or {}
        rows.append({"id": "B-N5", "plan": "§9.4: pesos entrenados de los checkpoints de la semilla A (AW)",
                     "done": f"pesos aleatorios en {', '.join(miss)}; la tabla LaTeX marca esas filas con ^r y lo "
                             f"dice en el pie",
                     "why": "no hay checkpoint de esa arquitectura (sin job T7 con checkpoint; `stat` no guarda ckpt)",
                     "impact": f"la comparacion entrenados/aleatorios de este JSON da una diferencia relativa "
                               f"maxima de p50 de {wc.get('max_rel_diff', float('nan')):.3f} en alguna celda, asi "
                               f"que la fila de pesos aleatorios NO es intercambiable con la entrenada: se marca "
                               f"y se declara"})
    if (merged["provenance"].get("options") or {}).get("postproc_max_rows"):
        rows.append({"id": "B-N6", "plan": "§9.4 punto 5: post-proceso sobre el stream de evaluacion",
                     "done": "stream truncado con --postproc-max-rows: la ventana de calibracion se reduce en la "
                             "misma proporcion",
                     "why": "prueba en seco / limite de tiempo",
                     "impact": "KMeans escala con n: las filas/s de calibracion NO son las del protocolo. El pie de "
                               "la tabla dice las filas realmente medidas y el .tex sale marcado como prueba en seco"})
    return rows


def build_opts(a):
    return {"warmup": a.warmup, "iters": a.iters, "compare_iters": a.compare_iters,
            "ensemble_iters": a.ensemble_iters, "cell_budget_s": a.cell_budget_s,
            "global_budget_h": a.global_budget_h, "max_call_s": a.max_call_s, "mem_cap_gib": a.mem_cap_gib,
            "gpu_mem_cap_gib": a.gpu_mem_cap_gib, "mem_safety_factor": a.mem_safety_factor,
            "mem_frac": a.mem_frac, "mem_rlimit_gib": a.mem_rlimit_gib,
            "cpu_threads": a.cpu_threads, "archs": a.archs, "code_root": str(a.code_root),
            "results_root": str(a.results_root), "postproc_scores": a.postproc_scores,
            "postproc_iters": a.postproc_iters, "postproc_max_rows": a.postproc_max_rows,
            "postproc_threads": a.postproc_threads, "dry_run": a.dry_run}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bloque B-final del plan §9.4 (coste computacional, R1.3 y R2.2)")
    p.add_argument("--run-config", choices=CONFIGS, help="modo hijo: mide una sola configuracion")
    p.add_argument("--opts", help="modo hijo: JSON con las opciones resueltas")
    p.add_argument("--out-partial", help="modo hijo: donde escribir el JSON parcial")
    p.add_argument("--configs", default=None, help="lista separada por comas (por defecto todas)")
    p.add_argument("--out", default=None, help="directorio de salida")
    p.add_argument("--dry-run", action="store_true", help="pocas iteraciones, sin GPU; para probar el script")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--compare-iters", type=int, default=200, help="iteraciones de la medida con pesos aleatorios")
    p.add_argument("--ensemble-iters", type=int, default=PLAN_ITERS)
    p.add_argument("--cell-budget-s", type=float, default=20.0)
    p.add_argument("--global-budget-h", type=float, default=3.0, help="por configuracion (hijo)")
    p.add_argument("--max-call-s", type=float, default=20.0, help="celda omitida si UNA llamada se estima mayor")
    p.add_argument("--mem-cap-gib", type=float, default=20.0, help="tope de activacion en CPU")
    p.add_argument("--gpu-mem-cap-gib", type=float, default=38.0, help="tope de activacion en la A6000 (48 GB)")
    p.add_argument("--mem-safety-factor", type=float, default=3.0,
                   help="factor sobre la activacion estimada (TF pide buffers de trabajo ademas de la salida)")
    p.add_argument("--mem-frac", type=float, default=0.5,
                   help="fraccion maxima de la memoria LIBRE que puede pedir una celda")
    p.add_argument("--mem-rlimit-gib", type=float, default=0.0,
                   help="si > 0, RLIMIT_AS en los hijos de CPU: MemoryError en vez de OOM killer")
    p.add_argument("--cpu-threads", type=int, default=24, help="threads de la configuracion CPU multi-thread")
    p.add_argument("--archs", default=",".join(ARCHS))
    p.add_argument("--code-root", default=str(C.DEFAULT_CODE_ROOT))
    p.add_argument("--results-root", default=str(C.RESULTS_ROOT))
    p.add_argument("--postproc-scores", default="auto", help="auto | synthetic | ruta a un .npy")
    p.add_argument("--postproc-iters", type=int, default=5)
    p.add_argument("--postproc-max-rows", type=int, default=0, help="0 = todas las filas desde T")
    p.add_argument("--postproc-threads", type=int, default=4)
    p.add_argument("--python-tf", default=str(PY_TF))
    p.add_argument("--python-postproc", default=str(PY_POSTPROC))
    p.add_argument("--no-nice", action="store_true")
    p.add_argument("--preflight", action="store_true", help="comprueba GPU, RAM y checkpoints y sale")
    p.add_argument("--tex-from", default=None, help="regenera la tabla LaTeX desde un bench_b.json ya escrito")
    a = p.parse_args(argv)
    a.archs = [x for x in a.archs.split(",") if x]
    a.code_root = Path(a.code_root).resolve()
    a.results_root = Path(a.results_root).resolve()
    if a.dry_run:
        a.warmup = min(a.warmup, 2)
        a.iters = min(a.iters, 5)
        a.compare_iters = min(a.compare_iters, 3)
        a.ensemble_iters = min(a.ensemble_iters, 3)
        a.cell_budget_s = min(a.cell_budget_s, 5.0)
        a.global_budget_h = min(a.global_budget_h, 0.25)
        a.max_call_s = min(a.max_call_s, 1.5)
        a.postproc_iters = min(a.postproc_iters, 2)
        a.postproc_max_rows = a.postproc_max_rows or 300_000
    if a.configs is None:
        a.configs = ["cpu1", "cpucnt", "postproc"] if a.dry_run else list(CONFIGS)
    else:
        a.configs = [x for x in a.configs.split(",") if x]
    if a.out is None:
        a.out = str(OUT_DRYRUN if a.dry_run else OUT_DEFAULT)
    a.out = Path(a.out)
    return a


def main(argv=None):
    a = parse_args(argv)
    if not a.no_nice:
        try:
            cur = os.nice(0)
            if cur < 19:
                os.nice(19 - cur)
        except OSError:
            pass
    opts = build_opts(a)
    if a.tex_from:
        src = Path(a.tex_from)
        merged = json.loads(src.read_text())
        tex = src.parent / tex_filename(merged)
        C.write_text_atomic(tex, latex_table(merged))
        print(f"[bench_b] LaTeX regenerado: {tex}")
        return 0
    if a.run_config:
        if a.opts:
            opts.update(json.loads(Path(a.opts).read_text()))
        def _log(msg):
            print(msg, flush=True)

        fn = run_postproc_config if a.run_config == "postproc" else (lambda o, log: run_device_config(
            a.run_config, o, log))
        try:
            res = fn(opts, _log)
        except Exception as e:
            import traceback
            res = {"config": a.run_config, "fatal": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()[-4000:], "finished": C.now_iso()}
        C.write_json_atomic(Path(a.out_partial), res)
        return 0 if "fatal" not in res else 1

    if a.preflight:
        print(f"[bench_b] preflight ({C.now_iso()})")
        print(f"  RAM disponible : {(mem_available_bytes() or 0) / (1 << 30):.1f} GiB")
        print(f"  GPU libre      : {(gpu_free_bytes() or 0) / (1 << 30):.1f} GiB")
        for ln in (nvidia_smi().get("compute_apps") or []):
            print(f"  GPU ocupada por: {ln}")
        for arch in a.archs:
            ck = checkpoint_info(arch, a.results_root)
            print(f"  {arch:9s} pesos entrenados: {'SI' if ck.get('available') else 'NO '} "
                  f"({ck.get('job_id') or ck.get('reason')})")
        print("  El bloque B-final exige la GPU en exclusiva (plan §9.4): no lo lances con jobs vivos.")
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    part_dir = a.out / "partials"
    part_dir.mkdir(exist_ok=True)
    opts_path = part_dir / "opts.json"
    C.write_json_atomic(opts_path, opts)
    print(f"[bench_b] salida: {a.out}")
    print(f"[bench_b] configuraciones: {', '.join(a.configs)}  (dry_run={a.dry_run})")
    partials, t0 = {}, time.perf_counter()
    for cfg in a.configs:
        py = a.python_postproc if cfg == "postproc" else a.python_tf
        pp = part_dir / f"{cfg}.json"
        cmd = [str(py), "-u", str(Path(__file__).resolve()), "--run-config", cfg, "--opts", str(opts_path),
               "--out-partial", str(pp)]
        if not a.no_nice:
            cmd = ["nice", "-n", "19"] + cmd
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(a.code_root), str(HERE), env.get("PYTHONPATH", "")])
        env["LD_LIBRARY_PATH"] = ""
        t1 = time.perf_counter()
        print(f"[bench_b] --> {cfg}: {' '.join(cmd)}")
        r = subprocess.run(cmd, env=env, cwd=str(a.code_root))
        dt = time.perf_counter() - t1
        if pp.exists():
            partials[cfg] = json.loads(pp.read_text())
            partials[cfg]["child_wall_s"] = dt
            partials[cfg]["child_returncode"] = r.returncode
        else:
            partials[cfg] = {"config": cfg, "fatal": f"el hijo no dejo JSON (rc={r.returncode})",
                             "child_wall_s": dt}
        print(f"[bench_b] <-- {cfg}: rc={r.returncode} en {dt:.1f} s")
    out_json = a.out / "bench_b.json"
    merged = merge(partials, opts, out_json)
    merged["wall_s"] = time.perf_counter() - t0
    merged["nvidia_smi_end"] = nvidia_smi()
    C.write_json_atomic(out_json, merged)
    tex = a.out / tex_filename(merged)
    C.write_text_atomic(tex, latex_table(merged))
    print(f"[bench_b] JSON  : {out_json}")
    print(f"[bench_b] LaTeX : {tex}")
    print(f"[bench_b] total : {merged['wall_s']:.1f} s")
    for d in merged["deviations_proposed"]:
        print(f"[bench_b] nota {d['id']}: {d['done']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
