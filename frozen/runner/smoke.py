#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke.py - sizing probe of block S: estimates the time and the GPU memory of a full job with the real shapes; never produces scores."""
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import instrument as I

BUDGETS = {"train_s": 40.0, "predict_s": 25.0, "warmup_steps": 50, "warmup_batches": 3, "loop_rows": 20_000,
           "barrier_timeout_s": 300.0, "mlp_block_rows": 50_000}
OVERRIDE_MAP = {"probe_train_s": "train_s", "probe_predict_s": "predict_s", "probe_warmup_steps": "warmup_steps",
                "probe_warmup_batches": "warmup_batches", "probe_loop_rows": "loop_rows",
                "probe_barrier_timeout_s": "barrier_timeout_s"}


def phase_workers(block):
    m = re.match(r"^S-w(\d+)$", block)
    return int(m.group(1)) if m else 1


def make_step_timer(tf):
    class StepTimer(tf.keras.callbacks.Callback):
        def __init__(self, warmup_steps, budget_s):
            super().__init__()
            self.warmup, self.budget = int(warmup_steps), float(budget_s)
            self.steps = 0
            self.meas_steps = 0
            self.t0 = None
            self.wall = [None, None]
            self.meas_s = 0.0
            self.stopped = False

        def on_train_batch_begin(self, batch, logs=None):
            if self.steps == self.warmup and self.t0 is None:
                self.t0 = time.perf_counter()
                self.wall[0] = time.time()

        def on_train_batch_end(self, batch, logs=None):
            self.steps += 1
            if self.t0 is not None:
                self.meas_steps += 1
                self.meas_s = time.perf_counter() - self.t0
                self.wall[1] = time.time()
                if self.meas_s >= self.budget:
                    self.model.stop_training = True
                    self.stopped = True

        def result(self):
            return {"warmup_steps": self.warmup, "total_steps": self.steps, "measured_steps": self.meas_steps,
                    "measured_s": self.meas_s, "steps_per_s": (self.meas_steps / self.meas_s) if self.meas_s else None,
                    "wall_interval": self.wall, "stopped_by_budget": self.stopped}
    return StepTimer


def barrier(results_root, spec, rows, hb, timeout_s, log):
    w = phase_workers(spec.block)
    if w <= 1:
        return {"workers": 1, "target": 1, "reached": 1, "waited_s": 0.0, "timed_out": False}, None
    R = Path(results_root)
    pending = [r for r in rows if r["block"] == spec.block and not (R / r["job_id"] / "DONE").exists()]
    target = max(1, min(w, len(pending)))
    d = R / ".state" / "_barrier" / spec.block
    d.mkdir(parents=True, exist_ok=True)
    host, pid = C.hostname(), os.getpid()
    me = d / f"{spec.job_id}__{host}__{pid}"
    me.write_text(C.now_iso() + "\n")
    t0 = time.time()
    hb.set_phase("barrera")
    while True:
        live = []
        for m in d.iterdir():
            parts = m.name.rsplit("__", 2)
            if len(parts) == 3 and parts[1] == host and C.pid_alive(parts[2], None, needle="run_job.py"):
                live.append(m.name)
        if len(live) >= target or time.time() - t0 > timeout_s:
            break
        time.sleep(1.0)
    res = {"workers": w, "target": target, "reached": len(live), "waited_s": time.time() - t0,
           "timed_out": len(live) < target, "peers": sorted(live), "start_wall": time.time()}
    log(f"barrera S: {res['reached']}/{target} sondas vivas tras {res['waited_s']:.1f} s")
    return res, me


def run_probe(spec, x, device, tmp, hb, overrides, test_mode, results_root, rows, phases, log):
    import tensorflow as tf
    from src.models import KitNET
    from src.models.__special__ import BATCH_SIZE, EXECUTION_BS
    cfgm = I.frozen_config(spec, overrides, test_mode)
    B = dict(BUDGETS)
    for k, v in cfgm["overrides"].items():
        if k in OVERRIDE_MAP:
            B[OVERRIDE_MAP[k]] = v
    n = len(x)
    fm, ad, L = cfgm["fm"], cfgm["ad"], cfgm["L"]
    anomalies = []
    probe = {"job_id": spec.job_id, "block": spec.block, "variant": spec.variant, "ar": spec.ar, "dataset": spec.dataset,
             "N": n, "fm": fm, "ad": ad, "L": L, "budgets": B, "device": device, "no_scores": True,
             "workers_in_phase": phase_workers(spec.block), "threads": I.thread_env(),
             "caveats": ["fit y predict truncados por presupuesto de tiempo",
                         "las primeras ventanas llevan relleno de ceros (misma forma y coste)",
                         "original y stat no se sondean"]}
    marker = None
    with tf.device(device):
        kn = KitNET(x.shape[-1], **I.kitnet_kwargs(x.shape[-1], cfgm, spec))
        I.check_kitnet_config(kn, cfgm, spec)
        hb.set_phase("probe_fm", n)
        I.gpu_peak_and_reset(device)
        t = time.perf_counter()
        with I.DBSCANRecorder(target_k=cfgm["k"]) as rec:
            for i in range(fm):
                kn.process(x[i], is_last=False)
                hb.row = i
        fm_s = time.perf_counter() - t
        I.check_aes(kn, cfgm, spec)
        try:
            C.write_json_atomic(tmp / "fm_stats.json", I.fm_stats(rec, kn, cfgm["k"], fm))
        except Exception:
            anomalies.append("fm_stats: " + traceback.format_exc()[-1000:])
        probe["fm"] = {"seconds": fm_s, "gpu_peak_bytes": I.gpu_peak_and_reset(device),
                       "groups": [len(g) for g in kn.feature_map]}
        nl = int(min(B["loop_rows"], max(ad - 1, 0), max(n - fm - 1, 0)))
        t = time.perf_counter()
        for i in range(fm, fm + nl):
            kn.process(x[i], is_last=False)
            hb.row = i
        loop_s = time.perf_counter() - t
        loop_rate = (nl / loop_s) if nl and loop_s > 0 else None
        probe["loop"] = {"rows": nl, "seconds": loop_s, "rows_per_s": loop_rate,
                         "note": "solo-append en [fm, fm+rows): coste por fila del bucle Python (S10)"}
        try:
            probe["barrier"], marker = barrier(results_root, spec, rows, hb, B["barrier_timeout_s"], log)
            train = x[fm:fm + ad]
            per_ae = []
            StepTimer = make_step_timer(tf)
            for idx, (ae, fmap) in enumerate(zip(kn.autoencoders, kn.feature_map)):
                r = {"index": idx, "class": type(ae).__name__, "n_visible": len(fmap), "n_hidden": int(ae.n_hidden)}
                xi_raw = train[:, fmap]
                mn, mx = np.min(xi_raw, axis=0), np.max(xi_raw, axis=0)
                xi = (xi_raw - mn) / (mx - mn + 1e-16)
                hb.set_phase(f"probe_train_ae{idx}")
                I.gpu_peak_and_reset(device)
                try:
                    cb = StepTimer(B["warmup_steps"], B["train_s"])
                    if spec.variant in C.WINDOWED_VARIANTS:
                        ds, _ = ae.create_windowed(xi, L, ae.window, batch_size=BATCH_SIZE)
                        ae.model.fit(ds, epochs=1, verbose=0, callbacks=[cb])
                    else:
                        ae.model.fit(xi, xi, epochs=1, batch_size=BATCH_SIZE, verbose=0, callbacks=[cb])
                    r["train"] = cb.result()
                except tf.errors.ResourceExhaustedError as e:
                    r["train_oom"] = str(e)[:800]
                r["gpu_peak_train_bytes"] = I.gpu_peak_and_reset(device)
                hb.set_phase(f"probe_predict_ae{idx}")
                try:
                    r["predict"] = _probe_predict(tf, ae, xi, L, EXECUTION_BS, B, spec)
                except tf.errors.ResourceExhaustedError as e:
                    r["predict_oom"] = str(e)[:800]
                r["gpu_peak_predict_bytes"] = I.gpu_peak_and_reset(device)
                per_ae.append(r)
                log(f"sonda ae{idx}: train {r.get('train', {}).get('steps_per_s')} pasos/s, predict "
                    f"{r.get('predict', {}).get('rows_per_s')} filas/s" + (" OOM" if ("train_oom" in r or "predict_oom" in r) else ""))
            probe["per_ae"] = per_ae
        finally:
            if marker is not None:
                try:
                    marker.unlink()
                except OSError:
                    pass
    exec_rows = max(n - fm - ad, 0)
    est_parts = {"cache_s": phases.get("cache_s"), "tf_import_s": phases.get("tf_import_s"), "fm_s": fm_s,
                 "loop_overhead_s": ((n - fm) / loop_rate) if loop_rate else None, "per_ae": []}
    total_ok = est_parts["loop_overhead_s"] is not None
    total = sum(v for k, v in est_parts.items() if k != "per_ae" and isinstance(v, (int, float)))
    for r in probe.get("per_ae", []):
        tr = (r.get("train") or {}).get("steps_per_s")
        pr = (r.get("predict") or {}).get("rows_per_s")
        e = {"train_fit_s": (ad / tr) if tr else None, "train_eval_s": (ad / pr) if pr else None,
             "exec_s": (exec_rows / pr) if pr else None}
        est_parts["per_ae"].append(e)
        for v in e.values():
            if v is None:
                total_ok = False
            else:
                total += v
    probe["estimate"] = {"parts": est_parts, "estimated_full_job_seconds": total if total_ok else None,
                         "excludes": "salidas, checkpoint y capa de salida (orden de segundos-minutos)",
                         "any_oom": any(("train_oom" in r or "predict_oom" in r) for r in probe.get("per_ae", []))}
    C.write_json_atomic(tmp / "probe.json", probe)
    timing = {"rows": n, "probe": True, "estimated_full_job_seconds": probe["estimate"]["estimated_full_job_seconds"],
              "gpu_peak_bytes": {"fm": probe["fm"]["gpu_peak_bytes"]}}
    return {"config": {"probe": True, "hyperparameters_frozen": cfgm, "budgets": B}, "timing": timing,
            "phases": {"fm_s": fm_s, "loop_probe_s": loop_s}, "anomalies": anomalies}


def _probe_predict(tf, ae, xi, L, EXECUTION_BS, B, spec):
    wb = int(B["warmup_batches"])
    rows, nb, t0, wall0 = 0, 0, None, None
    if spec.variant in C.WINDOWED_VARIANTS:
        ds_e, _ = ae.create_windowed(xi, L, ae.window, batch_size=EXECUTION_BS)
        for (x_batch, y_batch) in ds_e:
            predicted = ae.model.predict(x_batch, verbose=0)
            mse = tf.sqrt(tf.reduce_mean(tf.square(y_batch - predicted), axis=1)).numpy()
            nb += 1
            if nb == wb:
                t0, wall0 = time.perf_counter(), time.time()
            elif nb > wb:
                rows += len(mse)
                if time.perf_counter() - t0 >= B["predict_s"]:
                    break
    else:
        bs = int(B["mlp_block_rows"])
        t_end = None
        while t_end is None:
            for s in range(0, len(xi), bs):
                xb = xi[s:s + bs]
                z = ae.reconstruct(xb)
                _ = np.mean((xb - z) ** 2, axis=1)
                nb += 1
                if nb == wb:
                    t0, wall0 = time.perf_counter(), time.time()
                elif nb > wb:
                    rows += len(xb)
                    if time.perf_counter() - t0 >= B["predict_s"]:
                        t_end = True
                        break
            if nb <= wb and len(xi) == 0:
                break
    meas = (time.perf_counter() - t0) if t0 is not None else None
    return {"batches": nb, "warmup_batches": wb, "measured_rows": rows, "measured_s": meas,
            "rows_per_s": (rows / meas) if (meas and rows) else None, "wall_interval": [wall0, time.time()]}
