#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_job.py - runs one job of the manifest: guards, lock and heartbeat, stream cache, KitNET loop, outputs and atomic commit."""
import argparse
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR))
import common as C

ESSENTIAL = {"full": ["scores.npy", "config.json", "timing.json", "env.json"],
             "fake": ["scores.npy", "config.json", "timing.json", "env.json"],
             "probe": ["probe.json", "config.json", "timing.json", "env.json"]}
NON_ESSENTIAL_FULL = ["ae_errors.npy", "model_stats.npz", "fm_stats.json", "train_log.csv", "arrays.json", "labels.npy"]
_RECORDED = {"failures": 0}
DETECTOR_CALIB_FRACS = C.CALIB_FRACS


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Ejecuta un job del manifiesto del runner v1 (JISA). Ver DESIGN.md §3.")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--manifest", required=True, help="Ruta ABSOLUTA al manifest.csv del arbol principal.")
    ap.add_argument("--results-root", required=True, help="Ruta ABSOLUTA a results/revision (o raiz de test).")
    ap.add_argument("--gpu", required=True, help="Indice de GPU (CUDA_VISIBLE_DEVICES) o 'cpu'.")
    ap.add_argument("--code-root", default=str(C.DEFAULT_CODE_ROOT),
                    help="Arbol con src/ y experiments/ (produccion: worktree ../jisa-runner en runner-vN).")
    ap.add_argument("--data-root", default=str(C.DATA_ROOT))
    ap.add_argument("--cache-root", default=str(C.CACHE_ROOT))
    ap.add_argument("--dirty-wait-s", type=float, default=1800.0)
    ap.add_argument("--dirty-sleep-s", type=float, default=60.0)
    ap.add_argument("--cache-wait-s", type=float, default=3600.0)
    ap.add_argument("--test-mode", action="store_true", help="SOLO tests: variantes _fake_*, --max-rows, overrides.")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--test-override", default="", help="k=v,... (fm_grace, ad_grace, execution_window, "
                                                        "sequence_length, train_rows, probe_*)")
    ap.add_argument("--force-git-check", action="store_true", help="En test-mode, comprobar igualmente el worktree.")
    return ap.parse_args(argv)


def make_log(job_id):
    def log(msg):
        print(f"{C.now_iso()} [run_job {os.getpid()} {job_id}] {msg}", flush=True)
    return log


def parse_overrides(text):
    ov = {}
    for part in [p for p in (text or "").split(",") if p.strip()]:
        k, _, v = part.partition("=")
        ov[k.strip()] = float(v) if "." in v else int(v)
    return ov


class _SignalRaiser:
    def __init__(self):
        self.received = None

    def __call__(self, signum, frame):
        self.received = signum
        raise C.Interrupted(f"senal {signal.Signals(signum).name}")


def main(argv=None):
    a = parse_args(argv)
    t_start = time.time()
    log = make_log(a.job_id)
    for p, what in ((a.manifest, "--manifest"), (a.results_root, "--results-root"), (a.code_root, "--code-root"),
                    (a.data_root, "--data-root"), (a.cache_root, "--cache-root")):
        if not os.path.isabs(p):
            log(f"INVALID: {what} debe ser absoluta: {p}")
            return C.INVALID
    results_root = Path(a.results_root)
    P = C.JobPaths(results_root, a.job_id)
    if P.done.exists():
        log("DONE ya existe: nada que hacer (idempotente)")
        return C.OK
    if P.final.exists():
        log(f"ANOMALIA: {P.final} existe sin DONE; requiere persona (no se toca)")
        return C.ANOMALY
    try:
        row = C.manifest_row(a.manifest, a.job_id)
        spec = C.validate_row(row, test_mode=a.test_mode)
        C.check_supported(spec)
        overrides = parse_overrides(a.test_override)
        if (overrides or a.max_rows is not None or a.force_git_check) and not a.test_mode:
            raise C.InvalidJob("--max-rows, --test-override y --force-git-check solo con --test-mode")
    except NotImplementedError as e:
        log(f"INVALID: {e}")
        return C.INVALID
    except (C.InvalidJob, ValueError, OSError) as e:
        log(f"INVALID: {e}")
        return C.INVALID

    want = None if spec.seed is None else str(spec.seed_int)
    have = os.environ.get("PYTHONHASHSEED")
    if want != have and not spec.is_fake:
        if os.environ.get("JISA_SEED_REEXEC") == "1":
            log(f"ENV: PYTHONHASHSEED={have!r} tras re-exec, se esperaba {want!r}")
            return C.ENV
        env = dict(os.environ, JISA_SEED_REEXEC="1")
        if want is None:
            env.pop("PYTHONHASHSEED", None)
        else:
            env["PYTHONHASHSEED"] = want
        log(f"re-exec para fijar PYTHONHASHSEED={want!r} (heredado {have!r})")
        sys.stdout.flush()
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    sr = _SignalRaiser()
    signal.signal(signal.SIGTERM, sr)
    signal.signal(signal.SIGINT, sr)

    try:
        C.check_not_forbidden(results_root, "results_root")
        if a.test_mode and C.is_under(results_root, C.RESULTS_ROOT):
            raise C.EnvError(f"--test-mode no puede escribir bajo {C.RESULTS_ROOT} (produccion)")
        if a.test_mode and (C.is_under(a.cache_root, C.CACHE_ROOT) or C.is_under(C.CACHE_ROOT, a.cache_root)):
            raise C.EnvError(f"--test-mode no puede usar la cache de produccion {C.CACHE_ROOT} ni rutas que la contengan "
                             f"o esten dentro de ella: {a.cache_root} (M3)")
    except C.EnvError as e:
        log(f"ENV: {e}")
        return C.ENV
    C.account_orphan_moves(results_root, a.job_id, log)
    if C.read_fail_count(results_root, a.job_id) >= C.MAX_FAILS:
        log(f"BLOCKED: fail_count >= {C.MAX_FAILS}")
        return C.BLOCKED
    if os.environ.get("LD_LIBRARY_PATH"):
        log("ENV: LD_LIBRARY_PATH no esta vacio (CUDA del sistema); lanza con LD_LIBRARY_PATH=")
        return C.ENV
    code_root = Path(a.code_root).resolve()
    code_checks = {"code_root": str(code_root), "runner_dir": str(RUNNER_DIR), "main_repo": str(C.MAIN_REPO),
                   "head": C.head_commit(code_root), "tags_at_head": C.tags_at_head(code_root)}
    if not a.test_mode or a.force_git_check:
        if wait_clean(code_root, a.dirty_wait_s, a.dirty_sleep_s, log, results_root, a.job_id):
            return C.DIRTY
        code_checks["worktree_clean"] = True
    try:
        if not a.test_mode:
            if C.realpath(code_root) == C.realpath(C.MAIN_REPO):
                raise C.EnvError("code_root es el arbol principal: los jobs corren desde el worktree ../jisa-runner")
            if not C.is_under(RUNNER_DIR, code_root):
                raise C.EnvError(f"run_job.py ({RUNNER_DIR}) no esta dentro de code_root {code_root} (M4)")
            if not C.is_registered_worktree(code_root):
                raise C.EnvError(f"{code_root} no es un worktree registrado de {C.MAIN_REPO} (M4)")
            tag = C.runner_tag_at_head(code_root)
            if not tag:
                raise C.EnvError(f"HEAD de {code_root} no tiene un tag runner-vN exacto")
            code_checks.update(registered_worktree=True, runner_tag=tag)
        if not spec.is_fake:
            ok, detail = C.src_identical(code_root)
            code_checks["src_identical_to_submitted_v1"] = detail
            if not ok:
                raise C.EnvError(f"codigo original modificado: {detail}")
        if not a.test_mode and spec.save_ckpt:
            ok, detail = C.ckpt_paths_ignored(results_root, a.job_id)
            code_checks["ckpt_gitignore"] = detail
            if ok is False:
                raise C.EnvError(f"ckpt_gitignore: {detail}")
    except C.EnvError as e:
        log(f"ENV: {e}")
        return C.ENV

    host, pid = C.hostname(), os.getpid()
    info = {"job_id": a.job_id, "gpu": a.gpu, "start": C.now_iso(), "timing_ref": int(spec.timing_ref),
            "block": spec.block, "worker": os.environ.get("JISA_WORKER_PID"), "runner_commit": code_checks["head"],
            "test_mode": bool(a.test_mode)}
    lock = C.acquire_lock(P.lock, info)
    if lock is None:
        log("LOCKED: lock ajeno presente")
        return C.LOCKED
    my_ticks = C.proc_start_ticks(pid)
    hb = None
    tmp = None

    def ph():
        return (hb.phase, hb.row, hb.n) if hb is not None else ("lock", 0, 0)

    try:
        if P.done.exists():
            log("DONE aparecio mientras se esperaba antes del lock: nada que hacer (idempotente)")
            return C.OK
        if P.final.exists():
            log(f"ANOMALIA: {P.final} existe sin DONE; requiere persona (no se toca)")
            return C.ANOMALY
        others = C.live_locks_on_host(results_root, host, a.job_id)
        if spec.timing_ref and others:
            log(f"LOCKED: fila timing_ref y hay {len(others)} job(s) vivos en el host ({[j for j, _ in others][:3]}); "
                f"se cede el lock para medir en exclusiva (S11)")
            return C.LOCKED
        tr_live = [j for j, inf in others if str((inf or {}).get("timing_ref", 0)) in ("1", "True", "true")]
        if tr_live and not spec.timing_ref:
            log(f"LOCKED: hay un job timing_ref vivo en el host ({tr_live[0]}); se cede el lock (exclusividad S11)")
            return C.LOCKED
        hb = C.Heartbeat(P.lock, a.job_id)
        adopted = C.adopt_stale_tmps(P, host, pid, log)
        tmp, reuse = P.new_tmp(host, pid)
        if reuse:
            adopted.append(reuse)
            log(f"intento previo con el mismo host.PID: {reuse['action']} (M5)")
        cfg = {"job_id": a.job_id, "manifest_row": row, "manifest_path": a.manifest, "spec": spec.public(),
               "mode": spec.mode, "test_mode": bool(a.test_mode), "test_overrides": overrides, "max_rows": a.max_rows,
               "seed_label": spec.seed, "seed_int": spec.seed_int, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
               "gpu_arg": a.gpu, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "code": code_checks, "results_root": str(results_root), "nan_is_result": True,
               "started": C.now_iso(), "host": host, "pid": pid}
        timing = {"wall_seconds": None, "timing_ref": int(spec.timing_ref), "host": host, "gpu": a.gpu,
                  "test_mode": bool(a.test_mode), "phases": {"lock_to_start_s": time.time() - t_start},
                  "concurrent_jobs_same_host": {"start": len(C.live_locks_on_host(results_root, host, a.job_id))},
                  "gpu_compute_apps": {"start": C.gpu_compute_apps() if a.gpu != "cpu" else None}}
        if spec.is_fake:
            C.write_json_atomic(tmp / "env.json", {"host": host, "fake": True, "python": sys.version,
                                                   "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                                                   "threads": {k: os.environ.get(k) for k in (
                                                       "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                                                       "OMP_NUM_THREADS", "JISA_WORKER_THREADS")}})
            res = run_fake(spec, tmp, hb, P, log)
        else:
            res = run_real(spec, a, overrides, code_root, results_root, tmp, hb, cfg, timing, log)
        anomalies = list(res.get("anomalies", []))
        cfg.update(res.get("config", {}))
        cfg["previous_attempts"] = {"adopted_now": adopted, "in_state": C.list_previous_attempts(P)}
        timing["phases"].update(res.get("phases", {}))
        for k, v in res.get("timing", {}).items():
            timing[k] = v
        timing["concurrent_jobs_same_host"]["end"] = len(C.live_locks_on_host(results_root, host, a.job_id))
        timing["gpu_compute_apps"]["end"] = C.gpu_compute_apps() if a.gpu != "cpu" else None
        timing["rss_peak_bytes"] = _rss_peak()
        mode = spec.mode
        present = {p.name for p in tmp.iterdir()}
        if mode == "full":
            for f in NON_ESSENTIAL_FULL:
                if f not in present:
                    anomalies.append(f"falta la salida no esencial {f}")
            if spec.save_ckpt and "ckpt" not in present:
                anomalies.append("save_ckpt=1 pero no hay ckpt/")
        cfg["anomalies"] = anomalies
        cfg["outputs"] = sorted(present | {"config.json", "timing.json"} | ({"anomalies.json"} if anomalies else set()))
        timing["wall_seconds"] = time.time() - t_start
        C.write_json_atomic(tmp / "config.json", cfg)
        C.write_json_atomic(tmp / "timing.json", timing)
        if anomalies:
            C.write_json_atomic(tmp / "anomalies.json", {"job_id": a.job_id, "anomalies": anomalies,
                                                         "note": "fallos de instrumentacion que no invalidan los scores (M5)"})
        present = {p.name for p in tmp.iterdir()}
        missing = [f for f in ESSENTIAL[mode] if f not in present]
        if missing:
            raise RuntimeError(f"faltan salidas esenciales: {missing}")
        hb.set_phase("commit")
        rc = C.commit(tmp, P)
        if rc == C.DUPLICATE:
            log("DUPLICATE: otro intento hizo commit antes; este intento queda en .state/X/duplicate_*")
        else:
            log(f"DONE en {time.time() - t_start:.1f} s" + (f" con {len(anomalies)} anomalias" if anomalies else ""))
        return rc
    except C.Interrupted as e:
        phase, row_i, row_n = ph()
        C.record_event(results_root, a.job_id, "interrupted", f"{e}\nfase={phase} fila={row_i}/{row_n}\n"
                                                              f"{traceback.format_exc()[-3000:]}")
        _preserve(P, tmp, "interrumpido", log)
        log(f"INTERRUPTED: {e}")
        return C.INTERRUPTED
    except NotImplementedError as e:
        log(f"INVALID: {e}")
        return C.INVALID
    except C.NonExecutableCell as e:
        detail = dict(getattr(e, "detail", {}) or {})
        detail.update(ts=C.now_iso(), job_id=a.job_id, message=str(e), phase=ph()[0], counted=False,
                      note="celda no ejecutable por el codigo original (determinista): celda perdida, analysis_plan.md "
                           "§12.2; el worker no la vuelve a reclamar")
        path = C.record_event(results_root, a.job_id, "nonexec", detail, ext="json")
        _preserve(P, tmp, f"nonexec: {e}", log)
        log(f"INVALID (no ejecutable, no suma fallo): {e} -> {path}")
        return C.INVALID
    except C.InvalidJob as e:
        log(f"INVALID: {e}")
        _preserve(P, tmp, f"invalid: {e}", log)
        return C.INVALID
    except C.EnvError as e:
        log(f"ENV: {e}")
        if "DEV-029" in str(e):
            try:
                path = C.record_event(results_root, a.job_id, "env", f"{e}\nfase={ph()[0]}")
                log(f"ENV registrado (no suma fallo) -> {path}")
            except Exception:
                pass
        _preserve(P, tmp, f"env: {e}", log)
        return C.ENV
    except C.HaltError as e:
        tb = traceback.format_exc()
        path = C.record_event(results_root, a.job_id, "halt", f"{type(e).__name__}: {e}\nfase={ph()[0]}\n{tb[-6000:]}")
        _preserve(P, tmp, f"halt: {e}", log)
        log(f"HALT ({type(e).__name__}, no suma fallo; el worker se detiene): {e} -> {path}")
        return C.HALT
    except BaseException as e:
        tb = traceback.format_exc()
        phase, row_i, row_n = ph()
        ctx = f"fase={phase} fila={row_i}/{row_n} host={host} pid={pid} gpu={a.gpu}\n"
        if C.looks_like_oom(e) or C.looks_like_oom(tb[-4000:]):
            others = C.live_locks_on_host(results_root, host, a.job_id)
            counted = len(others) == 0
            n = C.record_failure(results_root, a.job_id, "OOM\n" + ctx + tb[-6000:]) if counted else None
            if counted:
                _RECORDED["failures"] += 1
            oom = {"ts": C.now_iso(), "job_id": a.job_id, "phase": phase, "row": row_i, "counted": counted,
                   "error_n": n, "concurrent_jobs_same_host": [j for j, _ in others],
                   "gpu_compute_apps": C.gpu_compute_apps(), "message": str(e)[:1000],
                   "oom_chain": {"paso_0": "descartar causas ajenas: con concurrencia > 1 se baja w (no suma fallo)",
                                 "paso_1": "acumulacion de gradiente: no aplicable (batch 1, __special__.py:17)",
                                 "paso_2": "activation checkpointing fp32: cambio de runner (tag nuevo)",
                                 "paso_3": "celda no ejecutable; nunca AMP"},
                   "variant": getattr(spec, "variant", None)}
            if getattr(spec, "variant", None) in C.EXCLUSIVE_VARIANTS:
                oom["oom_chain"]["paso_2b_execution_bs"] = (
                    "NO aplicable: el pico lo fijan pesos + estado de Adam + gradientes, no el lote de inferencia "
                    "(~0,13 GiB de 26 GiB); reducir EXECUTION_BS no recuperaria memoria")
                oom["oom_chain"]["paso_2c_exclusividad"] = (
                    f"{spec.variant} es exclusiva (common.EXCLUSIVE_VARIANTS = {C.EXCLUSIVE_VARIANTS[spec.variant]} "
                    "simultaneo(s)): repetir con un solo worker y sin mt-ssl-era5 antes de dar la celda por no "
                    "ejecutable")
            path = C.record_event(results_root, a.job_id, "oom", oom, ext="json")
            _preserve(P, tmp, "oom", log)
            log(f"OOM ({'suma fallo' if counted else 'con concurrencia: no suma fallo'}): {str(e)[:300]} -> {path}")
            return C.OOM
        if C.looks_like_gpu_kernel_error(e) or C.looks_like_gpu_kernel_error(tb[-4000:]):
            path = C.record_event(results_root, a.job_id, "env", f"kernels GPU de TF\n{ctx}{tb[-6000:]}")
            _preserve(P, tmp, "env: kernels GPU", log)
            log("ENV (kernels GPU de TF; no suma fallo; relanza con --gpu cpu o arregla el driver): "
                f"{type(e).__name__}: {str(e)[:300]} -> {path}")
            return C.ENV
        n = C.record_failure(results_root, a.job_id, ctx + tb[-8000:])
        _RECORDED["failures"] += 1
        _preserve(P, tmp, f"fallo {n}", log)
        log(f"FAILED (error_{n}.txt): {type(e).__name__}: {str(e)[:500]}")
        return C.FAILED
    finally:
        if hb is not None:
            hb.stop()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if not C.release_lock(P.lock, pid, my_ticks) and P.lock.exists():
            info_now = C.read_lock_info(P.lock) or {}
            if str(info_now.get("pid")) == str(pid):
                log("AVISO: no se pudo borrar el lock propio (ver .locks); lo recuperara un worker como obsoleto")
            else:
                log("aviso: el lock no era de este proceso (recuperado por otro worker); no se toca")
        if a.test_mode and os.environ.get("JISA_TEST_RAISE_IN_FINALLY") == "1":
            raise OSError(5, "Input/output error (simulado en la limpieza, solo --test-mode; test a25)")


def _entry(argv=None):
    """Wrapper of main: an exception raised outside its main try block is recorded here as a failure and exits with FAILED."""
    argv = sys.argv[1:] if argv is None else list(argv)
    job_id, results_root = None, None
    try:
        a = parse_args(argv)
        job_id, results_root = a.job_id, a.results_root
    except SystemExit:
        raise
    except Exception:
        pass
    try:
        return main(argv)
    except SystemExit:
        raise
    except C.Interrupted as e:
        print(f"{C.now_iso()} [run_job {os.getpid()} {job_id}] INTERRUPTED antes del lock: {e}", flush=True)
        try:
            if results_root and os.path.isabs(results_root):
                C.record_event(results_root, job_id, "interrupted", f"{e} (antes del lock)")
        except Exception:
            pass
        return C.INTERRUPTED
    except BaseException as e:
        tb = traceback.format_exc()
        print(f"{C.now_iso()} [run_job {os.getpid()} {job_id}] EXCEPCION NO CONTROLADA ({type(e).__name__}: {e})\n"
              f"{tb[-4000:]}", flush=True)
        try:
            done_now = bool(results_root and job_id and os.path.isabs(results_root) and
                            C.JobPaths(results_root, job_id).done.exists())
        except Exception:
            done_now = False
        if done_now and _RECORDED["failures"] == 0:
            print(f"{C.now_iso()} [run_job {os.getpid()} {job_id}] el job ya tiene DONE: la excepcion no suma fallo "
                  f"(revisar .locks por si quedo un lock propio)", flush=True)
            return C.OK
        if _RECORDED["failures"] == 0 and results_root and job_id and os.path.isabs(results_root):
            try:
                n = C.record_failure(results_root, job_id, f"excepcion no controlada en run_job (fuera del try "
                                                           f"principal o dentro de un except)\n{tb[-6000:]}")
                print(f"{C.now_iso()} [run_job {os.getpid()} {job_id}] FAILED (error_{n}.txt)", flush=True)
            except Exception as e2:
                print(f"{C.now_iso()} [run_job {os.getpid()} {job_id}] no se pudo registrar el fallo en .state "
                      f"({type(e2).__name__}: {e2}); lo contara el worker en memoria", flush=True)
        return C.FAILED


def _preserve(P, tmp, reason, log):
    if tmp is None:
        return
    try:
        dest = C.preserve_tmp(P, tmp, reason)
        if dest:
            log(f"intento con scores.npy conservado en {dest} (M5)")
    except Exception as e:
        log(f"aviso: no se pudo conservar {tmp}: {e}")


def _rss_peak():
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


def wait_clean(code_root, wait_s, sleep_s, log, results_root, job_id):
    """True si sigue sucio tras la espera (DIRTY). Nunca suma fallo."""
    t0 = time.time()
    first = True
    while True:
        dirty = C.worktree_dirty(code_root)
        if not dirty:
            return False
        if first:
            C.record_event(results_root, job_id, "dirty", f"worktree sucio en {code_root}:\n{dirty[:2000]}")
            first = False
        log(f"worktree sucio en {code_root} (no suma fallo): {' '.join(dirty.split())[:200]}")
        if time.time() - t0 + sleep_s > wait_s:
            return True
        time.sleep(sleep_s)


def setup_tf_env(gpu):
    if gpu == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        cur = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cur is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        elif cur != str(gpu):
            raise C.EnvError(f"CUDA_VISIBLE_DEVICES={cur!r} != --gpu {gpu}")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")


def resolve_device(tf, gpu):
    if gpu == "cpu":
        return "/CPU:0"
    logical = [d.name for d in tf.config.list_logical_devices("GPU")]
    if not any(n.endswith("GPU:0") for n in logical):
        raise C.EnvError(f"se pidio la GPU {gpu} y TensorFlow no ve /GPU:0 (logical={logical}); el original "
                         f"acabaria en CPU sin avisar")
    try:
        with tf.device("/GPU:0"):
            t = tf.constant([-1.5, 0.0, 2.0], dtype=tf.float32)
            r_t, s_t = tf.nn.relu(t), tf.sign(t)
        r, s = [float(v) for v in r_t.numpy()], [float(v) for v in s_t.numpy()]
        devs = [str(getattr(r_t, "device", "")), str(getattr(s_t, "device", ""))]
    except Exception as e:
        raise C.EnvError(f"DEV-029: kernels GPU de TF no ejecutan en /GPU:0 (tf.nn.relu/tf.sign: {type(e).__name__}: "
                         f"{str(e)[:400]}); lanza con --gpu cpu o corrige el driver")
    if r != [0.0, 0.0, 2.0] or s != [-1.0, 0.0, 1.0]:
        raise C.EnvError(f"DEV-029: relu/sign en /GPU:0 devuelven {r}/{s} (esperado [0, 0, 2]/[-1, 0, 1])")
    if not all(d.upper().endswith("GPU:0") for d in devs):
        raise C.EnvError(f"DEV-029: relu/sign no se ejecutaron en /GPU:0 (dispositivos {devs}); colocacion blanda a CPU")
    return "/GPU:0"


def run_real(spec, a, overrides, code_root, results_root, tmp, hb, cfg, timing, log):
    setup_tf_env(a.gpu)
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    hb.set_phase("tf_import")
    t = time.time()
    import tensorflow as tf
    phases = {"tf_import_s": time.time() - t}
    device = resolve_device(tf, a.gpu)
    import instrument as I
    import pipelines
    import src.models as sm
    if not C.is_under(sm.__file__, code_root):
        raise C.EnvError(f"src.models importado desde {sm.__file__}, no desde {code_root}")
    if spec.variant in C.CONTEXT_VARIANTS:
        cfg["context_variants"] = I.register_context_variants(spec)
        log(f"[EXPLORATORIO]: variante {spec.variant} registrada desde "
            f"{cfg['context_variants']['registered'][spec.variant]['module']}")
    if spec.variant in C.EPS_VARIANTS:
        import networks_eps as _EPS
        cfg["eps_variant"] = _EPS.register_eps_variant()
        log(f"[EXPLORATORIO] S-EPS: variante {spec.variant} registrada desde {cfg['eps_variant']['module']} (eps "
            f"entrenamiento {cfg['eps_variant']['eps_train']:g}, el envio usaba "
            f"{cfg['eps_variant']['eps_train_submitted']:g})")
    cfgm = I.frozen_config(spec, overrides, a.test_mode)
    if spec.contamination_pct and (spec.pipeline != "corrected" or spec.mode != "full"):
        raise C.InvalidJob("T4-E solo con el pipeline corrected y un job completo (analysis_plan.md §8.5)")
    if spec.pipeline == "corrected":
        tr_rows = int(overrides.get("train_rows", pipelines.default_train_rows(spec)))
        if tr_rows != cfgm["fm"] + cfgm["ad"]:
            raise C.FrozenConfigError(f"corrected: train_rows={tr_rows} != fm + ad = {cfgm['fm']} + {cfgm['ad']} "
                                      f"(las regiones de evaluacion solaparian el entrenamiento del modelo)")
    seed_calls = I.apply_seeds(spec.seed)
    cfg.update(device=device, seed_calls=seed_calls, code_hashes=I.code_hashes(code_root, RUNNER_DIR))
    C.write_json_atomic(tmp / "env.json", I.collect_env(code_root, device, a.test_mode))
    hb.set_phase("cache")
    t = time.time()
    x, y, cmeta = pipelines.load_stream(spec, cache_root=a.cache_root, data_root=a.data_root, code_root=code_root,
                                        test_mode=a.test_mode, max_rows=a.max_rows, overrides=overrides, log=log,
                                        cache_wait_s=a.cache_wait_s, hb=hb)
    phases["cache_s"] = time.time() - t
    cfg["cache"] = cmeta
    if not cmeta.get("cached"):
        import hashlib
        cfg["cache"]["y_inmemory_sha256"] = hashlib.sha256(y.tobytes()).hexdigest()
    log(f"stream {spec.dataset} {spec.pipeline}: x={x.shape} {x.dtype}, y={y.shape} ({phases['cache_s']:.1f} s)")
    t4e = None
    if spec.contamination_pct:
        hb.set_phase("t4e_inject")
        x, y, t4e, donor = I.t4e_inject(spec, x, y, cfgm, code_root, a.test_mode)
        import numpy as np
        np.save(tmp / "t4e_donor_rows.npy", donor)
        log(f"T4-E c={spec.contamination_pct} %: filas {t4e['injected_rows']} <- {t4e['B_c']} filas de ataque del pool "
            f"{t4e['donor_pool']}; excluir de E {t4e['eval_exclude_with_guards']} aguas abajo")
    if spec.mode == "probe":
        import smoke
        rows = C.read_manifest(a.manifest)
        res = smoke.run_probe(spec, x, device, tmp, hb, overrides, a.test_mode, results_root, rows, phases, log)
    else:
        ebs = I.execution_bs_override(spec, results_root)
        if ebs is not None:
            cfg["execution_bs_override"] = ebs
            log(f"EXECUTION_BS de {ebs['module']} = {ebs['value']} (antes {ebs['frozen']}): {ebs['reason']}")
        res = execute_kitnet(spec, x, y, cfgm, cmeta, device, tmp, hb, log, t4e=t4e, results_root=results_root)
    res.setdefault("phases", {}).update(phases)
    res["timing"]["device"] = device
    return res


REGIONS_CONTRACT = (
    "El runner NO fija la referencia del detector ni la region evaluada (la fija analysis_plan.md). Cada candidata "
    "es un PAR (calib, eval) disjunto. aggregate.py/metrics_rev.py DEBEN tomar calib y eval de UNA candidata con "
    "common.check_protocol_regions(regions, <candidata>), que exige calib ∩ eval = ∅ y eval ∩ train = ∅; nunca "
    "combinar la calib de una candidata con la eval de otra. 'post_train' NO es una region de evaluacion. Las "
    "candidatas regression_only (post-proceso tal como se envio) solo sirven para la regresion interna X2. Si la "
    "candidata trae eval_exclude o el job t4e_training_contamination (T4-E), la evaluacion son los SEGMENTOS que "
    "devuelve check_protocol_regions(..., with_exclusions=True); la referencia c = 0 usa extra_eval_exclude con "
    "common.t4e_eval_exclusion(dataset). F_cal excluye los duplicados de fit_set_exclusion (plan §2.1).")


def regions_for(spec, n, cfgm, cmeta, t4e=None):
    """Regions of the stream derived from the effective configuration of the model: disjoint (calibration, evaluation) candidates."""
    fm, ad = cfgm["fm"], cfgm["ad"]
    te = fm + ad
    reg = {"units": "indices de fila del stream (0-based, semiabiertos)", "n_rows": int(n),
           "fm": [0, min(fm, n)], "ae_train": [min(fm, n), min(te, n)], "execution": [min(te, n), n]}
    if spec.pipeline == "corrected":
        extra = cmeta.get("extra") or {}
        tr = list((extra.get("regions") or {}).get("train") or [])
        if tr != [0, te]:
            raise C.FrozenConfigError(f"regiones: train del stream {tr} != [0, fm + ad = {te})")
        if n <= te:
            raise C.FrozenConfigError(f"regiones: el stream ({n} filas) no tiene filas tras el entrenamiento ({te})")
        reg.update(train=[0, te], post_train=[te, n], train_benign_check=extra.get("train_benign_check"),
                   first_attack_index=extra.get("first_attack_index"))
        cands = {"_contract": REGIONS_CONTRACT}
        for pct in DETECTOR_CALIB_FRACS:
            c = int(math.floor(pct / 100.0 * (n - te)))
            cands[f"calib_first_{pct:02d}pct_post_train"] = {
                "calib": [te, te + c], "eval": [te + c, n], "calib_in_sample": False, "calib_uses_labels": False,
                "regression_only": False,
                "note": f"calibracion sin etiquetas con el primer {pct} % del tramo posterior al entrenamiento (en "
                        "coordenadas del modelo desde el final del entrenamiento); la evaluacion empieza DESPUES de "
                        "la calibracion"}
        ins0 = fm + ad // 3
        ref_note = ("misma referencia que el post-proceso enviado ([100k, 200k))" if [ins0, te] == [100_000, 200_000] else
                    f"[fm + floor(ad/3), T) = [{ins0}, {te}) (plan §2.9 S-INS; con este ad no coincide con la del envio)")
        cands["calib_in_sample_train_tail"] = {
            "calib": [ins0, te], "eval": [te, n], "calib_in_sample": True, "calib_uses_labels": False,
            "regression_only": False,
            "note": f"{ref_note}, pero evaluando solo [fm+ad, n). Scores IN-SAMPLE por partida doble: los AEs se "
                    "entrenaron con esas filas y la capa stat se ajusto con esos mismos errores (kitnet.py:176-185); "
                    "en las redes con ventana el error de entrenamiento usa relleno circular (conv1d.py:184-188). "
                    "Distribucion benigna optimista: NO es la referencia primaria salvo que analysis_plan.md lo fije "
                    "y lo declare"}
        reg["detector_protocol_candidates"] = cands
        reg["fit_set_exclusion"] = _fit_set_exclusion(spec, cmeta)
        if t4e is not None:
            ex = t4e["eval_exclude_with_guards"]
            cal10 = cands["calib_first_10pct_post_train"]
            if list(t4e["calib"]) != list(cal10["calib"]) or not (cal10["eval"][0] <= ex[0] < ex[1] <= cal10["eval"][1]):
                raise C.FrozenConfigError(f"T4-E: calib/pool {t4e['calib']}/{ex} no casan con la candidata del 10 % "
                                          f"{cal10} (analysis_plan.md §2.1, §8.3)")
            inj = t4e["injected_rows"]
            if not (te - (cfgm["ad"]) <= inj[0] < inj[1] <= te):
                raise C.FrozenConfigError(f"T4-E: bloque inyectado {inj} fuera del tramo AE [{cfgm['fm']}, {te})")
            reg["t4e_training_contamination"] = dict(t4e, train_benign_check_note=(
                "train_benign_check se hizo sobre el stream limpio ANTES de inyectar; el entrenamiento de este job "
                "contiene deliberadamente las filas de ataque de injected_rows (prueba de estres, no fuga)"))
            for k, cd in cands.items():
                if k.startswith("_"):
                    continue
                a, b = max(ex[0], cd["eval"][0]), min(ex[1], cd["eval"][1])
                if b > a:
                    cd["eval_exclude"] = [[int(a), int(b)]]
                    cd["eval_exclude_reason"] = ("T4-E: pool de donantes con guardas = filas con las que se entreno este "
                                                 "modelo; fuera de E aqui y en la referencia c = 0 (common."
                                                 "t4e_eval_exclusion; analysis_plan.md §8.5)")
        for k in cands:
            if not k.startswith("_"):
                try:
                    C.check_protocol_regions(reg, k, with_exclusions=True)
                except C.ProtocolRegionError as e:
                    raise C.FrozenConfigError(f"regiones generadas no validas: {e}")
    else:
        reg["detector_protocol_candidates"] = {
            "_contract": REGIONS_CONTRACT,
            "as_submitted_regression_only": {
                "calib": [100_000, 200_000], "eval": [0, int(n)], "calib_in_sample": True, "regression_only": True,
                "source": "post-proceso original (constantes de detectors_2025, no de cfgm)",
                "note": "FUGA DEL ENVIO: referencia dentro de la evaluacion y umbral con etiquetas de evaluacion. "
                        "Solo regresion interna X2; check_protocol_regions la rechaza sin allow_regression_only"}}
    return reg


def _fit_set_exclusion(spec, cmeta):
    """Pointer to the mask of exact training/execution duplicates that the calibration fit set must exclude (plan §2.1)."""
    extra = cmeta.get("extra") or {}
    dups = extra.get("duplicates_train_exec") or {}
    aux = (cmeta.get("aux") or {}).get("dup_train") or (cmeta.get("aux_inmemory") or {}).get("dup_train")
    out = {"rule": "F_cal = filas de la calibracion SIN las marcadas como duplicado exacto de una fila de [0, T) "
                   "(analysis_plan.md §2.1; los duplicados siguen en el stream, en las ventanas y en E)",
           "required": True, "fail_closed": "metrics_rev.py no ajusta nada si no obtiene la mascara"}
    if aux is not None:
        out.update(mask="cache aux_dup_train.npy (bool por fila)", path=aux.get("path"),
                   sha256=aux.get("sha256") or aux.get("sha256_bytes"), n_total=dups.get("n"),
                   by_calib_fraction=dups.get("by_calib_fraction"), reference=dups.get("reference"))
    elif spec.kind == "knad":
        out.update(mask=None, status="la cache KNAD corrected v1 no guarda la mascara (T0: 0 duplicados en los 6 "
                                     "streams; plan_support/label_counts.json calib.dup_of_train_rows)",
                   how="pipelines.train_duplicate_mask(x, T, warmup_audit.hash_matrix) sobre x de la cache")
    else:
        out.update(mask=None, status="sin mascara en la cache de este stream")
    return out


def execute_kitnet(spec, x, y, cfgm, cmeta, device, tmp, hb, log, t4e=None, results_root=None):
    import numpy as np
    import tensorflow as tf
    import instrument as I
    from src.models import KitNET
    if spec.variant in C.CONTEXT_VARIANTS:
        I.register_context_variants(spec)
    if spec.variant in C.EPS_VARIANTS:
        import networks_eps as _EPS
        _EPS.register_eps_variant()
    n = len(x)
    anomalies, early = [], {}
    rec_holder = {}
    grouping = {}
    regions = regions_for(spec, n, cfgm, cmeta, t4e=t4e)

    def on_fm_end(kn):
        rg = grouping.get("proxy")
        if rg is not None:
            kn.clustering_model = rg.real
            if rg.calls != 1 or kn.feature_map != rg.feature_map:
                raise C.FrozenConfigError(f"T3 random: cluster() llamado {rg.calls} veces o kn.feature_map distinto de la "
                                          f"particion inyectada")
            refs = I.t1a_feature_map_reference(results_root, spec) if results_root is not None else []
            bad = [j for j, fmap in refs if fmap != rg.dbscan_map]
            if bad:
                raise C.FrozenConfigError(f"T3 random: el mapa DBSCAN de este FM {[len(g) for g in rg.dbscan_map]} difiere "
                                          f"del de T1-A ({bad[:2]}); el plan fija los tamanos de T1-A (§7.3)")
            grouping["t1a_reference"] = {"checked_against": [j for j, _ in refs], "equal": not bad,
                                         "note": None if refs else "sin T1-A DONE de este dataset: tamanos del DBSCAN de "
                                                                   "este mismo FM (determinista con los mismos datos)"}
            log(f"T3 agrupacion aleatoria (partition_seed={rg.partition_seed}): tamanos {rg.info['sizes']} del DBSCAN; "
                f"T1-A comparados: {len(refs)}")
        I.check_aes(kn, cfgm, spec)
        try:
            early["fm_stats"] = I.fm_stats(rec_holder["rec"], kn, cfgm["k"], cfgm["fm"], grouping=rg)
            if rg is not None and grouping.get("t1a_reference"):
                early["fm_stats"]["random_grouping_t1a_reference"] = grouping["t1a_reference"]
            if early["fm_stats"]["consistency"].get("feature_map_equal") is False:
                anomalies.append("fm_stats (fila fm-1): replica de dbscan.py distinta de kn.feature_map")
        except Exception:
            anomalies.append("fm_stats (fila fm-1): " + traceback.format_exc()[-1500:])
        try:
            early["hyperparameters_effective"] = I.hyperparameters_effective(kn, cfgm, spec)
        except Exception:
            anomalies.append("hyperparameters_effective: " + traceback.format_exc()[-1500:])
        log(f"FM listo: grupos {[len(g) for g in kn.feature_map]}; AEs {type(kn.autoencoders[0]).__name__} "
            f"comprobados contra la config congelada")

    hb.set_phase("build", n)
    ckpt = None
    with tf.device(device):
        kn = KitNET(x.shape[-1], **I.kitnet_kwargs(x.shape[-1], cfgm, spec))
        kn_check = I.check_kitnet_config(kn, cfgm, spec)
        if spec.grouping == "random":
            grouping["proxy"] = I.RandomGroupingClusterer(kn.clustering_model, spec.partition_seed, kn.n_features)
            kn.clustering_model = grouping["proxy"]
        obs = I.LoopObserver(n, cfgm, hb, device, on_fm_end=on_fm_end)
        with I.DBSCANRecorder(target_k=cfgm["k"]) as rec:
            rec_holder["rec"] = rec
            obs.start()
            for _, packet in enumerate(x):
                kn.process(packet, is_last=False if _ < len(x) - 1 else True)
                obs.after(_, kn)
        hb.set_phase("outputs")
        t = time.time()
        scores = I.save_scores(kn, tmp)
        if scores.shape != (n,):
            raise RuntimeError(f"scores con forma {scores.shape} != ({n},)")
        n_nan, n_inf = int(np.isnan(scores).sum()), int(np.isinf(scores).sum())
        del scores
        try:
            arrays, outputs = I.capture_after_loop(kn, rec, obs, tmp, spec, cfgm, early.get("fm_stats"), anomalies)
        except Exception:
            arrays, outputs = {}, ["scores.npy"]
            anomalies.append("capture_after_loop: " + traceback.format_exc()[-1500:])
        try:
            if len(y) != n:
                raise ValueError(f"len(y)={len(y)} != n={n}")
            arrays["labels.npy"] = I.save_labels(y, tmp)
            arrays["labels.npy"]["alignment"] = ("orig KNAD: y[i] = etiqueta del paquete i y score[i] = paquete i+1 "
                                                 "(desfase +1 del envio)" if spec.pipeline == "orig" and spec.kind == "knad"
                                                 else "y[i] = etiqueta de la fila i del stream (alineadas)")
            if t4e is not None:
                arrays["labels.npy"]["t4e"] = (f"etiquetas del stream CONTAMINADO: filas {t4e['injected_rows']} = etiquetas "
                                               f"de las filas donantes (t4e_donor_rows.npy); limpias sha256 "
                                               f"{t4e['labels_clean_sha256'][:12]}")
            outputs.append("labels.npy")
        except Exception:
            anomalies.append("labels.npy: " + traceback.format_exc()[-1500:])
        if t4e is not None:
            try:
                arrays["t4e_donor_rows.npy"] = I.array_info(tmp / "t4e_donor_rows.npy")
                outputs.append("t4e_donor_rows.npy")
            except Exception:
                anomalies.append("t4e_donor_rows.npy: " + traceback.format_exc()[-1500:])
        outputs_s = time.time() - t
        ckpt_s = None
        if spec.save_ckpt:
            hb.set_phase("ckpt")
            I.gpu_peak_and_reset(device)
            try:
                ckpt = I.save_checkpoint(kn, tmp / "ckpt", spec.variant)
                ckpt_s = ckpt["seconds"]
            except Exception:
                ckpt = {"error": traceback.format_exc()[-3000:]}
                anomalies.append("checkpoint: KitNET.save fallo; los scores siguen siendo validos (M5 ii)")
            ckpt_gpu = I.gpu_peak_and_reset(device)
        else:
            ckpt_gpu = None
    anomalies.extend(obs.anomalies)
    try:
        C.write_json_atomic(tmp / "arrays.json", arrays)
    except Exception:
        anomalies.append("arrays.json: " + traceback.format_exc()[-1500:])
    try:
        ot = obs.timing()
    except Exception:
        ot = {"exec_rows_per_s": None, "exec_blocks": None, "gpu_peak_bytes": {}, "loop_total_s": None, "fm_s": None,
              "train_s": None, "exec_s": None}
        anomalies.append("obs.timing: " + traceback.format_exc()[-1500:])
    timing = {"rows": n, "exec_rows_per_s": ot["exec_rows_per_s"], "exec_blocks": ot["exec_blocks"],
              "gpu_peak_bytes": dict(ot["gpu_peak_bytes"] or {}, ckpt=ckpt_gpu), "loop_total_s": ot["loop_total_s"]}
    phases = {"fm_s": ot["fm_s"], "train_s": ot["train_s"], "exec_s": ot["exec_s"], "outputs_s": outputs_s,
              "ckpt_s": ckpt_s}
    try:
        fm_sizes = [len(g) for g in (kn.feature_map or [])]
    except Exception:
        fm_sizes = None
    config = {"hyperparameters_frozen": {k: v for k, v in cfgm.items()},
              "kitnet_config_check": kn_check,
              "hyperparameters_effective": early.get("hyperparameters_effective"),
              "launch_all_equivalence": {"constructor": "KitNET(x.shape[-1], feature_map=None, ad_grace_period, "
                                                        "execution_window, fm_grace_period, sequence_length, "
                                                        "hidden_ratio, autoencoder_size, clustering, ar, ae_type, "
                                                        "output_ae_type) = launch_all.py:54-65",
                                         "loop": "for _, packet in enumerate(x): kn.process(packet, is_last=False if "
                                                 "_ < len(x) - 1 else True) = launch_all.py:69-72",
                                         "scores": "np.log10(np.array(current_error) + 1) + np.save = kitnet.py:323-324,345"},
              "regions": regions,
              "score_row_semantics": I.S17_NOTES["score_row_semantics"],
              "scores_summary": {"n": n, "n_nan": n_nan, "n_inf": n_inf, "nan_is_result": True},
              "ckpt": ckpt, "feature_map_sizes": fm_sizes, "ablation": cfgm.get("ablation")}
    return {"config": config, "timing": timing, "phases": phases, "anomalies": anomalies, "outputs": outputs}


def run_fake(spec, tmp, hb, P, log):
    import numpy as np
    env = os.environ
    v = spec.variant
    anomalies = []
    hb.set_phase("fake", 10)

    def scores(arr):
        np.save(tmp / "scores.npy", np.asarray(arr, dtype=np.float64))

    if v == "_fake_ok":
        time.sleep(float(env.get("JISA_TEST_FAKE_SLEEP", "0.3")))
        scores(np.arange(10.0))
    elif v == "_fake_sleep":
        time.sleep(float(env.get("JISA_TEST_SLEEP_S", "8")))
        scores(np.arange(10.0))
    elif v == "_fake_fail":
        raise RuntimeError("fallo simulado (_fake_fail)")
    elif v == "_fake_nan":
        scores([0.0, float("nan"), 1.0])
    elif v == "_fake_kill":
        (tmp / "partial.txt").write_text("parcial\n")
        if env.get("JISA_TEST_OOM_KILL_FILE"):
            f = Path(env["JISA_TEST_OOM_KILL_FILE"])
            f.write_text(f"{C.oom_kill_counters(test_file=f)['test_file'] + 1}\n")
        if env.get("JISA_TEST_KILL_WITH_SCORES") == "1":
            scores([1.0, 2.0])
        if env.get("JISA_TEST_KILL_OK") != "1":
            os.kill(os.getpid(), signal.SIGKILL)
        scores(np.arange(10.0))
    elif v == "_fake_count":
        with open(env["JISA_TEST_COUNTER"], "a") as fh:
            fh.write(f"{spec.job_id} {os.getpid()}\n")
        time.sleep(float(env.get("JISA_TEST_FAKE_SLEEP", "0.2")))
        scores(np.arange(10.0))
    elif v == "_fake_oom":
        if env.get("JISA_TEST_OOM_OK") != "1":
            class ResourceExhaustedError(Exception):
                pass
            raise ResourceExhaustedError("OOM when allocating tensor with shape[512,19,800,800] (simulado)")
        scores(np.arange(10.0))
    elif v == "_fake_halt":
        raise C.PipelineError("el tramo de entrenamiento [0, 200000) contiene 1 etiquetas de ataque (simulado)")
    elif v == "_fake_scorefail":
        scores([3.0, 4.0])
        if env.get("JISA_TEST_SCOREFAIL_OK") != "1":
            raise RuntimeError("fallo simulado despues de escribir scores.npy")
    elif v == "_fake_ckptfail":
        scores(np.arange(10.0))
        try:
            raise OSError("No space left on device (checkpoint simulado)")
        except OSError:
            anomalies.append("checkpoint: KitNET.save fallo (simulado); los scores siguen siendo validos (M5 ii)")
    elif v == "_fake_dup":
        scores(np.arange(10.0))
        P.final.mkdir(parents=True, exist_ok=True)
        (P.final / "DONE").write_text("otro intento\n")
    elif v == "_fake_gpukernel":
        class InternalError(Exception):
            pass
        raise InternalError("Graph execution error: 'cuModuleLoadData(&module, fatbin.data())' failed with "
                            "'CUDA_ERROR_INVALID_IMAGE' [Op:Sign] (simulado)")
    else:
        raise C.InvalidJob(f"variante ficticia desconocida {v}")
    a = np.load(tmp / "scores.npy")
    C.write_json_atomic(tmp / "arrays.json", {"scores.npy": {"shape": list(a.shape), "dtype": str(a.dtype),
                                                             "n_nan": int(np.isnan(a).sum()),
                                                             "n_inf": int(np.isinf(a).sum()),
                                                             "sha256": C.sha256_file(tmp / "scores.npy")}})
    return {"config": {"fake": True, "scores_summary": {"n": int(a.shape[0]), "n_nan": int(np.isnan(a).sum())}},
            "timing": {"rows": int(a.shape[0])}, "phases": {}, "anomalies": anomalies}


if __name__ == "__main__":
    sys.exit(_entry())
