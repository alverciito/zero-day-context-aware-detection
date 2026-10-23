#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""worker.py - queue loop of the job runner: re-reads the manifest, claims the next job and launches run_job.py in a subprocess."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNNER_DIR))
import common as C

STATE = {"stopping": False, "child": None}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Worker de la cola del runner v1 (JISA). Ver DESIGN.md §4.")
    ap.add_argument("--gpu", required=True, help="Indice de GPU o 'cpu'.")
    ap.add_argument("--manifest", required=True, help="Ruta ABSOLUTA al manifest.csv del arbol principal.")
    ap.add_argument("--results-root", required=True, help="Ruta ABSOLUTA a results/revision del arbol principal.")
    ap.add_argument("--blocks", nargs="*", default=None, help="Filtra por bloque: b o b-* (p. ej. X2, S-w1, T1).")
    ap.add_argument("--once", action="store_true", help="Sale con 0 cuando no queda nada reclamable.")
    ap.add_argument("--max-jobs", type=int, default=0)
    ap.add_argument("--stale-heartbeat-min", type=float, default=15.0)
    ap.add_argument("--logs-root", default=str(C.LOGS_ROOT))
    ap.add_argument("--code-root", default=str(C.DEFAULT_CODE_ROOT))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--dirty-wait-s", type=float, default=None)
    ap.add_argument("--idle-sleep-s", type=float, default=60.0)
    ap.add_argument("--dirty-sleep-s", type=float, default=60.0)
    ap.add_argument("--env-sleep-s", type=float, default=300.0)
    ap.add_argument("--progress-every-s", type=float, default=600.0)
    ap.add_argument("--claim-timeout-s", type=float, default=900.0,
                    help="Espera maxima del flock de reclamacion del host (serializa elegir + lanzar entre workers).")
    ap.add_argument("--claim-wait-s", type=float, default=180.0,
                    help="Tras lanzar el hijo, espera hasta que cree su lock (o termine) antes de soltar la reclamacion.")
    ap.add_argument("--threads", type=int, default=None,
                    help="Threads por job en CPU: TF_NUM_INTRAOP_THREADS=N, TF_NUM_INTEROP_THREADS=max(1, min(2, "
                         "N)), OMP_NUM_THREADS=N en cada hijo. Sin la opcion no se toca el entorno heredado.")
    ap.add_argument("--test-mode", action="store_true", help="SOLO tests (pasa --test-mode a run_job).")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--test-override", default="")
    return ap.parse_args(argv)


class WLog:
    def __init__(self, logs_root, gpu):
        self.host = C.hostname()
        self.gpu = gpu
        Path(logs_root).mkdir(parents=True, exist_ok=True)
        self.path = Path(logs_root) / f"worker-{self.host}-{gpu}.log"
        self.fh = open(self.path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"{C.now_iso()} [worker {os.getpid()} gpu={self.gpu}] {msg}"
        print(line, flush=True)
        try:
            self.fh.write(line + "\n")
            self.fh.flush()
        except OSError:
            pass


def on_signal(signum, frame):
    STATE["stopping"] = True
    ch = STATE["child"]
    if ch is not None and ch.poll() is None:
        try:
            ch.send_signal(signal.SIGTERM)
        except OSError:
            pass


def thread_vars(threads):
    """Thread environment variables for --threads N (None leaves the inherited environment untouched)."""
    if threads is None:
        return {}
    n = int(threads)
    if n < 1:
        raise ValueError(f"--threads debe ser >= 1 (recibido {threads})")
    return {"TF_NUM_INTRAOP_THREADS": str(n), "TF_NUM_INTEROP_THREADS": str(max(1, min(2, n))),
            "OMP_NUM_THREADS": str(n), "JISA_WORKER_THREADS": str(n)}


def _threads_text(tvars):
    if tvars:
        return ", ".join(f"{k}={v}" for k, v in tvars.items())
    inherited = ", ".join(f"{k}={os.environ.get(k)}" for k in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                                                              "OMP_NUM_THREADS"))
    return f"heredados ({inherited})"


def child_env(row, gpu, worker_pid, threads=None):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="" if gpu == "cpu" else str(gpu), TF_FORCE_GPU_ALLOW_GROWTH="true",
               TF_CPP_MIN_LOG_LEVEL="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               LD_LIBRARY_PATH="", JISA_WORKER_PID=str(worker_pid))
    env.update(thread_vars(threads))
    env.pop("JISA_SEED_REEXEC", None)
    seed = row.get("seed")
    if seed in C.SEEDS:
        env["PYTHONHASHSEED"] = str(C.SEEDS[seed])
    else:
        env.pop("PYTHONHASHSEED", None)
    return env


def choose_job(a, R, rows, invalid, log, host):
    """Devuelve (fila, motivo_si_ninguna)."""
    stale_s = a.stale_heartbeat_min * 60.0
    live = C.live_locks_on_host(R, host)
    tr_live = [j for j, info in live if int(info.get("timing_ref", 0) or 0) == 1]
    if tr_live:
        return None, f"timing_ref activo ({tr_live[0]}): no se reclama nada (S11)"
    at = C.now()
    reason = "nada reclamable"
    by_jid = {x["job_id"]: x for x in rows}
    live_variants = {}
    for j, _info in live:
        v = (by_jid.get(j) or {}).get("variant")
        if v:
            live_variants[v] = live_variants.get(v, 0) + 1
    for r in sorted(rows, key=lambda r: (int(r["priority"]) if r["priority"].lstrip("-").isdigit() else 10**9, r["job_id"])):
        jid = r["job_id"]
        if not C.block_selected(r["block"], a.blocks) or jid in invalid:
            continue
        if (R / jid / "DONE").exists():
            continue
        if any((R / ".state" / jid).glob("nonexec_*.json")):
            continue
        C.account_orphan_moves(R, jid, log)
        if C.read_fail_count(R, jid) >= C.MAX_FAILS:
            continue
        L = R / ".locks" / jid
        if L.exists():
            st = C.lock_status(L, host)
            age = st["hb_age_s"] if st["hb_age_s"] is not None else 0
            if st["same_host"] and st["alive"] is False and age > stale_s:
                dest = C.recover_lock(R, jid, st["pid"], st["start_ticks"],
                                      reason=f"PID {st['pid']} muerto, mismo host, latido hace {age:.0f} s", log=log)
                if dest is None:
                    continue
                n = C.account_lock_move(R, jid, dest, f"proceso muerto sin registrar (PID {st['pid']}, latido hace "
                                                      f"{age:.0f} s > {stale_s:.0f} s; posible Killed/OOM del sistema)")
                log(f"lock obsoleto de {jid} movido a {dest.name}; error_{n}.txt, fail_count={C.read_fail_count(R, jid)}")
                if C.read_fail_count(R, jid) >= C.MAX_FAILS or L.exists():
                    continue
            else:
                continue
        if C.parse_bool(r.get("timing_ref", "0")) and live:
            return None, f"{jid} es timing_ref y hay {len(live)} job(s) vivos en el host: espera exclusividad (S11)"
        cap = C.EXCLUSIVE_VARIANTS.get(r["variant"])
        if cap is not None and live_variants.get(r["variant"], 0) >= cap:
            reason = (f"{jid}: la variante {r['variant']} es exclusiva por memoria de GPU (max {cap} simultaneo(s), "
                      f"{live_variants[r['variant']]} vivo(s) en el host)")
            continue
        if at >= C.FREEZE_DAY_START:
            ok, why = C.fits_before_freeze(r, R, rows, at)
            if not ok:
                reason = f"{jid} no cabe antes de la congelacion: {why}"
                continue
        return r, None
    return None, reason


def ensure_failure_recorded(R, jid, rc, jl, fc_before, log, unpersisted, invalid, text=None):
    """Record the failure of a child that ended as failed without raising fail_count; True if it was persisted."""
    try:
        if C.read_fail_count(R, jid) > fc_before:
            return True
    except Exception:
        pass
    tail = C.tail_text(jl, 4000)
    msg = text or (f"run_job rc={rc} ({C.CODE_NAMES.get(rc, C.signal_name(rc) or 'desconocido')}) sin fallo registrado "
                   f"(excepcion fuera de su captura o .state no escribible); log {jl}\n--- cola del log ---\n{tail[-2500:]}")
    try:
        n = C.record_failure(R, jid, msg)
        if C.read_fail_count(R, jid) > fc_before:
            log(f"{jid}: fallo registrado por el worker (error_{n}.txt, fail_count={C.read_fail_count(R, jid)})")
            return True
    except Exception as e:
        log(f"ERROR: no se pudo registrar el fallo de {jid} en .state ({type(e).__name__}: {e})")
    unpersisted[jid] = unpersisted.get(jid, 0) + 1
    log(f"ERROR: fallo de {jid} NO persistido ({unpersisted[jid]}/{C.MAX_FAILS} en memoria)")
    if unpersisted[jid] >= C.MAX_FAILS:
        invalid.add(jid)
        log(f"ERROR: {jid} excluido en esta sesion tras {unpersisted[jid]} fallos sin poder escribir .state (requiere persona)")
    return False


def handle_oom_kill(R, jid, child_pid, rc, jl, host, evidence, log):
    """Hijo muerto por SIGKILL con el OOM killer confirmado (evidence = incremento de common.oom_kill_counters). Mueve
    su lock a .state y deja oom_N.json (source=oom_killer). Sin otros jobs vivos en el host: suma UN fallo (el lock
    movido se contabiliza con account_lock_move o, si no habia lock, con record_failure). Con concurrencia: no suma
    (lock marcado 'accounted' sin fallo). Devuelve counted (bool)."""
    others = C.live_locks_on_host(R, host, exclude_job=jid)
    counted = len(others) == 0
    tail = C.tail_text(jl, 3000)
    text = (f"OOM killer del kernel: run_job rc={rc} {C.signal_name(rc)} pid={child_pid}; contadores {evidence}; "
            f"log {jl}\n--- cola del log ---\n{tail[-2000:]}")
    moved, n = None, None
    L = R / ".locks" / jid
    try:
        info = C.read_lock_info(L) or {}
        if info and str(info.get("pid")) == str(child_pid):
            moved = C.recover_lock(R, jid, child_pid, info.get("start_ticks"), reason=f"OOM killer rc={rc}", log=log)
    except Exception as e:
        log(f"ERROR recuperando el lock de {jid} tras el OOM killer: {type(e).__name__}: {e}")
    if counted:
        n = C.account_lock_move(R, jid, moved, "OOM\n" + text) if moved is not None else None
        if n is None:
            n = C.record_failure(R, jid, "OOM\n" + text)
    elif moved is not None:
        try:
            os.mkdir(moved / "accounted")
            C.write_json_atomic(moved / "accounted" / "by.json", {"pid": os.getpid(), "host": host, "ts": C.now_iso(),
                                                                  "error_n": None, "reason": "OOM con concurrencia"})
        except OSError:
            pass
    oom = {"ts": C.now_iso(), "job_id": jid, "source": "oom_killer", "rc": rc, "signal": C.signal_name(rc),
           "counted": counted, "error_n": n, "concurrent_jobs_same_host": [j for j, _ in others],
           "oom_kill_counter_delta": evidence, "lock_moved_to": str(moved) if moved is not None else None,
           "log": str(jl),
           "oom_chain": {"paso_0": "descartar causas ajenas: con concurrencia > 1 se baja w (no suma fallo)",
                         "paso_1": "T2 MHA L = 2000: lote de inferencia reducido (instrument.execution_bs_override, §7.2)",
                         "paso_2": "activation checkpointing fp32: cambio de runner (tag nuevo)",
                         "paso_3": "celda no ejecutable; nunca AMP"}}
    path = C.record_event(R, jid, "oom", oom, ext="json")
    log(f"ERROR OOM (OOM killer del kernel) en {jid}: rc={rc}; {'suma fallo' if counted else 'con concurrencia: no suma'}"
        f" (fail_count={C.read_fail_count(R, jid)}) -> {path}")
    return counted


def handle_dead_child(R, jid, child_pid, rc, jl, fc_before, log, unpersisted, invalid):
    tail = C.tail_text(jl, 4000)
    oom = C.looks_like_oom(tail)
    text = (f"run_job muerto sin registrar: rc={rc} {C.signal_name(rc)} pid={child_pid}"
            f"{' (patrones de OOM en el log)' if oom else ''}; log {jl}\n--- cola del log ---\n{tail[-2500:]}")
    L = R / ".locks" / jid
    try:
        info = C.read_lock_info(L) or {}
        if info and str(info.get("pid")) == str(child_pid):
            dest = C.recover_lock(R, jid, child_pid, info.get("start_ticks"), reason=f"hijo muerto rc={rc}", log=log)
            if dest is not None:
                n = C.account_lock_move(R, jid, dest, text)
                log(f"ERROR: {jid} murio (rc={rc} {C.signal_name(rc)}); lock movido a {dest.name}; error_{n}.txt")
                if n is not None:
                    return
    except Exception as e:
        log(f"ERROR recuperando el lock de {jid}: {type(e).__name__}: {e}")
    if ensure_failure_recorded(R, jid, rc, jl, fc_before, log, unpersisted, invalid, text=text):
        log(f"ERROR: {jid} murio (rc={rc} {C.signal_name(rc)}); fail_count={C.read_fail_count(R, jid)}")


def wait_child_lock(R, jid, p, timeout_s):
    """Espera a que el hijo cree .locks/<jid> con su PID (o termine). Devuelve 'lock', 'exited' o 'timeout'."""
    L = R / ".locks" / jid
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            return "exited"
        info = C.read_lock_info(L) or {}
        if str(info.get("pid")) == str(p.pid):
            return "lock"
        time.sleep(0.1)
    return "timeout"


def main(argv=None):
    a = parse_args(argv)
    log = WLog(a.logs_root, a.gpu)
    host = log.host
    rc_exit = 0
    try:
        rc_exit = _main(a, log, host)
    finally:
        others = C.other_live_workers(a.gpu, exclude_pid=os.getpid())
        if others:
            log(f"fin del worker con codigo {rc_exit} (quedan workers vivos en gpu={a.gpu}: {others})")
        else:
            log(f"[worker] EXIT={rc_exit}")
    return rc_exit


def _main(a, log, host):
    if os.environ.get("LD_LIBRARY_PATH"):
        log("ENV: LD_LIBRARY_PATH no esta vacio; lanza con LD_LIBRARY_PATH=")
        return C.ENV
    for p, what in ((a.manifest, "--manifest"), (a.results_root, "--results-root"), (a.code_root, "--code-root")):
        if not os.path.isabs(p):
            log(f"ERROR: {what} debe ser absoluta: {p}")
            return C.INVALID
    try:
        tvars = thread_vars(a.threads)
    except ValueError as e:
        log(f"ERROR: {e}")
        return C.INVALID
    R = Path(a.results_root)
    R.mkdir(parents=True, exist_ok=True)
    jobs_logs = Path(a.logs_root) / "jobs"
    jobs_logs.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    log(f"inicio: host={host} manifest={a.manifest} results={R} blocks={a.blocks} code_root={a.code_root} "
        f"run_job={RUNNER_DIR / 'run_job.py'} once={a.once} test_mode={a.test_mode} threads={_threads_text(tvars)}")
    invalid, env_streak, done_n, last_idle_log = set(), 0, 0, 0.0
    unpersisted = {}
    oom_test_file = os.environ.get("JISA_TEST_OOM_KILL_FILE") if a.test_mode else None
    oom_before = None
    while not STATE["stopping"]:
        if C.now() > C.FREEZE:
            log("computo congelado (30 sep 23:59): no se reclaman jobs")
            return C.OK
        try:
            rows = C.read_manifest(a.manifest)
        except Exception as e:
            log(f"ERROR leyendo el manifiesto: {e}")
            if a.once:
                return C.INVALID
            time.sleep(a.idle_sleep_s)
            continue
        job, why, p, fh, jl, fc_before = None, "sin reclamacion", None, None, None, 0
        with C.host_claim(R, host, timeout_s=a.claim_timeout_s, should_stop=lambda: STATE["stopping"]) as claimed:
            if not claimed:
                why = f"sin flock de reclamacion del host tras {a.claim_timeout_s:.0f} s"
            else:
                job, why = choose_job(a, R, rows, invalid, log, host)
                if job is not None:
                    jid = job["job_id"]
                    jl = jobs_logs / f"{jid}.{C.ts_compact()}.{os.getpid()}.log"
                    cmd = [sys.executable, "-u", str(RUNNER_DIR / "run_job.py"), "--job-id", jid, "--manifest",
                           a.manifest, "--results-root", str(R), "--gpu", a.gpu, "--code-root",
                           str(Path(a.code_root).resolve())]
                    for flag, val in (("--data-root", a.data_root), ("--cache-root", a.cache_root),
                                      ("--dirty-wait-s", a.dirty_wait_s)):
                        if val is not None:
                            cmd += [flag, str(val)]
                    if a.test_mode:
                        cmd.append("--test-mode")
                        if a.max_rows is not None:
                            cmd += ["--max-rows", str(a.max_rows)]
                        if a.test_override:
                            cmd += ["--test-override", a.test_override]
                    fc_before = C.read_fail_count(R, jid)
                    oom_before = C.oom_kill_counters(test_file=oom_test_file)
                    log(f"lanzo {jid} (prio {job['priority']}, fail_count={fc_before}) -> log {jl}")
                    fh = open(jl, "a")
                    p = subprocess.Popen(cmd, env=child_env(job, a.gpu, os.getpid(), a.threads), stdout=fh,
                                         stderr=subprocess.STDOUT, cwd=str(RUNNER_DIR))
                    STATE["child"] = p
                    w = wait_child_lock(R, jid, p, a.claim_wait_s)
                    if w == "timeout":
                        log(f"{jid}: el hijo no creo su lock en {a.claim_wait_s:.0f} s (espera de worktree o guardas); "
                            f"se suelta la reclamacion")
        if job is None:
            if STATE["stopping"]:
                break
            if a.once:
                log(f"--once: {why}; salgo")
                return C.OK
            if time.time() - last_idle_log >= min(a.progress_every_s, 600):
                log(f"en espera: {why}")
                last_idle_log = time.time()
            time.sleep(a.idle_sleep_s)
            continue
        try:
            last = time.time()
            while True:
                try:
                    rc = p.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if time.time() - last >= a.progress_every_s:
                        st = C.lock_status(R / ".locks" / jid, host)
                        age = st["hb_age_s"]
                        log(f"job {jid} activo: pid={p.pid} latido hace {age if age is None else round(age)} s; "
                            f"{st['heartbeat'][:160]}")
                        last = time.time()
        finally:
            STATE["child"] = None
            fh.close()
        name = C.CODE_NAMES.get(rc, f"rc={rc}")
        log(f"fin {jid}: {name} (rc={rc})")
        if rc != C.ENV:
            env_streak = 0
        if rc == C.OK:
            done_n += 1
        elif rc in (C.LOCKED, C.BLOCKED, C.DUPLICATE):
            pass
        elif rc in (C.INVALID, C.ANOMALY):
            invalid.add(jid)
            log(f"ERROR: {jid} {name}: excluido en esta sesion (ver {jl})")
        elif rc == C.DIRTY:
            time.sleep(a.dirty_sleep_s)
        elif rc == C.INTERRUPTED:
            if STATE["stopping"]:
                return C.INTERRUPTED
        elif rc == C.ENV:
            env_streak += 1
            log(f"ENV ({env_streak}/3): {C.tail_text(jl, 600).strip().splitlines()[-1:]}")
            if env_streak >= 3:
                return C.ENV
            time.sleep(a.env_sleep_s)
        elif rc == C.HALT:
            log(f"ERROR HALT en {jid}: error sistematico de stream/cache/config; el worker se detiene (ver .state/{jid}/halt_*.txt)")
            return C.HALT
        elif rc == C.OOM:
            st = R / ".state" / jid
            ooms = sorted(st.glob("oom_*.json"), key=lambda p: p.stat().st_mtime)
            last_oom = C.read_json(ooms[-1], {}) if ooms else {}
            if not last_oom.get("counted", True):
                log(f"ERROR OOM en {jid} con concurrencia {last_oom.get('concurrent_jobs_same_host')}: salgo para bajar w "
                    f"(paso 0 de la cadena OOM)")
                return C.OOM
            log(f"ERROR OOM en {jid} sin concurrencia (fail_count={C.read_fail_count(R, jid)}): ver oom_chain del evento")
        elif rc == C.FAILED:
            persisted = ensure_failure_recorded(R, jid, rc, jl, fc_before, log, unpersisted, invalid)
            log(f"{jid} FAILED (fail_count={C.read_fail_count(R, jid)}): {C.tail_text(jl, 400).strip()[-300:]}")
            if not persisted:
                time.sleep(min(a.dirty_sleep_s, 60.0))
        elif rc < 0 and STATE["stopping"] and -rc in (signal.SIGTERM, signal.SIGINT):
            log(f"{jid}: hijo terminado por la senal reenviada al parar el worker (rc={rc}); no suma fallo")
            try:
                C.record_event(R, jid, "interrupted", f"hijo terminado por senal al parar el worker (rc={rc})")
            except Exception:
                pass
            return C.INTERRUPTED
        else:
            evidence = C.oom_kill_delta(oom_before, C.oom_kill_counters(test_file=oom_test_file)) \
                if rc == -signal.SIGKILL else {}
            if evidence:
                if not handle_oom_kill(R, jid, p.pid, rc, jl, host, evidence, log):
                    log(f"ERROR OOM (OOM killer) en {jid} con concurrencia: salgo para bajar w (paso 0 de la cadena OOM)")
                    return C.OOM
            else:
                handle_dead_child(R, jid, p.pid, rc, jl, fc_before, log, unpersisted, invalid)
        if a.max_jobs and done_n >= a.max_jobs:
            log(f"--max-jobs {a.max_jobs} alcanzado")
            return C.OK
    return C.INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
