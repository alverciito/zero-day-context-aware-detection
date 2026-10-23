#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(a) Locks, DONE, .tmp, renombrado atomico, concurrencia y semantica de fallo del runner v1 (variantes ficticias
_fake_* que solo existen con --test-mode y no importan TF). CPU, < 2 min. DESIGN §12 (a) + M4, M5, S1, S3, S4, S6, S11.
"""
import importlib.util
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _util as U
from _util import C


def fake(variant, priority=1, **kw):
    return U.make_row(variant=variant, priority=priority, **kw)


def main():
    ck = U.Checker("test_a_locks")
    root = U.make_root("a")
    R = root / "results"
    man = root / "manifest.csv"
    try:
        lock = R / ".locks" / "A1"
        code = ("import sys,time; sys.path.insert(0, sys.argv[2]); import common as C; from pathlib import Path; "
                "r = C.acquire_lock(Path(sys.argv[1]), {'job_id': 'A1'}); print('WIN' if r else 'LOSE', flush=True); "
                "time.sleep(1.5)")
        procs = [subprocess.Popen([U.PY, "-c", code, str(lock), str(U.RUNNER_DIR)], stdout=subprocess.PIPE, text=True,
                                  env=U.base_env()) for _ in range(16)]
        outs = [p.communicate(timeout=60)[0].strip() for p in procs]
        ck.check("a1", outs.count("WIN") == 1 and outs.count("LOSE") == 15, f"WIN={outs.count('WIN')} LOSE={outs.count('LOSE')}")
        info = C.read_lock_info(lock) or {}
        ck.check("a1.info", list(info)[:2] == ["pid", "host"], f"info.json empieza por pid y host (estado.sh:211-213): {list(info)[:3]}")

        rows = []
        r_ok = fake("_fake_ok", 10)
        r_kill = fake("_fake_kill", 11)
        r_kill2 = fake("_fake_kill", 12, rep=2)
        r_fail = fake("_fake_fail", 13)
        r_nan = fake("_fake_nan", 14)
        r_sf = fake("_fake_scorefail", 15)
        r_ckf = fake("_fake_ckptfail", 16)
        r_halt = fake("_fake_halt", 17)
        r_oom = fake("_fake_oom", 18)
        r_oom2 = fake("_fake_oom", 19, rep=2)
        r_sleep = fake("_fake_sleep", 5)
        r_sleep_tr = fake("_fake_sleep", 6, rep=2, timing_ref=1)
        r_dup = fake("_fake_dup", 20)
        r_orphan = fake("_fake_ok", 21, rep=3)
        counts = [fake("_fake_count", 100 + i, rep=i + 1) for i in range(8)]
        r_prodrow = fake("_fake_ok", 30, rep=9)
        rows = [r_ok, r_kill, r_kill2, r_fail, r_nan, r_sf, r_ckf, r_halt, r_oom, r_oom2, r_sleep, r_sleep_tr, r_dup,
                r_orphan, r_prodrow] + counts
        U.write_manifest(man, rows)

        def sub_manifest(name, rs):
            p = root / name
            U.write_manifest(p, rs)
            return p

        rc, out = U.run_job(root, man, r_ok["job_id"])
        P = C.JobPaths(R, r_ok["job_id"])
        tmp_left = [t for t in (R / ".tmp").glob(r_ok["job_id"] + ".*")] if (R / ".tmp").exists() else []
        ck.check("a2.first", rc == 0 and P.done.exists() and not tmp_left and not P.lock.exists(),
                 f"rc={rc} DONE={P.done.exists()} tmp={len(tmp_left)} lock={P.lock.exists()}")
        mt = {p: p.stat().st_mtime_ns for p in P.final.rglob("*")}
        rc2, _ = U.run_job(root, man, r_ok["job_id"])
        mt2 = {p: p.stat().st_mtime_ns for p in P.final.rglob("*")}
        ck.check("a2.second", rc2 == 0 and mt == mt2, f"rc={rc2}; mtimes sin cambios={mt == mt2}")
        cfg = U.load_json(P.final / "config.json")
        ck.check("a2.code_root", cfg["code"]["runner_dir"] == str(U.RUNNER_DIR) and
                 Path(cfg["code"]["code_root"]) == U.CODE_ROOT.resolve(), f"code={cfg['code']['code_root']}")
        t = U.load_json(P.final / "timing.json")
        ck.check("a2.timing", list(t)[0] == "wall_seconds" and t["wall_seconds"] > 0, f"primera clave {list(t)[0]}")

        env_k = U.base_env(JISA_TEST_KILL_WITH_SCORES=1)
        rc, out = U.run_job(root, man, r_kill["job_id"], env=env_k)
        Pk = C.JobPaths(R, r_kill["job_id"])
        tmpk = list((R / ".tmp").glob(r_kill["job_id"] + ".*"))
        ck.check("a3.kill", rc == -signal.SIGKILL and Pk.lock.exists() and not Pk.final.exists() and len(tmpk) == 1
                 and (tmpk[0] / "scores.npy").exists(), f"rc={rc} lock={Pk.lock.exists()} tmp={[t.name for t in tmpk]}")
        time.sleep(2.5)
        mk = sub_manifest("m_a3.csv", [r_kill])
        rcw, outw = U.run_worker(root, mk, ["--stale-heartbeat-min", "0.02"], env=U.base_env(JISA_TEST_KILL_OK=1))
        moved = sorted(Pk.state.glob("lock_obsoleto_*"))
        attempts = sorted(Pk.state.glob("attempt_*"))
        ck.check("a3.recover", rcw == 0 and len(moved) == 1 and (moved[0] / "accounted").exists() and
                 C.read_fail_count(R, r_kill["job_id"]) == 1 and (Pk.state / "error_1.txt").exists(),
                 f"rc={rcw} movidos={[m.name for m in moved]} fail_count={C.read_fail_count(R, r_kill['job_id'])}")
        ck.check("a3.attempt_kept", Pk.done.exists() and len(attempts) == 1 and (attempts[0] / "scores.npy").exists(),
                 f"DONE={Pk.done.exists()} attempts={[a.name for a in attempts]}")
        cfgk = U.load_json(Pk.final / "config.json") if Pk.final.exists() else {}
        pa = cfgk.get("previous_attempts", {})
        ck.check("a3.previous_attempts", any(x.get("has_scores") for x in pa.get("in_state", [])) and
                 any(x.get("action") == "moved" for x in pa.get("adopted_now", [])), f"{pa}")

        mk2 = sub_manifest("m_a3b.csv", [r_kill2])
        rcw, outw = U.run_worker(root, mk2)
        fc = C.read_fail_count(R, r_kill2["job_id"])
        errs = sorted(C.JobPaths(R, r_kill2["job_id"]).state.glob("error_*.txt"))
        ck.check("a3b.dead_child", rcw == 0 and fc == 3 and len(errs) == 3 and "SIGKILL" in errs[0].read_text() and
                 not (R / ".locks" / r_kill2["job_id"]).exists(), f"rc={rcw} fail_count={fc} errores={len(errs)}")

        mf = sub_manifest("m_a4.csv", [r_fail])
        rcw, outw = U.run_worker(root, mf)
        Pf = C.JobPaths(R, r_fail["job_id"])
        errs = sorted(Pf.state.glob("error_*.txt"))
        ck.check("a4.blocked", rcw == 0 and C.read_fail_count(R, r_fail["job_id"]) == 3 and len(errs) == 3 and
                 not Pf.final.exists(), f"rc={rcw} fail_count={C.read_fail_count(R, r_fail['job_id'])} errores={len(errs)}")
        rc, _ = U.run_job(root, man, r_fail["job_id"])
        ck.check("a4.rc3", rc == C.BLOCKED, f"rc={rc}")

        rc, _ = U.run_job(root, man, r_nan["job_id"])
        Pn = C.JobPaths(R, r_nan["job_id"])
        arr = U.load_json(Pn.final / "arrays.json") if Pn.final.exists() else {}
        ck.check("a5.nan", rc == 0 and arr.get("scores.npy", {}).get("n_nan", 0) > 0 and
                 not (Pn.state / "fail_count").exists(), f"rc={rc} n_nan={arr.get('scores.npy', {}).get('n_nan')}")

        counter = root / "counter.txt"
        m6 = sub_manifest("m_a6.csv", counts)
        env6 = U.base_env(JISA_TEST_COUNTER=counter, JISA_TEST_FAKE_SLEEP=0.4)
        ws = [subprocess.Popen(U.worker_cmd(root, m6), env=env6, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
              for _ in range(2)]
        wout = [w.communicate(timeout=180) for w in ws]
        lines = counter.read_text().splitlines() if counter.exists() else []
        done8 = sum((R / r["job_id"] / "DONE").exists() for r in counts)
        cids = {r["job_id"] for r in counts}
        locks_left = [l.name for l in (R / ".locks").iterdir() if l.name in cids] if (R / ".locks").exists() else []
        ck.check("a6.concurrency", all(w.returncode == 0 for w in ws) and done8 == 8 and len(lines) == 8 and
                 len({ln.split()[0] for ln in lines}) == 8 and not locks_left,
                 f"rc={[w.returncode for w in ws]} DONE={done8} lineas={len(lines)} locks={locks_left}")

        gr = root / "gitrepo"
        gr.mkdir()
        g = lambda *a: subprocess.run(["git", "-C", str(gr), *a], capture_output=True, text=True)
        g("init", "-q")
        (gr / "f.txt").write_text("v1\n")
        g("add", "f.txt")
        g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
        (gr / "f.txt").write_text("v2\n")
        r7 = fake("_fake_ok", 40, rep=7)
        m7 = sub_manifest("m_a7.csv", [r7])
        rc, out = U.run_job(root, m7, r7["job_id"], ["--force-git-check", "--dirty-wait-s", "2", "--dirty-sleep-s", "1",
                                                     "--code-root", gr])
        ck.check("a7.dirty", rc == C.DIRTY and C.read_fail_count(R, r7["job_id"]) == 0 and "worktree sucio" in out,
                 f"rc={rc} fail_count={C.read_fail_count(R, r7['job_id'])}")
        g("checkout", "--", "f.txt")
        (gr / "untracked.txt").write_text("x\n")
        rc, out = U.run_job(root, m7, r7["job_id"], ["--force-git-check", "--dirty-wait-s", "2", "--dirty-sleep-s", "1",
                                                     "--code-root", gr])
        ck.check("a7.untracked_ok", rc == 0, f"rc={rc} (fichero no versionado no ensucia)")

        rc, out = U.run_job(root, man, r_prodrow["job_id"], test_mode=False)
        ck.check("a8.fake_without_test_mode", rc == C.INVALID, f"rc={rc}")
        cmd = U.job_cmd(root, man, r_prodrow["job_id"])
        cmd[cmd.index("--results-root") + 1] = str(C.RESULTS_ROOT / "_nunca")
        p = subprocess.run(cmd, env=U.base_env(), capture_output=True, text=True)
        ck.check("a8.test_mode_prod_results", p.returncode == C.ENV and not (C.RESULTS_ROOT / "_nunca").exists(),
                 f"rc={p.returncode}")
        cmd = U.job_cmd(root, man, r_prodrow["job_id"])
        cmd[cmd.index("--cache-root") + 1] = str(C.CACHE_ROOT)
        p = subprocess.run(cmd, env=U.base_env(), capture_output=True, text=True)
        ck.check("a8.test_mode_prod_cache", p.returncode == C.ENV, f"rc={p.returncode} (M3)")
        p = subprocess.run([U.SYS_PY, str(U.RUNNER_DIR / "make_manifest.py"), "--check"], capture_output=True, text=True)
        ck.check("a8.manifest_check", p.returncode == 0, p.stdout.strip()[:160])
        spec = importlib.util.spec_from_file_location("gates", U.REPO / ".claude/skills/lanzar-simulacion/scripts/gates.py")
        gates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gates)
        prod_rows = C.read_manifest(U.RUNNER_DIR / "manifest.csv")
        header = list(prod_rows[0].keys())
        ck.check("a8.gates_cols", all(c in header for c in gates.MANIFEST_COLS) and C.GATES_MANIFEST_COLS == gates.MANIFEST_COLS,
                 "columnas de gates.py presentes y copia sincronizada")
        import make_manifest as MM
        v2 = MM.ACTIVE == "v2"
        fams = {gates.block_family(r["block"]) for r in prod_rows}
        exp_fams = {"X2", "S", "T1", "T2", "T3", "T4"} if v2 else {"X2", "S", "T1"}
        ck.check("a8.gates_families", fams == exp_fams and all(C.KNAD_ATTACKS for _ in [0]),
                 f"familias segun gates.block_family: {sorted(fams)} (ACTIVE={MM.ACTIVE})")
        ids = [r["job_id"] for r in prod_rows]
        bad = []
        for r in prod_rows:
            try:
                s = C.validate_row(r, test_mode=False)
                off_knad = not gates.KNAD_RE.search(r["dataset"]) and (not v2 or r["pipeline"] != "corrected")
                if s.is_fake or (s.variant == "stat" and s.save_ckpt) or off_knad:
                    bad.append(r["job_id"])
            except C.InvalidJob as e:
                bad.append(f"{r['job_id']}: {e}")
        n_exp = 463 if v2 else 100
        ck.check("a8.manifest_rows", len(set(ids)) == len(ids) == n_exp and not bad,
                 f"{len(ids)} filas (ACTIVE={MM.ACTIVE}, se esperan {n_exp}), problemas={bad[:3]}")
        tr = [r for r in prod_rows if r["timing_ref"] == "1"]
        ck.check("a8.timing_ref", len(tr) == 12 and len({r["dataset"] for r in tr}) == 1, f"{len(tr)} filas timing_ref")
        x2 = [r for r in prod_rows if r["block"] == "X2"][0]
        cmd = [U.PY, str(U.RUNNER_DIR / "run_job.py"), "--job-id", x2["job_id"], "--manifest", str(U.RUNNER_DIR / "manifest.csv"),
               "--results-root", str(root / "results_prod_probe"), "--gpu", "cpu", "--dirty-wait-s", "0"]
        p = subprocess.run(cmd, env=U.base_env(), capture_output=True, text=True, timeout=120)
        ck.check("a8.prod_from_main_tree", p.returncode in (C.ENV, C.DIRTY) and
                 not (root / "results_prod_probe" / ".locks").exists() and
                 (p.returncode == C.DIRTY or "M4" in p.stdout), f"rc={p.returncode}: {p.stdout.strip().splitlines()[-1][:200]}")

        cr = root / "clean_repo"
        cr.mkdir()
        gc = lambda *a: subprocess.run(["git", "-C", str(cr), *a], capture_output=True, text=True)
        gc("init", "-q")
        (cr / "f.txt").write_text("v1\n")
        gc("add", "f.txt")
        gc("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
        cmd = [U.PY, str(U.RUNNER_DIR / "run_job.py"), "--job-id", x2["job_id"], "--manifest", str(U.RUNNER_DIR / "manifest.csv"),
               "--results-root", str(root / "results_prod_probe2"), "--gpu", "cpu", "--dirty-wait-s", "0", "--code-root", str(cr)]
        p = subprocess.run(cmd, env=U.base_env(), capture_output=True, text=True, timeout=120)
        ck.check("a8b.m4_code_root", p.returncode == C.ENV and "M4" in p.stdout and
                 not (root / "results_prod_probe2" / ".locks").exists(), f"rc={p.returncode}: {p.stdout.strip()[-160:]}")
        ck.check("a8b.m4_helpers", C.is_registered_worktree(cr) is False and C.runner_tag_at_head(cr) is None and
                 C.is_registered_worktree(U.REPO) is True, "worktree no registrado y sin tag runner-vN detectados")

        t0 = time.time()
        p = subprocess.run([U.SYS_PY, str(U.RUNNER_DIR / "status.py"), "--manifest", str(man), "--results-root", str(R)],
                           capture_output=True, text=True, timeout=60)
        dt_s = time.time() - t0
        pj = subprocess.run([U.SYS_PY, str(U.RUNNER_DIR / "status.py"), "--manifest", str(man), "--results-root", str(R),
                             "--json"], capture_output=True, text=True, timeout=60)
        import json as _j
        js = _j.loads(pj.stdout) if pj.returncode == 0 else {}
        tb = js.get("per_block", {}).get("TEST", {})
        ck.check("a9.status", p.returncode == 0 and dt_s < 5 and tb.get("bloqueado", 0) == 2 and tb.get("DONE", 0) == 11
                 and "lock_obsoleto?" not in tb, f"rc={p.returncode} {dt_s:.1f} s TEST={tb} (las filas _fake_* sin "
                                                 "ejecutar cuentan como invalido fuera de --test-mode)")
        ck.check("a9.no_metrics", not re.search(r"\b(auc|f1|roc|windowdiff)\b", p.stdout + pj.stdout, re.I),
                 "salida sin auc/f1/roc/windowdiff")

        rc, _ = U.run_job(root, man, r_sf["job_id"])
        Ps = C.JobPaths(R, r_sf["job_id"])
        att = sorted(Ps.state.glob("attempt_*"))
        ck.check("a10.preserved", rc == C.FAILED and len(att) == 1 and (att[0] / "scores.npy").exists() and
                 C.read_fail_count(R, r_sf["job_id"]) == 1, f"rc={rc} attempts={[a.name for a in att]}")
        rc, _ = U.run_job(root, man, r_sf["job_id"], env=U.base_env(JISA_TEST_SCOREFAIL_OK=1))
        cfg = U.load_json(Ps.final / "config.json") if Ps.final.exists() else {}
        ck.check("a10.listed", rc == 0 and any(x["has_scores"] for x in cfg.get("previous_attempts", {}).get("in_state", [])),
                 f"rc={rc} previous_attempts={cfg.get('previous_attempts')}")

        rc, _ = U.run_job(root, man, r_ckf["job_id"])
        Pc = C.JobPaths(R, r_ckf["job_id"])
        ck.check("a11.anomaly_done", rc == 0 and Pc.done.exists() and (Pc.final / "anomalies.json").exists() and
                 not (Pc.state / "fail_count").exists(), f"rc={rc}")

        rc, out = U.run_job(root, man, r_halt["job_id"])
        Ph = C.JobPaths(R, r_halt["job_id"])
        ck.check("a12.halt", rc == C.HALT and C.read_fail_count(R, r_halt["job_id"]) == 0 and
                 len(list(Ph.state.glob("halt_*.txt"))) == 1, f"rc={rc}")
        mh = sub_manifest("m_a12.csv", [r_halt])
        rcw, outw = U.run_worker(root, mh)
        ck.check("a12.worker_stops", rcw == C.HALT and C.read_fail_count(R, r_halt["job_id"]) == 0, f"rc worker={rcw}")

        rc, _ = U.run_job(root, man, r_oom["job_id"])
        Po = C.JobPaths(R, r_oom["job_id"])
        oj = sorted(Po.state.glob("oom_*.json"))
        ck.check("a13.oom_alone", rc == C.OOM and C.read_fail_count(R, r_oom["job_id"]) == 1 and oj and
                 U.load_json(oj[0])["counted"] is True, f"rc={rc}")
        ps = U.popen_job(root, man, r_sleep["job_id"], env=U.base_env(JISA_TEST_SLEEP_S=10), log_path=root / "sleep.log")
        U.wait_for(lambda: (C.read_lock_info(R / ".locks" / r_sleep["job_id"]) or {}).get("pid"), 20)
        mo = sub_manifest("m_a13.csv", [r_sleep, r_oom2])
        rcw, outw = U.run_worker(root, mo)
        Po2 = C.JobPaths(R, r_oom2["job_id"])
        oj2 = sorted(Po2.state.glob("oom_*.json"))
        ck.check("a13.oom_concurrent", rcw == C.OOM and C.read_fail_count(R, r_oom2["job_id"]) == 0 and oj2 and
                 U.load_json(oj2[0])["counted"] is False, f"rc worker={rcw}")
        ps.wait(timeout=60)

        dead = subprocess.Popen([U.PY, "-c", "pass"])
        dead.wait()
        jid = "A14"
        L = R / ".locks" / jid
        L.mkdir(parents=True)
        C.write_json_atomic(L / "info.json", {"pid": os.getpid(), "host": C.hostname(), "start_ticks": C.proc_start_ticks(os.getpid())})
        d = C.recover_lock(R, jid, dead.pid, None, reason="test")
        ck.check("a14.race_returned", d is None and L.exists() and (C.read_lock_info(L) or {}).get("pid") == os.getpid(),
                 "lock distinto del visto muerto: devuelto a .locks")
        ck.check("a14.release_not_owner", C.release_lock(L, dead.pid, None) is False and L.exists(), "no se libera un lock ajeno")
        d = C.recover_lock(R, jid, os.getpid(), C.proc_start_ticks(os.getpid()), reason="test")
        ck.check("a14.recovered", d is not None and not L.exists() and (d / "recovered.json").exists(), f"{d}")
        rc, _ = U.run_job(root, man, r_dup["job_id"])
        Pd = C.JobPaths(R, r_dup["job_id"])
        dups = sorted(Pd.state.glob("duplicate_*"))
        ck.check("a14.duplicate", rc == C.DUPLICATE and len(dups) == 1 and (dups[0] / "scores.npy").exists() and
                 C.read_fail_count(R, r_dup["job_id"]) == 0, f"rc={rc} dups={[x.name for x in dups]}")

        r_p1 = fake("_fake_ok", 50, rep=11)
        ps = U.popen_job(root, sub_manifest("m_a15a.csv", [r_sleep_tr, r_p1]), r_sleep_tr["job_id"],
                         env=U.base_env(JISA_TEST_SLEEP_S=6), log_path=root / "sleep_tr.log")
        U.wait_for(lambda: (C.read_lock_info(R / ".locks" / r_sleep_tr["job_id"]) or {}).get("pid"), 20)
        rcw, outw = U.run_worker(root, root / "m_a15a.csv")
        ck.check("a15.tr_running_blocks_others", rcw == 0 and not (R / r_p1["job_id"] / "DONE").exists() and
                 "timing_ref activo" in outw, f"rc={rcw}")
        ps.wait(timeout=60)
        r_s3 = fake("_fake_sleep", 60, rep=3)
        r_tr = fake("_fake_ok", 61, rep=12, timing_ref=1)
        m15 = sub_manifest("m_a15b.csv", [r_s3, r_tr])
        ps = U.popen_job(root, m15, r_s3["job_id"], env=U.base_env(JISA_TEST_SLEEP_S=6), log_path=root / "sleep3.log")
        U.wait_for(lambda: (C.read_lock_info(R / ".locks" / r_s3["job_id"]) or {}).get("pid"), 20)
        rcw, outw = U.run_worker(root, m15)
        ck.check("a15.tr_waits_exclusive", rcw == 0 and not (R / r_tr["job_id"] / "DONE").exists() and
                 "exclusividad" in outw, f"rc={rcw}")
        ps.wait(timeout=60)
        rcw, outw = U.run_worker(root, m15)
        ck.check("a15.tr_runs_alone", rcw == 0 and (R / r_tr["job_id"] / "DONE").exists(), f"rc={rcw}")

        jo = r_orphan["job_id"]
        Lo = R / ".locks" / jo
        Lo.mkdir(parents=True)
        C.write_json_atomic(Lo / "info.json", {"pid": dead.pid, "host": C.hostname(), "start_ticks": None})
        (R / ".state" / jo).mkdir(parents=True, exist_ok=True)
        os.rename(Lo, R / ".state" / jo / f"lock_obsoleto_{C.ts_compact()}")
        rcw, outw = U.run_worker(root, sub_manifest("m_a16.csv", [r_orphan]))
        n_again = C.account_orphan_moves(R, jo)
        ck.check("a16.orphan_accounted", rcw == 0 and C.read_fail_count(R, jo) == 1 and n_again == 0 and
                 (R / jo / "DONE").exists(), f"rc={rcw} fail_count={C.read_fail_count(R, jo)} recontado={n_again}")

        r_f = fake("_fake_ok", 70, rep=13)
        mfz = sub_manifest("m_a17.csv", [r_f])
        rcw, outw = U.run_worker(root, mfz, env=U.base_env(JISA_NOW="2026-10-01 00:30"))
        ck.check("a17.frozen", rcw == 0 and "congelado" in outw and not (R / r_f["job_id"]).exists(), f"rc={rcw}")
        rcw, outw = U.run_worker(root, mfz, env=U.base_env(JISA_NOW="2026-09-30 12:00"))
        ck.check("a17.freeze_day_unknown_tbar", rcw == 0 and not (R / r_f["job_id"]).exists() and "congelacion" in outw,
                 f"rc={rcw} (t_medio desconocido -> no se reclama)")

        ok_fn, detail = C.ckpt_paths_ignored(C.RESULTS_ROOT, "JOBX")
        gi = subprocess.run(["git", "-C", str(U.REPO), "check-ignore", "--no-index", "results/revision/JOBX/ckpt/model.pkl"],
                            capture_output=True, text=True)
        ck.check("a18.gitignore_fn", (gi.returncode == 0) or ok_fn is False, f"funcion={ok_fn}; git rc={gi.returncode}")
        if not ok_fn:
            ck.warn("a18.PRECONDICION", "results/revision/**/ckpt y *.pkl NO estan en .gitignore: run_job de "
                                        "produccion con save_ckpt=1 devolvera ENV hasta anadirlo")

        R19 = root / "results19"
        rs19 = U.make_row(block="S-w1", variant="lstm", pipeline="corrected", seed="A", dataset="KNAD:Active_Wiretap", priority=1)
        rx19 = U.make_row(block="X2", variant="lstm", pipeline="orig", dataset="KNAD:OS_Scan", priority=2)
        rt19 = U.make_row(block="T1-A-KNAD", variant="lstm", pipeline="corrected", seed="A", dataset="KNAD:Active_Wiretap", priority=3)
        ro19 = U.make_row(block="T1-A-KNAD", variant="lstm", pipeline="corrected", seed="A", dataset="KNAD:OS_Scan", priority=4)

        def done19(r, w):
            (R19 / r["job_id"]).mkdir(parents=True, exist_ok=True)
            (R19 / r["job_id"] / "DONE").write_text("x\n")
            C.write_json_atomic(R19 / r["job_id"] / "timing.json", {"wall_seconds": w})

        rows19 = [rs19, rx19, rt19, ro19]
        at19 = __import__("datetime").datetime(2026, 9, 30, 23, 30)
        done19(rs19, 360.0)
        i1 = C.tbar_info(C.collect_timings(R19, rows19), rt19)
        f1 = C.fits_before_freeze(rt19, R19, rows19, at=at19)
        done19(rx19, 7200.0)
        i2 = C.tbar_info(C.collect_timings(R19, rows19), rt19)
        f2 = C.fits_before_freeze(rt19, R19, rows19, at=at19)
        done19(ro19, 1000.0)
        i3 = C.tbar_info(C.collect_timings(R19, rows19), rt19)
        f3 = C.fits_before_freeze(rt19, R19, rows19, at=at19)
        ck.check("a19.eta_families", i1 == (None, None, False) and f1[0] is False and i2[0] == 7200.0 and i2[2] is False
                 and f2[0] is False and "otra familia" in f2[1] and i3 == (1000.0, "familia+variante", True) and f3[0] is True,
                 f"solo S: {i1} {f1[0]}; +X2: {i2} {f2[0]}; +T1 otro dataset: {i3} {f3[0]}")

        r20 = fake("_fake_fail", 90, rep=20)
        m20 = sub_manifest("m_a20.csv", [r20])
        st20 = R / ".state" / r20["job_id"]
        st20.mkdir(parents=True, exist_ok=True)
        (st20 / "fail_count").mkdir()
        t20 = time.time()
        rcw, outw = U.run_worker(root, m20, timeout=150)
        launches = outw.count(f"lanzo {r20['job_id']}")
        ck.check("a20.unpersisted_failure_bounded", rcw == 0 and launches == 3 and "excluido en esta sesion" in outw
                 and time.time() - t20 < 140, f"rc={rcw} lanzamientos={launches} en {time.time() - t20:.0f} s")

        r21 = fake("_fake_count", 91, rep=21)
        m21 = sub_manifest("m_a21.csv", [r21])
        counter21 = root / "counter21.txt"
        env21 = U.base_env(JISA_TEST_COUNTER=counter21, JISA_TEST_FAKE_SLEEP=0.2)
        (gr / "f.txt").write_text("v3\n")
        pB = U.popen_job(root, m21, r21["job_id"], ["--force-git-check", "--dirty-wait-s", "60", "--dirty-sleep-s", "1",
                                                   "--code-root", gr], env=env21, log_path=root / "a21_B.log")
        U.wait_for(lambda: bool(list((R / ".state" / r21["job_id"]).glob("dirty_*.txt"))), 30)
        rcA, _ = U.run_job(root, m21, r21["job_id"], env=env21)
        g("checkout", "--", "f.txt")
        rcB = pB.wait(timeout=60)
        n21 = len(counter21.read_text().splitlines()) if counter21.exists() else 0
        dups21 = list((R / ".state" / r21["job_id"]).glob("duplicate_*"))
        logB = (root / "a21_B.log").read_text()
        ck.check("a21.done_recheck_after_lock", rcA == 0 and rcB == 0 and n21 == 1 and not dups21 and "DONE aparecio" in logB,
                 f"rcA={rcA} rcB={rcB} ejecuciones={n21} duplicados={len(dups21)}")

        P22 = C.JobPaths(R, "A22JOB")
        t22 = P22.tmp_root / f"A22JOB.{C.hostname()}.{os.getpid()}"
        t22.mkdir(parents=True)
        (t22 / "scores.npy").write_bytes(b"intento previo")
        t_new, act22 = P22.new_tmp(C.hostname(), os.getpid())
        moved22 = sorted(P22.state.glob("attempt_*_pidreuse"))
        ck.check("a22.pid_reuse_preserved", bool(act22) and act22["action"] == "moved_pid_reuse" and len(moved22) == 1 and
                 (moved22[0] / "scores.npy").exists() and t_new.exists() and not (t_new / "scores.npy").exists(),
                 f"accion={act22} movidos={[m.name for m in moved22]}")

        L23 = R / ".locks" / "A23"
        C.acquire_lock(L23, {"job_id": "A23"})
        (L23 / f".heartbeat.tmp.{os.getpid()}.999").write_text("latido a medio escribir\n")
        rel23 = C.release_lock(L23)
        hb23 = C.Heartbeat(R / ".locks" / "A23b", "A23b", every_s=0.01)
        time.sleep(0.05)
        hb23.stop()
        L24 = R / ".locks" / "A24"
        orig_wja = C.write_json_atomic

        def boom(*_a, **_k):
            raise OSError(28, "No space left on device (simulado)")
        C.write_json_atomic = boom
        try:
            raised24 = False
            try:
                C.acquire_lock(L24, {"job_id": "A24"})
            except OSError:
                raised24 = True
        finally:
            C.write_json_atomic = orig_wja
        ck.check("a23.lock_cleanup", rel23 is True and not L23.exists() and not hb23._t.is_alive() and raised24 and
                 not L24.exists(), f"release={rel23} lock23={L23.exists()} thread_vivo={hb23._t.is_alive()} "
                                   f"acquire_lanza={raised24} lock24={L24.exists()}")

        r24s = fake("_fake_sleep", 92, rep=24)
        r24t = fake("_fake_ok", 93, rep=24, timing_ref=1)
        r24s2 = fake("_fake_sleep", 94, rep=25, timing_ref=1)
        r24o = fake("_fake_ok", 95, rep=25)
        m24 = sub_manifest("m_a24.csv", [r24s, r24t, r24s2, r24o])
        ps = U.popen_job(root, m24, r24s["job_id"], env=U.base_env(JISA_TEST_SLEEP_S=5), log_path=root / "a24_s.log")
        U.wait_for(lambda: (C.read_lock_info(R / ".locks" / r24s["job_id"]) or {}).get("pid"), 20)
        rc_t, out_t = U.run_job(root, m24, r24t["job_id"])
        ps.wait(timeout=60)
        ps2 = U.popen_job(root, m24, r24s2["job_id"], env=U.base_env(JISA_TEST_SLEEP_S=5), log_path=root / "a24_s2.log")
        U.wait_for(lambda: (C.read_lock_info(R / ".locks" / r24s2["job_id"]) or {}).get("pid"), 20)
        rc_o, out_o = U.run_job(root, m24, r24o["job_id"])
        ps2.wait(timeout=60)
        rc_t2, _ = U.run_job(root, m24, r24t["job_id"])
        ok24 = (rc_t == C.LOCKED and "exclusiva" in out_t and rc_o == C.LOCKED and "exclusividad" in out_o and
                not (R / r24o["job_id"]).exists() and rc_t2 == C.OK and (R / r24t["job_id"] / "DONE").exists() and
                all(C.read_fail_count(R, r["job_id"]) == 0 for r in (r24t, r24o)) and
                not (R / ".locks" / r24t["job_id"]).exists() and not (R / ".locks" / r24o["job_id"]).exists() and
                (R / r24s["job_id"] / "DONE").exists() and (R / r24s2["job_id"] / "DONE").exists())
        ck.check("a24.late_lock_exclusive", ok24, f"timing_ref tarde rc={rc_t}; normal tarde rc={rc_o}; timing_ref solo rc={rc_t2}")

        r25 = fake("_fake_ok", 96, rep=26)
        m25 = sub_manifest("m_a25.csv", [r25])
        rc25, out25 = U.run_job(root, m25, r25["job_id"], env=U.base_env(JISA_TEST_RAISE_IN_FINALLY=1))
        ck.check("a25.exception_after_commit", rc25 == C.OK and (R / r25["job_id"] / "DONE").exists() and
                 C.read_fail_count(R, r25["job_id"]) == 0 and "ya tiene DONE" in out25,
                 f"rc={rc25} DONE={(R / r25['job_id'] / 'DONE').exists()} fail_count={C.read_fail_count(R, r25['job_id'])}")
    except Exception:
        import traceback
        traceback.print_exc()
        ck.check("a.exception", False, "excepcion en el test")
    rc = ck.finish()
    U.cleanup(root, rc == 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
