#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""status.py - state of the job queue: counts per block, locks and their heartbeat age, events, time estimates and compute used."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

STATES = ["DONE", "en_curso", "lock_obsoleto?", "bloqueado", "fallido_reintentable", "no_ejecutable", "invalido",
          "pendiente"]
EVENT_PREFIXES = ("halt", "oom", "dirty", "interrupted", "attempt", "duplicate", "env", "nonexec")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Estado de la cola del runner v1 (solo recuentos, locks y tiempos).")
    ap.add_argument("--manifest", default=str(C.MANIFEST))
    ap.add_argument("--results-root", default=str(C.RESULTS_ROOT))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--stale-heartbeat-min", type=float, default=15.0)
    a = ap.parse_args(argv)
    R = Path(a.results_root)
    host = C.hostname()
    if not Path(a.manifest).exists():
        print(f"manifiesto no existe: {a.manifest}")
        return 0
    rows = C.read_manifest(a.manifest)
    per_block = defaultdict(Counter)
    states = {}
    for r in rows:
        jid = r["job_id"]
        L = R / ".locks" / jid
        fc = C.read_fail_count(R, jid)
        if (R / jid / "DONE").exists():
            s = "DONE"
        elif L.exists():
            st = C.lock_status(L, host)
            age = st["hb_age_s"] or 0
            s = "en_curso" if (st["alive"] or (st["alive"] is None and age <= a.stale_heartbeat_min * 60)) else "lock_obsoleto?"
        elif any((R / ".state" / jid).glob("nonexec_*.json")):
            s = "no_ejecutable"
        elif fc >= C.MAX_FAILS:
            s = "bloqueado"
        elif fc > 0:
            s = "fallido_reintentable"
        else:
            try:
                C.validate_row(r)
                s = "pendiente"
            except C.InvalidJob:
                s = "invalido"
        states[jid] = s
        per_block[r["block"]][s] += 1
    timings = C.collect_timings(R, rows)
    now = C.now()
    pend_s, pend_unrel_s, unknown, unreliable = 0.0, 0.0, 0, 0
    expl_s = 0.0
    for r in rows:
        s = states[r["job_id"]]
        if s in ("pendiente", "fallido_reintentable", "en_curso", "lock_obsoleto?"):
            if C.block_family(r["block"]) in C.TIMING_EXCLUDED_FAMILIES:
                unknown += 1
                continue
            tb, _lvl, reliable = C.tbar_info(timings, r)
            if tb is None:
                unknown += 1
                continue
            exploratory = C.block_family(r["block"]) in C.EXPLORATORY_FAMILIES
            if s == "en_curso":
                info = C.read_lock_info(R / ".locks" / r["job_id"]) or {}
                try:
                    import datetime as dt
                    el = (dt.datetime.now() - dt.datetime.fromisoformat(info.get("start"))).total_seconds()
                except Exception:
                    el = 0
                add = max(tb - el, 0)
            else:
                add = tb
            if exploratory:
                expl_s += add
            elif reliable:
                pend_s += add
            else:
                unreliable += 1
                pend_unrel_s += add
    consumed = sum(w for _, w in timings)
    hours_to_freeze = (C.FREEZE - now).total_seconds() / 3600
    locks = []
    ld = R / ".locks"
    if ld.exists():
        for L in sorted(ld.iterdir()):
            if not L.is_dir():
                continue
            st = C.lock_status(L, host)
            info = st["info"] or {}
            locks.append({"job_id": L.name, "host": st["host"], "pid": st["pid"],
                          "alive": st["alive"], "gpu": info.get("gpu"), "start": info.get("start"),
                          "timing_ref": info.get("timing_ref"),
                          "hb_age_s": None if st["hb_age_s"] is None else round(st["hb_age_s"]),
                          "heartbeat": st["heartbeat"][:160]})
    events = []
    sd = R / ".state"
    if sd.exists():
        for d in sorted(sd.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            ev = Counter(p.name.split("_")[0] for p in d.iterdir()
                         if p.name.split("_")[0] in EVENT_PREFIXES)
            if ev:
                events.append({"job_id": d.name, **ev})
    gi_ok, gi_detail = (None, "sin filas con save_ckpt")
    ck = [r for r in rows if r.get("save_ckpt") == "1"]
    if ck:
        gi_ok, gi_detail = C.ckpt_paths_ignored(R, ck[0]["job_id"])
    out = {"host": host, "now": now.isoformat(timespec="minutes"), "manifest": a.manifest, "results_root": str(R),
           "per_block": {b: dict(c) for b, c in sorted(per_block.items())}, "locks": locks, "events_no_fail": events,
           "eta": {"pending_hours_known": round(pend_s / 3600, 2), "jobs_without_tbar": unknown,
                   "pending_hours_confirmatory": round(pend_s / 3600, 2),
                   "pending_hours_exploratory": round(expl_s / 3600, 2),
                   "eta_rule": "la escalera de §13.3 se decide con pending_hours_confirmatory; "
                               "pending_hours_exploratory (familia T7) se muestra al lado y NO la dispara",
                   "pending_hours_cross_family_unreliable": round(pend_unrel_s / 3600, 2),
                   "jobs_cross_family_unreliable": unreliable,
                   "hours_to_freeze": round(hours_to_freeze, 1), "consumed_hours_done": round(consumed / 3600, 2),
                   "done_with_timing": len(timings),
                   "jobs_nonexecutable_excluded": sum(1 for v in states.values() if v == "no_ejecutable"),
                   "method": "mediana de wall_seconds de DONE sin la familia S: (familia, variante, dataset, factores "
                             "window/k_clusters/grouping/contamination) -> (familia, variante, factores) [fiable]; "
                             "variante de otra familia -> aparte, NO fiable; celdas no_ejecutable fuera"},
           "precondition_ckpt_gitignore": {"ok": gi_ok, "detail": gi_detail}}
    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    print(f"status runner v1 - {out['now']} en {host} - {len(rows)} filas en el manifiesto")
    w = max([len(b) for b in per_block] + [6])
    print(f"  {'bloque':<{w}} " + " ".join(f"{s:>14}" for s in STATES))
    for b, c in sorted(per_block.items()):
        print(f"  {b:<{w}} " + " ".join(f"{c.get(s, 0):>14}" for s in STATES))
    print("locks:" if locks else "locks: ninguno")
    for l in locks:
        print(f"  {l['job_id']} host={l['host']} pid={l['pid']} vivo={l['alive']} gpu={l['gpu']} inicio={l['start']} "
              f"latido hace {l['hb_age_s']} s timing_ref={l['timing_ref']} | {l['heartbeat']}")
    if events:
        print(f"eventos en .state ({'/'.join(EVENT_PREFIXES)}; un oom contado tambien suma fallo):")
        for e in events:
            print("  " + " ".join(f"{k}={v}" for k, v in e.items()))
    e = out["eta"]
    print(f"ETA: {e['pending_hours_known']} h de GPU por ejecutar con t_medio de la misma familia; "
          f"{e['jobs_cross_family_unreliable']} jobs solo con t_medio de otra familia ("
          f"{e['pending_hours_cross_family_unreliable']} h, NO fiable); {e['jobs_without_tbar']} jobs sin t_medio "
          f"(incluye sondas S; sin datos: no se inventan cifras); {e['hours_to_freeze']} h hasta la congelacion; "
          f"consumido {e['consumed_hours_done']} h en {e['done_with_timing']} jobs DONE (sin S)")
    short = gi_detail if len(gi_detail) < 220 else gi_detail.split(": [")[0] + f" (p. ej. .../{ck[0]['job_id']}/ckpt/model.pkl)"
    print(f"precondicion .gitignore de checkpoints: {gi_ok} - {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
