#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(e) Variantes Keras con formas reducidas (L=16) y checkpoint en las 5 familias (S9), mas la sonda S (smoke.py) con
presupuestos minimos y la barrera de dos sondas concurrentes (S10). CPU, ~3 min. No entra en run_all --quick.
Datos reales de OS_Scan: orig max_rows=4000 (fm=1000, ad=1500, ew=1000) y corrected max_rows=4000 con train_rows=2500.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _util as U
from _util import C

ORIG_ARGS = ["--max-rows", 4000, "--test-override", "fm_grace=1000,ad_grace=1500,execution_window=1000,sequence_length=16",
             "--data-root", str(C.DATA_ROOT)]
PROBE_ARGS = ["--max-rows", 4000, "--test-override",
              "fm_grace=1000,ad_grace=1500,sequence_length=16,train_rows=2500,probe_train_s=1.5,probe_predict_s=1.0,"
              "probe_warmup_steps=5,probe_warmup_batches=1,probe_loop_rows=200,probe_barrier_timeout_s=60",
              "--data-root", str(C.DATA_ROOT)]


def main():
    ck = U.Checker("test_e_windowed_smoke")
    root = U.make_root("e")
    R = root / "results"
    try:
        fam = [("conv1d", True), ("conv2d", False), ("lstm", True), ("mha", False), ("mlp", True)]
        rows = [U.make_row(variant=v, ar=ar, seed="A", save_ckpt=1, priority=i) for i, (v, ar) in enumerate(fam)]
        probes = [U.make_row(block="S-w1", variant="lstm", ar=True, pipeline="corrected", seed="A", priority=20),
                  U.make_row(block="S-w2", variant="mha", ar=True, pipeline="corrected", seed="A", priority=21, rep=1),
                  U.make_row(block="S-w2", variant="mha", ar=True, pipeline="corrected", seed="A", priority=22, rep=2)]
        man = root / "manifest.csv"
        U.write_manifest(man, rows + probes)
        t0 = time.time()
        env = U.base_env(PYTHONHASHSEED=1234)
        procs = {r["variant"]: U.popen_job(root, man, r["job_id"], ORIG_ARGS, env=env, log_path=root / "logs" / f"{r['variant']}.log")
                 for r in rows}
        procs.update({f"probe{i}": U.popen_job(root, man, r["job_id"], PROBE_ARGS, env=env, log_path=root / "logs" / f"probe{i}.log")
                      for i, r in enumerate(probes)})
        rcs = {k: p.wait(timeout=1200) for k, p in procs.items()}
        ck.info(f"rc={rcs} en {time.time() - t0:.0f} s")
        for (v, ar), r in zip(fam, rows):
            J = R / r["job_id"]
            if rcs[v] != 0:
                ck.check(f"e.{v}", False, f"rc={rcs[v]}\n{C.tail_text(root / 'logs' / f'{v}.log', 2500)}")
                continue
            arr = U.load_json(J / "arrays.json")
            cfg = U.load_json(J / "config.json")
            s, e = arr["scores.npy"], arr.get("ae_errors.npy", {})
            ok_arr = (s["shape"] == [4000] and s["dtype"] == "float64" and e.get("shape") == [4000, 4] and
                      e.get("dtype") == "float64" and e.get("selfcheck_output_layer_from_saved_file", {}).get("all_equal") is True)
            ck.check(f"e.{v}.arrays", ok_arr, f"scores {s['shape']} {s['dtype']} n_nan={s.get('n_nan')}; ae_errors {e.get('shape')} "
                                              f"{e.get('dtype')} autochequeo={e.get('selfcheck_output_layer_from_saved_file', {}).get('all_equal')}")
            import csv as _csv
            tl = list(_csv.DictReader((J / "train_log.csv").open()))
            ck.check(f"e.{v}.train_log", len(tl) == 4 and all(r["loss_epoch_1"] not in ("NA", "") for r in tl) and
                     all(r["train_steps"] == "1500" and r["optimizer_iterations"] in ("1500", "1501") and r["epochs"] == "1"
                         for r in tl),
                     f"loss={[r['loss_epoch_1'][:8] for r in tl]} pasos={[r['train_steps'] for r in tl]} "
                     f"iter_opt={[r['optimizer_iterations'] for r in tl]} "
                     f"fuente={tl[0]['loss_source'][:20] if tl else None}")
            hp = cfg["hyperparameters_effective"]["autoencoders"][0]
            if v != "mlp":
                exp = "create_windowed_data_ar" if ar else "create_windowed_data"
                ck.check(f"e.{v}.window", hp.get("create_windowed") == exp and hp.get("sequence_length") == 16 and
                         cfg["test_overrides"].get("sequence_length") == 16, f"{hp.get('create_windowed')} L={hp.get('sequence_length')}")
            ckp = J / "ckpt"
            need = [ckp / f"ae{i}" / s_ / "saved_model.pb" for i in range(4) for s_ in ("model", "encoder", "decoder")]
            need += [ckp / "model.pkl", ckp / "output.pkl"]
            ck.check(f"e.{v}.ckpt", all(p.exists() for p in need) and cfg["ckpt"] and "error" not in cfg["ckpt"],
                     f"{sum(p.exists() for p in need)}/{len(need)} en {cfg['ckpt'].get('seconds', 0):.1f} s")
        for i, r in enumerate(probes):
            J = R / r["job_id"]
            k = f"probe{i}"
            if rcs[k] != 0:
                ck.check(f"e.{k}", False, f"rc={rcs[k]}\n{C.tail_text(root / 'logs' / f'{k}.log', 2500)}")
                continue
            pj = U.load_json(J / "probe.json")
            est = pj["estimate"]["estimated_full_job_seconds"]
            ok = (not (J / "scores.npy").exists() and len(pj["per_ae"]) == 4 and est and est > 0 and
                  all((a.get("train") or {}).get("steps_per_s") for a in pj["per_ae"]))
            ck.check(f"e.{k}.probe", ok, f"estimacion {est and round(est, 1)} s; barrera {pj.get('barrier')}; sin scores")
        b = [U.load_json(R / r["job_id"] / "probe.json").get("barrier", {}) for r in probes[1:]
             if (R / r["job_id"] / "probe.json").exists()]
        ck.check("e.barrier_w2", len(b) == 2 and all(x.get("target") == 2 and x.get("reached") == 2 for x in b),
                 f"{[(x.get('target'), x.get('reached'), round(x.get('waited_s', 0), 1)) for x in b]}")
    except Exception:
        import traceback
        traceback.print_exc()
        ck.check("e.exception", False, "excepcion en el test")
    rc = ck.finish()
    U.cleanup(root, rc == 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
