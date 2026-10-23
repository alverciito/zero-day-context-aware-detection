#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(c) EQUIVALENCIA runner == codigo original con la misma entrada (pipeline orig KNAD OS_Scan real, CPU). DESIGN §12 (c).
c1 stat, seed None: sha256(runner scores.npy) == sha256(.npy del show_stats real); feature_map; autochequeo de
   ae_errors.npy (float64, M1) sobre el fichero guardado; seed_calls == []; PYTHONHASHSEED sin definir
c2 original, seed None, save_ckpt=1: igual que c1 + ckpt/kitnet_model.pkl
c3 mlp, seed A, save_ckpt=1: bytes iguales (PASS) o |d| <= max(1e-6, 2*|d control|) con dos rutas directas (PASS con
   tolerancia, documentado); train_log.csv con loss de 4 AEs; PYTHONHASHSEED=1234; ckpt/ae0..3, model.pkl, output.pkl
c5 stat con save_ckpt=1 -> INVALID (2)
Grace periods reducidos SOLO con --test-mode (registrado en config.json): stat/original fm=5000 ad=10000 ew=4000
sobre 20.000 filas (ad >= 10.000 por el ELM); mlp fm=2000 ad=2000 ew=1500 sobre 6.000 filas.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _util as U
from _util import C

CFG = {"stat": dict(max_rows=20000, fm=5000, ad=10000, ew=4000),
       "original": dict(max_rows=20000, fm=5000, ad=10000, ew=4000),
       "mlp": dict(max_rows=6000, fm=2000, ad=2000, ew=1500)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runner_args(v):
    c = CFG[v]
    return ["--max-rows", c["max_rows"], "--test-override",
            f"fm_grace={c['fm']},ad_grace={c['ad']},execution_window={c['ew']}", "--data-root", str(C.DATA_ROOT)]


def direct_cmd(v, seed, out):
    c = CFG[v]
    return [U.PY, str(U.TESTS_DIR / "_direct_original.py"), "--code-root", str(U.CODE_ROOT), "--data-root", str(C.DATA_ROOT),
            "--variant", v, "--ar", "True", "--seed", seed, "--max-rows", str(c["max_rows"]), "--fm", str(c["fm"]),
            "--ad", str(c["ad"]), "--ew", str(c["ew"]), "--out-dir", str(out)]


def main():
    ck = U.Checker("test_c_equivalence")
    root = U.make_root("c")
    R = root / "results"
    try:
        r1 = U.make_row(variant="stat", priority=1)
        r2 = U.make_row(variant="original", priority=2, save_ckpt=1)
        r3 = U.make_row(variant="mlp", priority=3, seed="A", save_ckpt=1)
        r5 = U.make_row(variant="stat", priority=5, rep=2, save_ckpt=1)
        r6 = U.make_row(variant="stat", priority=6, pipeline="corrected")
        r7 = U.make_row(variant="stat", priority=7, pipeline="corrected", rep=2)
        man = root / "manifest.csv"
        U.write_manifest(man, [r1, r2, r3, r5, r6, r7])
        rc, out = U.run_job(root, man, r5["job_id"])
        ck.check("c5.stat_ckpt_invalid", rc == C.INVALID, f"rc={rc}")

        t0 = time.time()
        logs = root / "logs"
        procs = {}
        env_none = U.base_env()
        env_a = U.base_env(PYTHONHASHSEED=1234)
        procs["run_stat"] = U.popen_job(root, man, r1["job_id"], runner_args("stat"), env=env_none, log_path=logs / "run_stat.log")
        procs["run_original"] = U.popen_job(root, man, r2["job_id"], runner_args("original"), env=env_none, log_path=logs / "run_original.log")
        procs["dir_stat"] = subprocess.Popen(direct_cmd("stat", "None", root / "direct_stat"), env=env_none,
                                             stdout=open(logs / "dir_stat.log", "w"), stderr=subprocess.STDOUT)
        procs["dir_original"] = subprocess.Popen(direct_cmd("original", "None", root / "direct_original"), env=env_none,
                                                 stdout=open(logs / "dir_original.log", "w"), stderr=subprocess.STDOUT)
        procs["run_mlp"] = U.popen_job(root, man, r3["job_id"], runner_args("mlp"), env=U.base_env(), log_path=logs / "run_mlp.log")
        procs["dir_mlp1"] = subprocess.Popen(direct_cmd("mlp", "1234", root / "direct_mlp1"), env=env_a,
                                             stdout=open(logs / "dir_mlp1.log", "w"), stderr=subprocess.STDOUT)
        corr_args = ["--max-rows", 4000, "--test-override", "fm_grace=1000,ad_grace=1500,execution_window=1000,train_rows=2500",
                     "--data-root", str(C.DATA_ROOT)]
        bad_args = ["--max-rows", 4000, "--test-override", "fm_grace=1000,ad_grace=1500,execution_window=1000,train_rows=2000",
                    "--data-root", str(C.DATA_ROOT)]
        procs["run_corrected"] = U.popen_job(root, man, r6["job_id"], corr_args, env=env_none, log_path=logs / "run_corrected.log")
        procs["run_mismatch"] = U.popen_job(root, man, r7["job_id"], bad_args, env=env_none, log_path=logs / "run_mismatch.log")
        rcs = {k: p.wait(timeout=900) for k, p in procs.items()}
        ck.info(f"ejecuciones en paralelo: {rcs} en {time.time() - t0:.0f} s")
        for k, v in rcs.items():
            if v != 0 and k != "run_mismatch":
                ck.info(f"--- cola de {k}.log ---\n" + C.tail_text(logs / f"{k}.log", 2500))

        J6 = R / r6["job_id"]
        if rcs["run_corrected"] == 0 and (J6 / "DONE").exists():
            cfg6 = U.load_json(J6 / "config.json")
            reg = cfg6["regions"]
            cands = {k: v for k, v in reg["detector_protocol_candidates"].items() if not k.startswith("_")}
            pairs = {k: list(C.check_protocol_regions(reg, k)) for k in cands}
            ck.check("c6.regions_contract", "evaluation" not in reg and "evaluation" not in json.dumps(cfg6.get("cache", {})) and
                     reg["train"] == [0, 2500] and reg["post_train"] == [2500, 4000] and
                     pairs["calib_first_10pct_post_train"] == [[2500, 2650], [2650, 4000]] and
                     pairs["calib_in_sample_train_tail"] == [[1500, 2500], [2500, 4000]] and
                     cands["calib_in_sample_train_tail"]["calib_in_sample"] is True and
                     all(not v["regression_only"] for v in cands.values()) and not cfg6.get("anomalies"),
                     f"candidatas {pairs}; anomalias={cfg6.get('anomalies')}")
            bad = json.loads(json.dumps(reg))
            bad["detector_protocol_candidates"]["calib_first_10pct_post_train"]["eval"] = [2600, 4000]
            bad2 = json.loads(json.dumps(reg))
            bad2["detector_protocol_candidates"]["calib_in_sample_train_tail"]["eval"] = [2000, 4000]
            raised = []
            for rr, k in ((bad, "calib_first_10pct_post_train"), (bad2, "calib_in_sample_train_tail")):
                try:
                    C.check_protocol_regions(rr, k)
                    raised.append(False)
                except C.ProtocolRegionError:
                    raised.append(True)
            ck.check("c6.overlap_rejected", raised == [True, True], f"calib∩eval y eval∩train rechazados: {raised}")
            import numpy as np
            import pandas as pd
            lab = np.load(J6 / "labels.npy")
            y_csv = pd.read_csv(C.DATA_ROOT / "KITSUNE" / "OS Scan" / "OS_Scan_labels.csv", nrows=4000)["x"].to_numpy()
            arr6 = U.load_json(J6 / "arrays.json")
            ck.check("c6.labels", lab.dtype == np.int8 and lab.shape == (4000,) and np.array_equal(lab, y_csv) and
                     arr6["labels.npy"]["sha256"] == C.sha256_file(J6 / "labels.npy"),
                     f"labels.npy {lab.dtype} {lab.shape} == etiquetas del CSV alineadas")
        else:
            ck.check("c6.done", False, f"corrected sin DONE (rc={rcs['run_corrected']})\n{C.tail_text(logs / 'run_corrected.log', 2500)}")
        st7 = C.JobPaths(R, r7["job_id"]).state
        ck.check("c7.train_rows_mismatch_halt", rcs["run_mismatch"] == C.HALT and not (R / r7["job_id"]).exists() and
                 len(list(st7.glob("halt_*.txt"))) == 1 and C.read_fail_count(R, r7["job_id"]) == 0 and
                 "train_rows=2000" in C.tail_text(logs / "run_mismatch.log", 4000),
                 f"rc={rcs['run_mismatch']} (train_rows != fm + ad -> HALT antes de cargar datos)")

        for cid, row, v, ddir in (("c1", r1, "stat", "direct_stat"), ("c2", r2, "original", "direct_original")):
            J = R / row["job_id"]
            if not (J / "DONE").exists():
                ck.check(f"{cid}.done", False, f"sin DONE (rc={rcs['run_' + v]})")
                continue
            dnpy = root / ddir / f"{v}_ar_True.npy"
            ck.check(f"{cid}.bytes", dnpy.exists() and sha(J / "scores.npy") == sha(dnpy),
                     f"sha256 runner {sha(J / 'scores.npy')[:16]} vs show_stats {sha(dnpy)[:16] if dnpy.exists() else None}")
            fs = U.load_json(J / "fm_stats.json")
            dj = U.load_json(root / ddir / "direct.json")
            ck.check(f"{cid}.feature_map", fs["feature_map"] == dj["feature_map"] and fs["consistency"]["feature_map_equal"],
                     f"tamanos {[len(g) for g in fs['feature_map']]}; iteraciones DBSCAN {fs['iterations']}")
            arr = U.load_json(J / "arrays.json")
            ae = arr.get("ae_errors.npy", {})
            ck.check(f"{cid}.ae_errors", ae.get("dtype") == "float64" and ae.get("shape") == [CFG[v]["max_rows"], 4] and
                     ae.get("selfcheck_output_layer_from_saved_file", {}).get("all_equal") is True,
                     f"dtype={ae.get('dtype')} shape={ae.get('shape')} autochequeo={ae.get('selfcheck_output_layer_from_saved_file', {}).get('all_equal')}")
            cfg = U.load_json(J / "config.json")
            env = U.load_json(J / "env.json")
            ck.check(f"{cid}.seeds", cfg["seed_calls"] == [] and env["PYTHONHASHSEED"] is None,
                     f"seed_calls={cfg['seed_calls']} PYTHONHASHSEED={env['PYTHONHASHSEED']}")
            ck.check(f"{cid}.config", cfg["test_mode"] is True and cfg["test_overrides"].get("ad_grace") == CFG[v]["ad"] and
                     cfg["regions"]["execution"] == [CFG[v]["fm"] + CFG[v]["ad"], CFG[v]["max_rows"]] and
                     len(cfg["hyperparameters_effective"]["autoencoders"]) == 4 and not cfg.get("anomalies"),
                     f"anomalias={cfg.get('anomalies')}")
            if v == "original":
                ck.check("c2.ckpt", (J / "ckpt" / "kitnet_model.pkl").exists(), "ckpt/kitnet_model.pkl")
            else:
                cand = cfg["regions"]["detector_protocol_candidates"]["as_submitted_regression_only"]
                try:
                    C.check_protocol_regions(cfg["regions"], "as_submitted_regression_only")
                    rej = False
                except C.ProtocolRegionError as e:
                    rej = "solo regresion" in str(e)
                ok_allow = (cand["calib"] == [100_000, 200_000] and cand["eval"] == [0, CFG[v]["max_rows"]] and
                            cand["calib_in_sample"] is True)
                lab = arr.get("labels.npy", {})
                ck.check("c1.orig_regions_labels", cand["regression_only"] and rej and ok_allow and
                         lab.get("dtype") == "int8" and lab.get("shape") == [CFG[v]["max_rows"]] and "desfase" in lab.get("alignment", ""),
                         f"regression_only rechazada sin permiso={rej}; labels={lab.get('shape')} {lab.get('dtype')}")

        J = R / r3["job_id"]
        if (J / "DONE").exists() and (root / "direct_mlp1" / "mlp_ar_True.npy").exists():
            import numpy as np
            a_run = np.load(J / "scores.npy")
            a_d1 = np.load(root / "direct_mlp1" / "mlp_ar_True.npy")
            if sha(J / "scores.npy") == sha(root / "direct_mlp1" / "mlp_ar_True.npy"):
                ck.check("c3.bytes", True, "mlp seed A: runner == directo byte a byte")
            else:
                ck.info("mlp: runner != directo en bytes; se lanza la ruta de control (directo x2)")
                p = subprocess.run(direct_cmd("mlp", "1234", root / "direct_mlp2"), env=env_a, capture_output=True, text=True,
                                   timeout=900)
                a_d2 = np.load(root / "direct_mlp2" / "mlp_ar_True.npy")
                d_run = float(np.max(np.abs(a_run - a_d1)))
                d_ctl = float(np.max(np.abs(a_d2 - a_d1)))
                tol = max(1e-6, 2 * d_ctl)
                ck.check("c3.tolerance", d_run <= tol, f"max|runner-directo|={d_run:.3e} <= tol={tol:.3e} (control {d_ctl:.3e})")
            cfg = U.load_json(J / "config.json")
            env = U.load_json(J / "env.json")
            ck.check("c3.seeds", env["PYTHONHASHSEED"] == "1234" and len(cfg["seed_calls"]) == 3,
                     f"PYTHONHASHSEED={env['PYTHONHASHSEED']} seed_calls={cfg['seed_calls']}")
            import csv as _csv
            tl = list(_csv.DictReader((J / "train_log.csv").open()))
            losses = [r["loss_epoch_1"] for r in tl]
            ck.check("c3.train_log", len(tl) == 4 and all(l not in ("NA", "") for l in losses) and
                     all(r["train_steps"] == str(CFG["mlp"]["ad"]) and r["optimizer_iterations"] == str(CFG["mlp"]["ad"])
                         and r["epochs"] == "1" for r in tl),
                     f"loss_epoch_1={losses}; pasos={[r['train_steps'] for r in tl]} (= ad, 1 epoca batch 1)")
            ckpt = J / "ckpt"
            need = [ckpt / f"ae{i}" / s / "saved_model.pb" for i in range(4) for s in ("model", "encoder", "decoder")]
            need += [ckpt / "model.pkl", ckpt / "output.pkl"] + [ckpt / f"ae{i}" / "class.pkl" for i in range(4)]
            ck.check("c3.ckpt", all(p.exists() for p in need), f"{sum(p.exists() for p in need)}/{len(need)} ficheros")
            arr = U.load_json(J / "arrays.json")
            ck.check("c3.ae_errors", arr["ae_errors.npy"]["dtype"] == "float64" and
                     arr["ae_errors.npy"]["selfcheck_output_layer_from_saved_file"]["all_equal"] is True, "float64 + autochequeo")
        else:
            ck.check("c3.done", False, f"mlp sin DONE o sin salida directa (rc={rcs.get('run_mlp')}, {rcs.get('dir_mlp1')})")
    except Exception:
        import traceback
        traceback.print_exc()
        ck.check("c.exception", False, "excepcion en el test")
    rc = ck.finish()
    U.cleanup(root, rc == 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
