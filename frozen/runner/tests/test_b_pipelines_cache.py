#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(b) Stream pipelines and cache of the job runner (CPU): original and corrected KNAD loaders, benign guard, cache integrity."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _util as U
from _util import C

sys.path.insert(0, str(U.CODE_ROOT))
import numpy as np
import pandas as pd
import pipelines


def spec_for(dataset="KNAD:OS_Scan", pipeline="orig", variant="stat"):
    return C.validate_row(U.make_row(dataset=dataset, pipeline=pipeline, variant=variant), test_mode=True)


def load(spec, root, data_root, cache="cache", **kw):
    return pipelines.load_stream(spec, cache_root=str(root / cache), data_root=str(data_root), code_root=str(U.CODE_ROOT),
                                 test_mode=kw.pop("test_mode", True), log=lambda m: None, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-aci", action="store_true")
    a = ap.parse_args()
    ck = U.Checker("test_b_pipelines_cache")
    root = U.make_root("b")
    try:
        fx = root / "data"
        att = fx / "KITSUNE" / "OS Scan"
        U.copy_head_lines(U.KNAD_OS / "OS_Scan_dataset.csv", att / "OS_Scan_dataset.csv", 5001)
        U.copy_head_lines(U.KNAD_OS / "OS_Scan_labels.csv", att / "OS_Scan_labels.csv", 5002)
        s_orig = spec_for()
        s_corr = spec_for(pipeline="corrected")

        code = ("import sys,json; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
                "sys.path.append(sys.argv[5]); import common as C, "
                "pipelines; from _util import make_row; s = C.validate_row(make_row(), test_mode=True); "
                "x, y, m = pipelines.load_stream(s, cache_root=sys.argv[3], data_root=sys.argv[4], code_root=sys.argv[5], "
                "test_mode=True, log=lambda *_: None, cache_wait_s=60); "
                "print('JSON=' + json.dumps({'x': m['x']['sha256'], 'y': m['y']['sha256'], 'built': m.get('built_now', False)}))")
        cc = root / "cache_conc"
        procs = [subprocess.Popen([U.PY, "-c", code, str(U.RUNNER_DIR), str(U.TESTS_DIR), str(cc), str(fx), str(U.CODE_ROOT)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=U.base_env())
                 for _ in range(2)]
        outs = [p.communicate(timeout=120)[0] for p in procs]
        js = []
        for o in outs:
            ln = [l for l in o.splitlines() if l.startswith("JSON=")]
            js.append(json.loads(ln[-1][5:]) if ln else None)
        ck.check("b4.concurrent", all(js) and js[0]["x"] == js[1]["x"] and js[0]["y"] == js[1]["y"] and
                 sum(j["built"] for j in js) == 1 and not any((cc / ".locks").iterdir()),
                 f"built={[j and j['built'] for j in js]}" if all(js) else f"salidas: {[o[-300:] for o in outs]}")

        import src.database.test_database.database as kdb
        kdb.KITSUNE_DATABASE_PATHS = [str(att)]
        x_cls, y_cls = kdb.KitsuneDatabase(regularized=False).get("OS_Scan")
        y_cls = y_cls[:len(x_cls)] if len(x_cls) != len(y_cls) else y_cls
        x_c, y_c, meta = load(s_orig, root, fx)
        ck.check("b1.class_eq_cache", np.array_equal(x_cls, x_c) and np.array_equal(y_cls, y_c) and x_c.shape == (5000, 115)
                 and x_c.dtype == np.float64 and x_cls.flags.f_contiguous == x_c.flags.f_contiguous and y_c.dtype == y_cls.dtype,
                 f"x {x_c.shape} {x_c.dtype} F={x_c.flags.f_contiguous}; y {y_c.shape} {y_c.dtype}")
        x_r, y_r, m_r = load(s_orig, root, fx, cache="cache_r", max_rows=5000)
        ck.check("b1.replica_eq_class", np.array_equal(x_r, x_cls) and np.array_equal(y_r, y_cls) and m_r["cached"] is False,
                 "replica (nrows) == clase")
        x_r3, y_r3, _ = load(s_orig, root, fx, cache="cache_r", max_rows=3000)
        ck.check("b1.replica_prefix", np.array_equal(x_r3, x_cls[:3000]) and np.array_equal(y_r3, y_cls[:3000]),
                 "replica con nrows=3000 == clase[:3000]")
        ck.check("b4.max_rows_not_cached", not (root / "cache_r").exists() or not any((root / "cache_r").glob("knad-*")),
                 "max_rows no crea directorio de cache")

        x_k, y_k, m_k = load(s_corr, root, fx, overrides={"train_rows": 3000})
        ck.check("b2.shift", x_k.shape == (5001, 115) and np.array_equal(x_c, x_k[1:]) and np.array_equal(y_c, y_k[:5000])
                 and len(x_k) == len(y_k), "orig.x == corrected.x[1:]; orig.y == corrected.y[:-1]; corrected alineado")
        real_h0 = pd.read_csv(U.KNAD_OS / "OS_Scan_dataset.csv", nrows=3000).to_numpy()
        real_hn = pd.read_csv(U.KNAD_OS / "OS_Scan_dataset.csv", header=None, nrows=3001).to_numpy()
        ck.check("b2.real_header", np.array_equal(real_h0, real_hn[1:]), "CSV real: header=0 consume el paquete 0")

        tb = (m_k.get("extra") or {}).get("train_benign_check", {})
        ck.check("b3.pass", tb.get("n_attack") == 0 and tb.get("rows") == [0, 3000], f"{tb}")
        fx2 = root / "data_attack"
        att2 = fx2 / "KITSUNE" / "OS Scan"
        att2.mkdir(parents=True)
        shutil.copy(att / "OS_Scan_dataset.csv", att2 / "OS_Scan_dataset.csv")
        lines = (att / "OS_Scan_labels.csv").read_text().splitlines(keepends=True)
        lines[1 + 1500] = lines[1 + 1500].rsplit(",", 1)[0] + ",1\n"
        (att2 / "OS_Scan_labels.csv").write_text("".join(lines))
        try:
            load(s_corr, root, fx2, overrides={"train_rows": 3000})
            ck.check("b3.fail", False, "no lanzo PipelineError")
        except C.PipelineError as e:
            msg = str(e)
            ck.check("b3.fail", "[0, 3000)" in msg and "1500" in msg and "D-7" in msg, msg[:200])

        key = pipelines.cache_key(s_orig)
        cdir = root / "cache" / key
        meta_j = json.loads((cdir / "meta.json").read_text())
        need = {"key", "sources", "builder", "x", "y", "expected_rows", "test_mode", "data_root", "semantics", "versions"}
        ck.check("b4.meta", need <= set(meta_j) and all({"path", "size", "mtime_ns", "sha256"} <= set(s) for s in meta_j["sources"])
                 and meta_j["test_mode"] is True and meta_j["expected_rows"] == 5000, f"claves={sorted(meta_j)[:12]}")
        ro = (os.stat(cdir).st_mode & 0o222) == 0 and all((os.stat(f).st_mode & 0o222) == 0 for f in cdir.iterdir())
        ck.check("b4.readonly", ro, "directorio y ficheros de la cache sin permiso de escritura (S15)")
        ck.check("b4.public_meta", meta.get("y_sha256") == meta_j["y"]["sha256"] and meta.get("y_path", "").endswith("y.npy")
                 and meta["verified_on_load"]["sha256_recomputed"] in (True, "calculado al construir (mismo fichero renombrado)"),
                 "config.cache registra y_path, y_sha256 y verificacion")

        def clone(name, mutate):
            d = root / name / key
            d.parent.mkdir(parents=True)
            shutil.copytree(cdir, d)
            for p in [d] + list(d.iterdir()):
                os.chmod(p, 0o755 if p.is_dir() else 0o644)
            mutate(d)
            return name

        def flip(d):
            with open(d / "x.npy", "r+b") as fh:
                fh.seek(-7, 2)
                b = fh.read(1)
                fh.seek(-7, 2)
                fh.write(bytes([b[0] ^ 0xFF]))

        def edit_meta(field, value):
            def f(d):
                m = json.loads((d / "meta.json").read_text())
                if field == "fp":
                    m["builder"]["fingerprint_sha256"] = "0" * 64
                else:
                    m[field] = value
                (d / "meta.json").write_text(json.dumps(m))
            return f

        for cid, name, mut, exc in (("b4.integrity", "c_flip", flip, C.CacheIntegrityError),
                                    ("b4.builder_mismatch", "c_fp", edit_meta("fp", None), C.CacheMismatch),
                                    ("b4.test_mode_mismatch", "c_tm", edit_meta("test_mode", False), C.CacheMismatch)):
            clone(name, mut)
            try:
                load(s_orig, root, fx, cache=name)
                ck.check(cid, False, "no lanzo excepcion")
            except exc as e:
                ck.check(cid, True, f"{type(e).__name__}: {str(e)[:120]}")
        fx3 = root / "data_copy"
        shutil.copytree(fx, fx3)
        try:
            load(s_orig, root, fx3)
            ck.check("b4.data_root_mismatch", False, "no lanzo excepcion")
        except C.CacheMismatch as e:
            ck.check("b4.data_root_mismatch", True, str(e)[:120])
        load(s_orig, root, fx3, cache="cache3")
        lab3 = fx3 / "KITSUNE" / "OS Scan" / "OS_Scan_labels.csv"
        st = os.stat(lab3)
        os.utime(lab3, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        try:
            load(s_orig, root, fx3, cache="cache3")
            ck.check("b4.source_mtime_mismatch", False, "no lanzo excepcion")
        except C.CacheMismatch as e:
            ck.check("b4.source_mtime_mismatch", True, str(e)[:120])
        try:
            pipelines.load_stream(s_orig, cache_root=str(C.CACHE_ROOT), data_root=str(fx), code_root=str(U.CODE_ROOT),
                                  test_mode=True, log=lambda m: None)
            ck.check("b4.test_mode_prod_cache", False, "no lanzo EnvError")
        except C.EnvError:
            ck.check("b4.test_mode_prod_cache", True, "test-mode no puede usar la cache de produccion (M3)")
        try:
            pipelines.load_stream(s_orig, cache_root=str(C.CACHE_ROOT), data_root=str(fx), code_root=str(U.CODE_ROOT),
                                  test_mode=False, log=lambda m: None)
            ck.check("b4.prod_requires_prod_data", False, "no lanzo EnvError")
        except C.EnvError:
            ck.check("b4.prod_requires_prod_data", True, "produccion exige data_root de produccion (M3)")

        msgs = []
        for ds in C.OTHER_DATASETS:
            try:
                C.check_supported(spec_for(dataset=ds, pipeline="corrected"))
                msgs.append(None)
            except NotImplementedError as e:
                msgs.append(str(e))
        other = spec_for(dataset="CIC2017", pipeline="corrected")
        other.kind = "sin_constructor"
        try:
            C.check_supported(other)
            msg_other = None
        except NotImplementedError as e:
            msg_other = str(e)
        ck.check("b5.corrected_supported", msgs == [None] * 3 and msg_other is not None and "D-7" in msg_other and
                 "T0" in msg_other and pipelines.builder_for(spec_for(dataset="CIC2018", pipeline="corrected"), None)
                 is pipelines.build_cic_aci_corrected,
                 f"CIC/ACI corrected admitidos; tipo sin constructor -> {(msg_other or '')[:100]}")

        import run_job as RJ
        s_prod = C.validate_row(U.make_row(block="T1-A-KNAD", dataset="KNAD:OS_Scan", pipeline="corrected", variant="lstm",
                                           seed="A"), test_mode=False)
        n_p = 1_697_851
        cm_p = {"extra": {"regions": {"train": [0, 200_000], "post_train": [200_000, n_p]}, "first_attack_index": 1_314_658,
                          "train_benign_check": {"rows": [0, 200_000], "n_attack": 0, "passed": True}}}
        reg_p = RJ.regions_for(s_prod, n_p, {"fm": 50_000, "ad": 150_000}, cm_p)
        c_p = {k: list(C.check_protocol_regions(reg_p, k)) for k in reg_p["detector_protocol_candidates"] if not k.startswith("_")}
        ck.check("b7.regions_prod", "evaluation" not in reg_p and reg_p["post_train"] == [200_000, n_p] and
                 set(c_p) == {"calib_first_05pct_post_train", "calib_first_10pct_post_train", "calib_first_20pct_post_train",
                              "calib_in_sample_train_tail"} and
                 c_p["calib_first_10pct_post_train"] == [[200_000, 349_785], [349_785, n_p]] and
                 c_p["calib_in_sample_train_tail"] == [[100_000, 200_000], [200_000, n_p]], f"{c_p}")
        try:
            RJ.regions_for(s_prod, n_p, {"fm": 50_000, "ad": 150_000}, {"extra": {"regions": {"train": [0, 150_000]}}})
            ck.check("b7.train_mismatch", False, "no lanzo FrozenConfigError")
        except C.FrozenConfigError as e:
            ck.check("b7.train_mismatch", True, str(e)[:120])
        reg_o = RJ.regions_for(s_orig, 1_697_850, {"fm": 50_000, "ad": 150_000}, {})
        try:
            C.check_protocol_regions(reg_o, "as_submitted_regression_only")
            ck.check("b7.orig_regression_only", False, "la candidata del envio no se rechazo")
        except C.ProtocolRegionError:
            ck.check("b7.orig_regression_only", C.check_protocol_regions(reg_o, "as_submitted_regression_only",
                                                                         allow_regression_only=True)[1] == [0, 1_697_850],
                     "post-proceso enviado: solo con allow_regression_only")

        import instrument as I

        class _Out:
            mean, std, norm_min, norm_max = np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)

            def execute(self, E):
                return np.asarray(E).sum(axis=1)

        class _KN:
            pass

        class _Obs:
            exec_blocks = [{"start_row": 30, "n_rows": 20}]
            t = {}

        E8 = np.random.default_rng(0).random((50, 4))
        E8[:10] = 0.0
        E8[35, 2] = np.nan
        E8[45:, :] = np.nan
        kn8 = _KN()
        kn8.tracking = {"exec_loss": [E8[:, i].copy() for i in range(4)]}
        kn8.output_autoencoder = _Out()
        kn8.current_error = list(E8.sum(axis=1))
        kn8.trained_instances, kn8.autoencoders, kn8.feature_map = 50, [], []
        tmp8 = root / "b8"
        tmp8.mkdir()
        np.save(tmp8 / "scores.npy", np.log10(np.array(kn8.current_error) + 1))
        an8 = []
        arr8, _ = I.capture_after_loop(kn8, None, _Obs(), tmp8, s_orig, {"fm": 10, "ad": 20, "k": 4},
                                       {"consistency": {"feature_map_equal": True}}, an8)
        sc8 = arr8.get("ae_errors.npy", {}).get("selfcheck_output_layer_from_saved_file", {})
        ck.check("b8.nan_selfcheck", sc8.get("all_equal") is True and not any("autochequeo" in x for x in an8) and
                 sum(s["n_nan_current_error"] for s in sc8.get("segments", [])) == 6,
                 f"all_equal={sc8.get('all_equal')} anomalias={an8} nan={[s['n_nan_current_error'] for s in sc8.get('segments', [])]}")

        info9 = I.save_labels(np.array([0, 1, 1, 0], dtype=np.int64), tmp8)
        try:
            I.save_labels(np.array([0, 300]), tmp8)
            bad9 = False
        except ValueError:
            bad9 = True
        ck.check("b9.labels", info9["dtype"] == "int8" and info9["values"] == [0, 1] and bad9, f"{info9['dtype']} {info9['values']} "
                                                                                              f"rechaza int no representable={bad9}")

        def rejects(mut, k):
            rr = json.loads(json.dumps(reg_p))
            mut(rr["detector_protocol_candidates"][k])
            try:
                C.check_protocol_regions(rr, k)
                return False
            except C.ProtocolRegionError:
                return True

        k10 = "calib_first_10pct_post_train"
        rej10 = {
            "calib_despues_de_eval": rejects(lambda c: c.update(calib=[n_p - 1000, n_p], eval=[300_000, n_p - 1000]), k10),
            "calib_en_train_sin_marca": rejects(lambda c: c.update(calib=[150_000, 200_000]), k10),
            "calib_vacia": rejects(lambda c: c.update(calib=[200_000, 200_000]), k10),
            "in_sample_sin_marca": rejects(lambda c: c.update(calib_in_sample=False), "calib_in_sample_train_tail"),
        }
        ck.check("b10.contract_guards", all(rej10.values()), f"{rej10}")

        rej11 = []
        for cr in (C.CACHE_ROOT / "_sub_de_test", C.CACHE_ROOT.parent):
            try:
                pipelines.load_stream(s_orig, cache_root=str(cr), data_root=str(fx), code_root=str(U.CODE_ROOT),
                                      test_mode=True, log=lambda m: None)
                rej11.append(False)
            except C.EnvError:
                rej11.append(True)
        ck.check("b11.test_mode_cache_overlap", rej11 == [True, True] and not (C.CACHE_ROOT / "_sub_de_test").exists(),
                 f"subdirectorio y padre de la cache de produccion rechazados: {rej11}")

        if a.with_aci:
            s_aci = spec_for(dataset="ACI-IOT-2023", pipeline="orig")
            x_a, y_a, _ = pipelines.load_stream(s_aci, cache_root=str(root / "cache_aci"), data_root=str(C.DATA_ROOT),
                                                code_root=str(U.CODE_ROOT), test_mode=True, log=print)
            tgt = np.load(C.ORIG_RESULTS / "ACI-IOT-2023" / "target.npy")
            ck.check("b6.aci", x_a.shape[0] == 1_531_411 and np.array_equal(y_a, tgt) and y_a.dtype == tgt.dtype,
                     f"x={x_a.shape} y==target.npy")
    except Exception:
        import traceback
        traceback.print_exc()
        ck.check("b.exception", False, "excepcion en el test")
    rc = ck.finish()
    U.cleanup(root, rc == 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
