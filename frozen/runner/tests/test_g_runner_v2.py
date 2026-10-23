#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(g) Runner v2: stable job ids, candidate manifest, ablation columns, GPU-kernel classification and small CPU runs of T3 and T4-E."""
import contextlib
import json
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _util as U
from _util import C

sys.path.insert(0, str(U.RUNNER_DIR))
import make_manifest as MM

THREAD_VARS = ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "JISA_WORKER_THREADS")
SMALL = ["--max-rows", 4000, "--test-override", "fm_grace=1000,ad_grace=1500,execution_window=1000,train_rows=2500"]
T4E_ROWS = 20_000
T4E_ARGS = ["--max-rows", T4E_ROWS, "--test-override", "fm_grace=1000,ad_grace=1500,execution_window=4000,train_rows=2500"]


def row(**kw):
    """Fila con las columnas v2 (make_manifest.row) y el job_id derivado."""
    base = dict(block="TEST", priority=1, dataset="KNAD:OS_Scan", variant="stat", ar=True, pipeline="corrected", seed="A")
    base.update(kw)
    return MM.row(**base)


def mutate(r, **kw):
    r = dict(r)
    r.update({k: str(v) for k, v in kw.items()})
    r["job_id"] = C.derive_job_id(r)
    return r


def invalid(r, test_mode=False):
    try:
        C.validate_row(r, test_mode=test_mode)
        return False
    except C.InvalidJob:
        return True


def env_threads(**extra):
    env = U.base_env(**extra)
    for k in THREAD_VARS:
        if k not in extra:
            env.pop(k, None)
    return env


def fake_tf(relu_error=None, device="/job:localhost/replica:0/task:0/device:GPU:0", gpus=("/device:GPU:0",)):
    class T:
        def __init__(self, vals):
            self.vals, self.device = vals, device

        def numpy(self):
            return self.vals

    def relu(t):
        if relu_error:
            raise relu_error
        return T([max(0.0, v) for v in t])

    def sign(t):
        return T([(v > 0) - (v < 0) for v in t])

    return types.SimpleNamespace(
        config=types.SimpleNamespace(list_logical_devices=lambda kind: [types.SimpleNamespace(name=g) for g in gpus]),
        device=lambda name: contextlib.nullcontext(), constant=lambda v, dtype=None: list(v), float32="float32",
        nn=types.SimpleNamespace(relu=relu), sign=sign)


def rejects_plain(regions, cand):
    """F1: sin with_exclusions, la evaluacion de un job T4-E (contiene el pool entrenado) no se entrega como intervalo."""
    try:
        C.check_protocol_regions(regions, cand)
        return False
    except C.ProtocolRegionError as e:
        return "with_exclusions" in str(e)


def write_labels(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write('"","x"\n')
        for i, v in enumerate(labels, start=1):
            fh.write(f'"{i}",{int(v)}\n')


def main():
    ck = U.Checker("test_g_runner_v2")
    root = U.make_root("g")
    extra_roots = []
    R = root / "results"
    try:
        man_v1 = U.RUNNER_DIR / "manifest.csv"
        rows_man = C.read_manifest(man_v1)
        ck.check("g1.generate_equals_manifest", MM.to_csv(MM.generate()) == man_v1.read_text(),
                 f"make_manifest.generate() (ACTIVE={MM.ACTIVE}) == manifest.csv byte a byte (lo que exige --check)")
        bad = [r["job_id"] for r in rows_man if C.derive_job_id(r) != r["job_id"] or invalid(r)]
        ck.check("g1.manifest_rows_valid_same_id", not bad, f"{len(rows_man)} filas; problemas={bad[:3]}")
        p = subprocess.run(["git", "-C", str(U.REPO), "show", "runner-v1:revision/experiments/manifest.csv"],
                           capture_output=True, text=True)
        if p.returncode == 0:
            tag_lines = p.stdout.splitlines()
            man_lines = man_v1.read_text().splitlines()
            pos = {ln: i for i, ln in enumerate(man_lines)}
            idx = [pos.get(ln) for ln in tag_lines]
            ck.check("g1.tag_runner_v1_subset", None not in idx and idx == sorted(idx) and len(tag_lines) == 101 and
                     MM.to_csv(MM.generate_v1()) == p.stdout.rstrip("\n") + "\n",
                     f"las {len(tag_lines) - 1} filas del tag runner-v1 estan literales y en orden en manifest.csv; "
                     f"generate_v1() == tag")
        else:
            ck.warn("g1.tag_runner_v1_subset", f"sin tag runner-v1 legible: {p.stderr.strip()[:120]}")
        golden = {"X2__knad-os_scan__stat__arT__orig.v1__L800__k4__sNone__r1",
                  "S-L2000__knad-active_wiretap__mha__arT__corrected.v1__L2000__k4__sA__r1",
                  "T1-A-KNAD__knad-active_wiretap__lstm__arT__corrected.v1__L800__k4__sA__r1"}
        ck.check("g1.golden_ids", golden <= {r["job_id"] for r in rows_man}, "formato v1 del job_id intacto")
        act0 = MM.ACTIVE
        try:
            MM.ACTIVE = "v2"
            promoted = MM.to_csv(MM.generate())
            MM.ACTIVE = "v1"
            back_v1 = MM.to_csv(MM.generate())
        except RecursionError:
            promoted = back_v1 = None
        finally:
            MM.ACTIVE = act0
        ck.check("g1.promotion_active_switch", promoted == MM.to_csv(MM.generate_v2(MM.ACTIVE_SECOND)) and
                 back_v1 == MM.to_csv(MM.generate_v1()),
                 f"ACTIVE='v2' -> generate() == generate_v2(ACTIVE_SECOND) (sin RecursionError); ACTIVE='v1' -> "
                 f"generate_v1() (ACTIVE en disco = {act0})")

        cand = MM.generate_v2()
        text = MM.to_csv(cand)
        errs_v1 = MM.check_v1_preserved(None, cand, text)
        ck.check("g2.v1_preserved", not errs_v1, f"{errs_v1[:2]}")
        fam = {}
        for r in cand:
            fam[r["block"]] = fam.get(r["block"], 0) + 1
        exp = {"T1-A-ACI": 12, "T1-A-CIC2017": 12, "T1-A-CIC2018": 12, "T2-min": 8, "T2-ext": 8, "T2-B": 8,
               "T3-min-a": 81, "T3-min-b": 18, "T3-min-c": 6, "T3-ext-a": 6, "T3-ext-b": 6, "T4-E": 6}
        t1b = sum(v for k, v in fam.items() if k.startswith("T1-B-"))
        t1c = sum(v for k, v in fam.items() if k.startswith("T1-C-"))
        ck.check("g2.counts", all(fam.get(k) == v for k, v in exp.items()) and t1b == 90 and t1c == 90 and len(cand) == 463
                 and "S-w6" not in fam and "S-w6" not in C.KNOWN_BLOCKS,
                 f"{len(cand)} filas (sin S-w6, M4); T1-B={t1b} T1-C={t1c}; {fam}")
        ids = [r["job_id"] for r in cand]
        inv = []
        for r in cand:
            try:
                s = C.validate_row(r, test_mode=False)
                if s.variant == "stat" and s.save_ckpt:
                    inv.append(f"{r['job_id']}: stat con ckpt")
            except C.InvalidJob as e:
                inv.append(f"{r['job_id']}: {e}")
        ck.check("g2.all_valid_unique", len(set(ids)) == len(ids) and not inv, f"invalidas={inv[:2]}")

        def prio_range(pred):
            ps = [int(r["priority"]) for r in cand if pred(r)]
            return (min(ps), max(ps)) if ps else (None, None)

        order = [("P0 X2+S v1", lambda r: r["block"][:2] in ("X2", "S-")),
                 ("P1 T1-A", lambda r: r["block"].startswith("T1-A-")),
                 ("P2 T1-B", lambda r: r["block"].startswith("T1-B-")),
                 ("P3", lambda r: r["block"] in ("T2-min", "T3-min-a", "T3-min-b", "T3-min-c", "T4-E")),
                 ("P4 T1-C", lambda r: r["block"].startswith("T1-C-")),
                 ("P5", lambda r: r["block"] in ("T2-ext", "T3-ext-a", "T3-ext-b", "T2-B"))]
        ranges = [prio_range(f) for _, f in order]
        ok_order = all(ranges[i][1] < ranges[i + 1][0] for i in range(len(ranges) - 1))
        ck.check("g2.priority_order_P0_P5", ok_order, f"{list(zip([n for n, _ in order], ranges))}")
        p3 = [prio_range(lambda r, b=b: r["block"] == b) for b in ("T2-min", "T3-min-a", "T3-min-b", "T3-min-c", "T4-E")]
        p5 = [prio_range(lambda r, b=b: r["block"] == b) for b in ("T2-ext", "T3-ext-a", "T3-ext-b", "T2-B")]
        ck.check("g2.priority_within_P3_P5", all(a[1] < b[0] for a, b in zip(p3, p3[1:])) and
                 all(a[1] < b[0] for a, b in zip(p5, p5[1:])), f"P3={p3} P5={p5}")
        contiguous = []
        for pref in ("T1-B-", "T1-C-", "T3-min-a"):
            rs = sorted([r for r in cand if r["block"].startswith(pref)], key=lambda r: int(r["priority"]))
            seq = []
            for r in rs:
                if not seq or seq[-1] != r["dataset"]:
                    seq.append(r["dataset"])
            contiguous.append(seq == MM.ALL9_ORDER)
        ck.check("g2.datasets_complete_order", all(contiguous) and MM.ALL9_ORDER[0] == "KNAD:Active_Wiretap" and
                 [MM.STREAM_N[d] for d in MM.ALL9_ORDER[1:]] == sorted(MM.STREAM_N[d] for d in MM.ALL9_ORDER[1:]),
                 f"{contiguous}; orden={MM.ALL9_ORDER}")
        seeds_ok = all(r["seed"] == ("B" if r["block"] in ("T2-B",) or r["block"].startswith("T1-B-") else
                                     "C" if r["block"].startswith("T1-C-") else "A")
                       for r in cand if r["block"] not in ("X2",))
        no_det = not [r for r in cand if r["block"][:4] in ("T1-B", "T1-C") and r["variant"] in ("original", "stat")]
        ck.check("g2.seeds_and_deterministic_once", seeds_ok and no_det, f"semillas por bloque={seeds_ok}; sin original/stat "
                                                                           f"en T1-B/C={no_det}")
        r_rand = [r for r in cand if r["block"] == "T3-min-b"][0]
        r_t4 = [r for r in cand if r["block"] == "T4-E"][0]
        ck.check("g2.job_id_suffixes", "__grandom-p101__sA__r1" in r_rand["job_id"] and "__c5__sA__r1" in r_t4["job_id"],
                 f"{r_rand['job_id']} | {r_t4['job_id']}")
        cand_os = MM.generate_v2("KNAD:OS_Scan")
        abl_ds = {r["dataset"] for r in cand_os if r["block"] in ("T2-min", "T2-ext", "T2-B", "T3-min-b", "T3-min-c",
                                                                   "T3-ext-a", "T3-ext-b")}
        ck.check("g2.fallback_os_scan", abl_ds == {"KNAD:Active_Wiretap", "KNAD:OS_Scan"} and len(cand_os) == 463 and
                 {r["dataset"] for r in cand_os if r["block"] == "T3-min-a"} == set(MM.ALL9_ORDER),
                 f"datasets de ablacion con respaldo: {sorted(abl_ds)}")
        cout = root / "cand.csv"
        p = subprocess.run([U.SYS_PY, str(U.RUNNER_DIR / "make_manifest.py"), "--write-candidate", "--candidate-out", str(cout),
                            "--no-label-guard", "--results-root", str(R)], capture_output=True, text=True, timeout=120)
        p2 = subprocess.run([U.SYS_PY, str(U.RUNNER_DIR / "make_manifest.py"), "--check-candidate", "--candidate-out", str(cout),
                             "--no-label-guard", "--results-root", str(R)], capture_output=True, text=True, timeout=120)
        ck.check("g2.cli_write_check", p.returncode == 0 and p2.returncode == 0 and cout.read_text() == text,
                 f"rc={p.returncode}/{p2.returncode}: {(p.stdout + p.stderr).strip()[-200:]}")
        p = subprocess.run([U.SYS_PY, str(U.RUNNER_DIR / "make_manifest.py"), "--write-candidate", "--candidate-out",
                            str(root / "manifest.csv"), "--no-label-guard"], capture_output=True, text=True, timeout=120)
        ck.check("g2.never_manifest_csv", p.returncode == 1 and not (root / "manifest.csv").exists(), p.stdout.strip()[:120])
        t2b = [r for r in cand if r["block"] == "T2-B"]
        ck.check("g2.t2b_reference_t1b", t2b and all("referencia T1-B-" in r["notes"] for r in t2b) and
                 all("referencia T1-A-" in r["notes"] for r in cand if r["block"] == "T2-min"),
                 f"{t2b[0]['notes'] if t2b else None}")
        mixed = cand + [r for r in cand_os if r["block"] == "T2-min" and r["dataset"] == "KNAD:OS_Scan"]
        errs_mixed, _g = MM.validate(mixed, check_labels=False)
        ck.check("g2.single_second_dataset", any("segundo dataset" in e for e in errs_mixed) and
                 not any("segundo dataset" in e for e in MM.validate(cand, check_labels=False)[0]),
                 f"{[e for e in errs_mixed if 'segundo dataset' in e][:1]}")
        wout = root / "w" / "manifest.csv"
        wout.parent.mkdir(parents=True, exist_ok=True)
        text_mixed = MM.to_csv(sorted(mixed, key=lambda r: (int(r["priority"]), r["job_id"])))
        wout.write_text(text_mixed)
        p = subprocess.run([U.SYS_PY, str(U.RUNNER_DIR / "make_manifest.py"), "--write", "--out", str(wout),
                            "--no-label-guard", "--results-root", str(R)], capture_output=True, text=True, timeout=120)
        ck.check("g2.write_refuses_removal", p.returncode == 1 and wout.read_text() == text_mixed and "desaparecerian" in p.stdout,
                 f"rc={p.returncode}: {p.stdout.strip()[:160]}")

        t1 = row(block="T1-A-KNAD", dataset="KNAD:Active_Wiretap", variant="lstm", save_ckpt=1)
        t2 = row(block="T2-min", dataset="KNAD:Active_Wiretap", variant="lstm", window=200, save_ckpt=1)
        t3k = row(block="T3-min-a", dataset="CIC2018", variant="stat", k_clusters=8)
        t3r = row(block="T3-min-b", dataset="CIC2017", variant="mlp", grouping="random", partition_seed=202, save_ckpt=1)
        t4 = row(block="T4-E", dataset="KNAD:Active_Wiretap", variant="lstm", contamination=20, save_ckpt=1)
        t1b = row(block="T1-B-CIC2017", dataset="CIC2017", variant="mha", ar=False, seed="B", save_ckpt=1)
        valid = [t1, t2, t3k, t3r, t4, t1b, mutate(t2, window=2000), mutate(t3k, k_clusters=1), mutate(t4, contamination=5)]
        ck.check("g3.accepted", not any(invalid(r) for r in valid), f"{[r['job_id'] for r in valid if invalid(r)][:2]}")
        rejects = {
            "T1 window 2000": mutate(t1, window=2000), "T2 window 500": mutate(t2, window=500),
            "T2 window 800": mutate(t2, window=800), "T2 conv1d": mutate(t2, variant="conv1d"),
            "T2-min TSR": mutate(t2, ar=False, paradigm="TSR"), "T2-min seed B": mutate(t2, seed="B"),
            "T2 dataset SSDP": mutate(t2, dataset="KNAD:SSDP_Flood"), "T1 k=8": mutate(t1, k_clusters=8),
            "T3 k=3": mutate(t3k, k_clusters=3), "T3 k=4": mutate(t3k, k_clusters=4),
            "T3-min-a lstm": mutate(t3k, variant="lstm"), "T3 k con partition_seed": mutate(t3k, partition_seed=101),
            "T3 random sin seed": mutate(t3r, partition_seed="NA"), "T3 random seed 7": mutate(t3r, partition_seed=7),
            "T3 random k=2": mutate(t3r, k_clusters=2), "T1 random": mutate(t1, grouping="random", partition_seed=101),
            "T2 contaminacion": mutate(t2, contamination=5), "T4-E c=10": mutate(t4, contamination=10),
            "T4-E CIC2017": mutate(t4, dataset="CIC2017"), "T4-E conv2d": mutate(t4, variant="conv2d"),
            "warmup_frac": mutate(t1, warmup_frac="0.1"), "decontaminate": mutate(t1, decontaminate="1"),
            "contaminacion no entera": mutate(t4, contamination="0.2"), "T1-B stat": mutate(t1b, variant="stat", save_ckpt=0),
            "T1-B seed A": mutate(t1b, seed="A"), "T1-B-CIC2017 con KNAD": mutate(t1b, dataset="KNAD:Fuzzing"),
            "T4-E orig": mutate(t4, pipeline="orig"),
            "job_id sin sufijo": dict(t3r, job_id=t3r["job_id"].replace("__grandom-p202", "")),
            "partition_seed no canonico": mutate(t3r, partition_seed="0202"), "contaminacion +20": mutate(t4, contamination="+20"),
            "k no canonico": mutate(t3k, k_clusters="08"), "T4-E stat": mutate(t4, variant="stat", save_ckpt=0),
        }
        acc = [k for k, r in rejects.items() if not invalid(r)]
        ck.check("g3.rejected", not acc, f"aceptadas por error: {acc}")
        ck.check("g3.test_mode_dataset_relaxed", not invalid(mutate(t4, dataset="KNAD:OS_Scan"), test_mode=True) and
                 invalid(mutate(t4, dataset="KNAD:OS_Scan")), "T4-E en OS_Scan solo con --test-mode (fixtures)")
        s4 = C.validate_row(t4)
        ck.check("g3.ablation_descriptor", s4.ablation() == {"block": "T4-E", "factor": "contamination", "value": 20,
                                                             "reference": 0, "reference_block": "T1-A-KNAD",
                                                             "plan": "§8.5 T4-E", "only_this_factor_changes": True} and
                 C.validate_row(t1).ablation() is None and
                 C.validate_row(mutate(t2, block="T2-B", seed="B")).ablation()["reference_block"] == "T1-B-KNAD" and
                 C.validate_row(mutate(t3r, block="T3-min-b")).ablation()["reference_block"] == "T1-A-CIC2017",
                 f"{s4.ablation()}")

        texts = {"Graph execution error: 'cuModuleLoadData(&module, fatbin.data())' failed with 'CUDA_ERROR_INVALID_IMAGE'": True,
                 "cuLaunchKernel failed": True, "CUDA_ERROR_OUT_OF_MEMORY: out of memory": False,
                 "ValueError: shapes (3,) and (4,) not aligned": False,
                 "CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered": False,
                 "failed call to cuInit: CUDA_ERROR_NO_BINARY_FOR_GPU": True}
        ck.check("g4.regex", all(C.looks_like_gpu_kernel_error(t) == v for t, v in texts.items()) and
                 C.looks_like_oom("CUDA_ERROR_OUT_OF_MEMORY"), "INVALID_IMAGE/cuLaunchKernel -> ENV; OOM sigue siendo OOM")
        import run_job as RJ
        cases = {}
        for name, tf_obj in (("kernel_error", fake_tf(relu_error=RuntimeError("InternalError: 'cuModuleLoadData' failed "
                                                                                "with 'CUDA_ERROR_INVALID_IMAGE' [Op:Relu]"))),
                             ("soft_placement_cpu", fake_tf(device="/job:localhost/replica:0/task:0/device:CPU:0")),
                             ("no_gpu", fake_tf(gpus=()))):
            try:
                RJ.resolve_device(tf_obj, "0")
                cases[name] = "sin EnvError"
            except C.EnvError as e:
                cases[name] = "ok" if (name == "no_gpu" or "DEV-029" in str(e)) else str(e)[:80]
        ok_dev = RJ.resolve_device(fake_tf(), "0") == "/GPU:0" and RJ.resolve_device(None, "cpu") == "/CPU:0"
        ck.check("g4.resolve_device", ok_dev and all(v == "ok" for v in cases.values()), f"{cases}; ok_path={ok_dev}")
        rk = U.make_row(variant="_fake_gpukernel", priority=1)
        rok = U.make_row(variant="_fake_ok", priority=2, rep=1)
        rok1 = U.make_row(variant="_fake_ok", priority=3, rep=2)
        rok0 = U.make_row(variant="_fake_ok", priority=4, rep=3)
        man = root / "m_g4.csv"
        U.write_manifest(man, [rk])
        rc, out = U.run_job(root, man, rk["job_id"])
        st = R / ".state" / rk["job_id"]
        ck.check("g4.run_job_env", rc == C.ENV and C.read_fail_count(R, rk["job_id"]) == 0 and (st / "env_1.txt").exists()
                 and not (R / rk["job_id"]).exists() and not (R / ".locks" / rk["job_id"]).exists(),
                 f"rc={rc} fail_count={C.read_fail_count(R, rk['job_id'])} {out.strip().splitlines()[-1][:160]}")
        rc, out = U.run_worker(root, man, ["--env-sleep-s", "0.2"], timeout=120)
        ck.check("g4.worker_exits_after_3_env", rc == C.ENV and C.read_fail_count(R, rk["job_id"]) == 0 and
                 len(list(st.glob("env_*.txt"))) == 4, f"rc={rc} eventos env={len(list(st.glob('env_*.txt')))}")

        results = {}
        for name, r, extra, env in (("t3", rok, ["--threads", "3"], env_threads()),
                                    ("t1", rok1, ["--threads", "1"], env_threads()),
                                    ("none", rok0, [], env_threads())):
            m = root / f"m_g5_{name}.csv"
            U.write_manifest(m, [r])
            rc, out = U.run_worker(root, m, extra, env=env, timeout=120)
            e = U.load_json(R / r["job_id"] / "env.json") if (R / r["job_id"] / "DONE").exists() else {}
            results[name] = (rc, e.get("threads"), out)
        t3, t1_, t0 = results["t3"][1] or {}, results["t1"][1] or {}, results["none"][1] or {}
        ck.check("g5.threads_3", results["t3"][0] == 0 and t3 == {"TF_NUM_INTRAOP_THREADS": "3", "TF_NUM_INTEROP_THREADS": "2",
                                                                "OMP_NUM_THREADS": "3", "JISA_WORKER_THREADS": "3"} and
                 "threads=TF_NUM_INTRAOP_THREADS=3" in results["t3"][2], f"{t3}")
        ck.check("g5.threads_1", results["t1"][0] == 0 and t1_.get("TF_NUM_INTEROP_THREADS") == "1" and
                 t1_.get("TF_NUM_INTRAOP_THREADS") == "1", f"{t1_}")
        ck.check("g5.threads_untouched", results["none"][0] == 0 and all(v is None for v in t0.values()) and
                 "heredados" in results["none"][2], f"{t0}")
        rc, out = U.run_worker(root, root / "m_g5_t3.csv", ["--threads", "0"], timeout=60)
        ck.check("g5.threads_invalid", rc == C.INVALID, f"rc={rc}")

        import numpy as np
        import instrument as I

        def lab(sizes):
            return np.concatenate([np.full(s, i - 1) for i, s in enumerate(sizes)]).astype(int)

        hf = {"aw_k8": I.hotfix_outcome(lab([31, 8, 20, 26, 10, 10, 10]), 8),
              "k1_noise": I.hotfix_outcome(lab([115]), 1),
              "k1_one_cluster": I.hotfix_outcome(np.zeros(115, int), 1),
              "k2_flat": I.hotfix_outcome(lab([4, 4]), 2),
              "k2_oscillates": I.hotfix_outcome(lab([15, 7]), 2)}
        ck.check("g6.hotfix_outcome", hf["aw_k8"]["status"] == "ok" and hf["aw_k8"]["sizes_after"] == [23, 8, 20, 26, 10, 10, 10, 8]
                 and hf["k1_noise"]["status"] == "ok" and hf["k1_one_cluster"]["status"] == "crash_pop_empty" and
                 hf["k2_flat"]["status"] == "no_termina" and hf["k2_oscillates"]["status"] == "no_termina",
                 f"{ {k: v['status'] for k, v in hf.items()} }")
        groups, perm = I.random_partition(115, [82, 15, 8, 10], 101)
        ck.check("g6.random_partition", [len(g) for g in groups] == [82, 15, 8, 10] and
                 sorted(v for g in groups for v in g) == list(range(115)) and
                 groups[0] == [int(v) for v in np.random.default_rng(101).permutation(115)[:82]], "permutacion en grupos consecutivos")

        s_prod = C.validate_row(row(block="T4-E", dataset="KNAD:Active_Wiretap", variant="mlp", contamination=20), test_mode=True)
        yy = np.zeros(T4E_ROWS, dtype=np.int64)
        yy[6000:7000] = 1
        xx = np.arange(T4E_ROWS * 3, dtype=np.float64).reshape(T4E_ROWS, 3)
        cfg = {"fm": 1000, "ad": 1500}
        fake_code = root / "code_lc"
        lc = {"knad_active_wiretap": {"calib": {"rows": 1750, "fit_rows_dedup": 1750},
                                      "t4p_donor_pool": {"attack_rows_needed": 350, "pool": [6000, 6350],
                                                         "excluded_with_guards": [5200, 7150]}}}
        (fake_code / Path(I.LABEL_COUNTS_REL).parent).mkdir(parents=True)
        (fake_code / I.LABEL_COUNTS_REL).write_text(json.dumps(lc))
        x2, y2, info, donor = I.t4e_inject(s_prod, xx.copy(), yy, cfg, fake_code, test_mode=False)
        ok7 = (info["injected_rows"] == [1600, 1900] and info["B_c"] == 300 and list(donor[[0, -1]]) == [6000, 6299] and
               np.array_equal(x2[1600:1900], xx[6000:6300]) and y2[1600:1900].all() and int(y2[:2500].sum()) == 300 and
               np.array_equal(x2[:1600], xx[:1600]) and info["eval_exclude_with_guards"] == [5200, 7150] and
               info["label_counts_crosscheck"] is True and int(yy[:2500].sum()) == 0)
        lc["knad_active_wiretap"]["t4p_donor_pool"]["pool"] = [6000, 6351]
        (fake_code / I.LABEL_COUNTS_REL).write_text(json.dumps(lc))
        try:
            I.t4e_inject(s_prod, xx.copy(), yy, cfg, fake_code, test_mode=False)
            halt7 = False
        except C.PipelineError:
            halt7 = True
        ck.check("g7.t4e_inject", ok7 and halt7, f"info={ {k: info[k] for k in ('injected_rows', 'donor_pool', 'eval_exclude_with_guards')} }; "
                                                 f"discrepancia con label_counts -> HALT={halt7}")
        halts3 = {}
        for name, pool in (("sin_clave", None), ("sin_guardas", {"attack_rows_needed": 350, "pool": [6000, 6350]})):
            ent = {"calib": {"rows": 1750, "fit_rows_dedup": 1750}}
            if pool is not None:
                ent["t4p_donor_pool"] = pool
            (fake_code / I.LABEL_COUNTS_REL).write_text(json.dumps({"knad_active_wiretap": ent}))
            try:
                I.t4e_inject(s_prod, xx.copy(), yy, cfg, fake_code, test_mode=False)
                halts3[name] = False
            except C.PipelineError as e:
                halts3[name] = "t4p_donor_pool" in str(e)
        ck.check("g7.t4e_pool_fail_closed", all(halts3.values()), f"{halts3}")

        fx = root / "data"
        att = fx / "KITSUNE" / "OS Scan"
        U.copy_head_lines(U.KNAD_OS / "OS_Scan_dataset.csv", att / "OS_Scan_dataset.csv", T4E_ROWS)
        lab_os = np.zeros(T4E_ROWS, dtype=int)
        lab_os[6000:7000] = 1
        write_labels(att / "OS_Scan_labels.csv", lab_os)
        fx_c = root / "data_const"
        att_c = fx_c / "KITSUNE" / "OS Scan"
        att_c.mkdir(parents=True)
        (att_c / "OS_Scan_dataset.csv").write_text(("," .join(["1.0"] * 115) + "\n") * 4000)
        write_labels(att_c / "OS_Scan_labels.csv", np.zeros(4000, dtype=int))
        r_k2 = row(block="T3-min-a", variant="stat", k_clusters=2, priority=1)
        r_rand = row(block="T3-min-b", variant="stat", grouping="random", partition_seed=101, priority=2)
        r_t4e = row(block="T4-E", variant="mlp", contamination=20, priority=3)
        r_k1c = row(block="T3-min-a", variant="stat", k_clusters=1, priority=4)
        man8 = root / "m_g8.csv"
        U.write_manifest(man8, [r_k2, r_rand, r_t4e, r_k1c])
        env8 = U.base_env(TF_NUM_INTRAOP_THREADS=2, TF_NUM_INTEROP_THREADS=1, OMP_NUM_THREADS=2, JISA_WORKER_THREADS=2,
                          PYTHONHASHSEED=1234)
        logs = root / "logs"
        jobs = {"k2": (r_k2, SMALL + ["--data-root", fx]), "rand": (r_rand, SMALL + ["--data-root", fx]),
                "t4e": (r_t4e, T4E_ARGS + ["--data-root", fx]), "k1c": (r_k1c, SMALL + ["--data-root", fx_c])}
        procs = {k: U.popen_job(root, man8, r["job_id"], a, env=env8, log_path=logs / f"g8_{k}.log") for k, (r, a) in jobs.items()}
        code = ("import sys, json; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); sys.path.insert(0, sys.argv[3]); "
                "import common as C, instrument as I, make_manifest as MM\n"
                "out = {}\n"
                "def fc(**kw):\n"
                "    b = dict(block='TEST', priority=1, dataset='KNAD:Active_Wiretap', variant='lstm', ar=True, pipeline='corrected', seed='A')\n"
                "    b.update(kw); s = C.validate_row(MM.row(**b)); c = I.frozen_config(s, {}, False)\n"
                "    return {k: c[k] for k in ('fm', 'ad', 'L', 'k', 'grouping', 'partition_seed', 'contamination_pct')}\n"
                "out['t2'] = fc(block='T2-min', window=2000, save_ckpt=1)\n"
                "out['t3c'] = fc(block='T3-min-c', k_clusters=8, save_ckpt=1)\n"
                "out['t1'] = fc(block='T1-A-KNAD', save_ckpt=1)\n"
                "out['aci'] = fc(block='T1-A-ACI', dataset='ACI-IOT-2023', save_ckpt=1)\n"
                "print('JSON=' + json.dumps(out))\n")
        pfc = subprocess.run([U.PY, "-c", code, str(U.RUNNER_DIR), str(U.TESTS_DIR), str(U.CODE_ROOT)], env=env8,
                             capture_output=True, text=True, timeout=600)
        rcs = {k: p.wait(timeout=900) for k, p in procs.items()}
        ck.info(f"rc={rcs}")
        ln = [x for x in pfc.stdout.splitlines() if x.startswith("JSON=")]
        fcv = json.loads(ln[-1][5:]) if ln else {}
        ck.check("g8.frozen_config", fcv.get("t2", {}).get("L") == 2000 and fcv["t2"]["k"] == 4 and fcv["t2"]["ad"] == 150_000
                 and fcv.get("t3c", {}).get("k") == 8 and fcv["t3c"]["L"] == 800 and fcv.get("t1", {}).get("L") == 800 and
                 fcv.get("aci", {}).get("ad") == 22_264 and fcv["aci"]["fm"] == 50_000,
                 f"{fcv or pfc.stderr[-600:]}")

        J = R / r_k2["job_id"]
        if rcs["k2"] == 0 and (J / "DONE").exists():
            fs, cfg2, arr = U.load_json(J / "fm_stats.json"), U.load_json(J / "config.json"), U.load_json(J / "arrays.json")
            envj = U.load_json(J / "env.json")
            ck.check("g8.k2_done", fs["k"] == 2 and len(fs["feature_map"]) == 2 and fs["consistency"]["feature_map_equal"] and
                     fs.get("hotfix_prediction", {}).get("status") == "ok" and cfg2["hyperparameters_frozen"]["k"] == 2 and
                     cfg2["kitnet_config_check"]["autoencoder_config"]["autoencoder_size"] == 2 and
                     cfg2["ablation"]["factor"] == "k" and arr["ae_errors.npy"]["shape"] == [4000, 2] and
                     arr["ae_errors.npy"]["selfcheck_output_layer_from_saved_file"]["all_equal"] is True,
                     f"mapa {fs.get('sizes_final')} en {fs.get('iterations')} ajustes; hot-fix {fs.get('hotfix_prediction')}")
            th = envj.get("threads", {})
            ck.check("g8.env_threads", th.get("TF_NUM_INTRAOP_THREADS") == "2" and th.get("TF_NUM_INTEROP_THREADS") == "1" and
                     th.get("OMP_NUM_THREADS") == "2" and th.get("JISA_WORKER_THREADS") == "2" and
                     isinstance(th.get("process_threads_now"), int) and envj["env_vars"].get("TF_NUM_INTRAOP_THREADS") == "2",
                     f"{th}")
        else:
            ck.check("g8.k2_done", False, f"rc={rcs['k2']}\n{C.tail_text(logs / 'g8_k2.log', 2500)}")

        J = R / r_rand["job_id"]
        if rcs["rand"] == 0 and (J / "DONE").exists():
            fs, cfgr = U.load_json(J / "fm_stats.json"), U.load_json(J / "config.json")
            rg = fs.get("random_grouping", {})
            sizes = [len(g) for g in fs.get("feature_map_dbscan", [])]
            exp_groups, _ = I.random_partition(115, sizes, 101)
            ck.check("g8.random_grouping", fs["feature_map"] == exp_groups and rg.get("sizes") == sizes and
                     fs["consistency"] == {"feature_map_equal": True, "random_map_is_kn_feature_map": True,
                                           "sizes_equal_dbscan": True} and fs["feature_map"] != fs["feature_map_dbscan"] and
                     cfgr["regions"]["train"] == [0, 2500] and not cfgr.get("anomalies") and
                     fs.get("random_grouping_t1a_reference", {}).get("checked_against") == [],
                     f"tamanos DBSCAN {sizes}; anomalias={cfgr.get('anomalies')}")
            ok_root, bad_root = U.make_root("g-ref-ok"), U.make_root("g-ref-bad")
            extra_roots += [ok_root, bad_root]
            ref_id = "T1-A-KNAD__knad-os_scan__stat__arT__corrected.v1__L800__k4__sA__r1"
            for rr, fmap in ((ok_root, fs["feature_map_dbscan"]), (bad_root, [list(range(115)), [], [], []])):
                d = rr / "results" / ref_id
                d.mkdir(parents=True)
                (d / "fm_stats.json").write_text(json.dumps({"feature_map": fmap}))
                (d / "DONE").write_text("test\n")
                U.write_manifest(rr / "manifest.csv", [r_rand])
            pr = {name: U.popen_job(rr, rr / "manifest.csv", r_rand["job_id"], SMALL + ["--data-root", fx], env=env8,
                                    log_path=logs / f"g8_ref_{name}.log") for name, rr in (("ok", ok_root), ("bad", bad_root))}
            rr_rc = {k: p.wait(timeout=900) for k, p in pr.items()}
            fs_ok = U.load_json(ok_root / "results" / r_rand["job_id"] / "fm_stats.json") if rr_rc["ok"] == 0 else {}
            ck.check("g8.random_t1a_reference", rr_rc["ok"] == 0 and
                     fs_ok.get("random_grouping_t1a_reference", {}).get("checked_against") == [ref_id] and
                     rr_rc["bad"] == C.HALT and C.read_fail_count(bad_root / "results", r_rand["job_id"]) == 0 and
                     not (bad_root / "results" / r_rand["job_id"]).exists(),
                     f"rc={rr_rc}; {C.tail_text(logs / 'g8_ref_bad.log', 400).strip()[-200:]}")
        else:
            ck.check("g8.random_grouping", False, f"rc={rcs['rand']}\n{C.tail_text(logs / 'g8_rand.log', 2500)}")

        J = R / r_t4e["job_id"]
        if rcs["t4e"] == 0 and (J / "DONE").exists():
            import hashlib
            import pandas as pd
            cfgt = U.load_json(J / "config.json")
            t4 = cfgt["regions"].get("t4e_training_contamination", {})
            labs = np.load(J / "labels.npy")
            donor = np.load(J / "t4e_donor_rows.npy")
            xs = pd.read_csv(att / "OS_Scan_dataset.csv", header=None).to_numpy()
            h_after = hashlib.sha256(np.ascontiguousarray(xs[6000:6300]).tobytes()).hexdigest()
            exp_lab = lab_os.copy()
            exp_lab[1600:1900] = 1
            ck.check("g8.t4e_done", t4.get("injected_rows") == [1600, 1900] and t4.get("B_c") == 300 and
                     t4.get("calib") == [2500, 4250] and t4.get("donor_pool") == [6000, 6350] and
                     t4.get("eval_exclude_with_guards") == [5200, 7150] and t4.get("x_block_sha256_after") == h_after and
                     np.array_equal(labs, exp_lab) and list(donor[[0, -1]]) == [6000, 6299] and
                     cfgt["ablation"]["value"] == 20 and C.check_protocol_regions(cfgt["regions"], "calib_first_10pct_post_train",
                                                                                  with_exclusions=True)
                     == ([2500, 4250], [[4250, 5200], [7150, T4E_ROWS]]) and
                     cfgt["regions"]["detector_protocol_candidates"]["calib_first_10pct_post_train"]["eval_exclude"] == [[5200, 7150]] and
                     rejects_plain(cfgt["regions"], "calib_first_10pct_post_train") and
                     rejects_plain(cfgt["regions"], "calib_first_05pct_post_train"),
                     f"{ {k: t4.get(k) for k in ('injected_rows', 'calib', 'donor_pool', 'eval_exclude_with_guards')} }")
        else:
            ck.check("g8.t4e_done", False, f"rc={rcs['t4e']}\n{C.tail_text(logs / 'g8_t4e.log', 2500)}")

        st = R / ".state" / r_k1c["job_id"]
        ev = sorted(st.glob("nonexec_*.json")) if st.exists() else []
        detail = U.load_json(ev[0]) if ev else {}
        ck.check("g8.nonexec_k1", rcs["k1c"] == C.INVALID and len(ev) == 1 and C.read_fail_count(R, r_k1c["job_id"]) == 0 and
                 detail.get("hotfix_prediction", {}).get("status") == "crash_pop_empty" and
                 not (R / r_k1c["job_id"]).exists() and not (R / ".locks" / r_k1c["job_id"]).exists(),
                 f"rc={rcs['k1c']} eventos={len(ev)} {detail.get('message', '')[:160]}")
        man_ne = root / "m_g8_nonexec.csv"
        U.write_manifest(man_ne, [r_k1c])
        rc, out = U.run_worker(root, man_ne, SMALL + ["--data-root", fx_c], env=env8, timeout=120)
        ck.check("g8.worker_skips_nonexec", rc == 0 and "lanzo" not in out and len(list(st.glob("nonexec_*.json"))) == 1,
                 f"rc={rc}: {out.strip().splitlines()[-1][:160] if out.strip() else ''}")
        p = subprocess.run([U.SYS_PY, str(U.RUNNER_DIR / "status.py"), "--json", "--manifest", str(man_ne),
                            "--results-root", str(R)], capture_output=True, text=True, timeout=120)
        sj = json.loads(p.stdout) if p.returncode == 0 else {}
        ck.check("g8.status_nonexec", sj.get("per_block", {}).get(r_k1c["block"]) == {"no_ejecutable": 1} and
                 sj["eta"]["jobs_nonexecutable_excluded"] == 1 and sj["eta"]["jobs_without_tbar"] == 0 and
                 any(e.get("job_id") == r_k1c["job_id"] and e.get("nonexec") == 1 for e in sj.get("events_no_fail", [])),
                 f"{sj.get('per_block')} eta={sj.get('eta', {}).get('jobs_nonexecutable_excluded')} {p.stderr[-200:]}")

        r_t2 = row(block="T2-min", variant="mha", window=2000, save_ckpt=0, priority=1)
        s_t2 = C.validate_row(r_t2, test_mode=True)
        s_t1m = C.validate_row(row(block="T1-A-KNAD", dataset="KNAD:Active_Wiretap", variant="mha"), test_mode=True)
        R9u = root / "g9_unit"
        (R9u / ".state" / s_t1m.job_id).mkdir(parents=True)
        (R9u / ".state" / s_t1m.job_id / "oom_1.json").write_text(json.dumps({"counted": True}))
        ck.check("g9.override_scope", I.execution_bs_override(s_t1m, R9u) is None and I.execution_bs_override(s_t2, R9u) is None,
                 "T1 MHA con OOM contado y T2 MHA L=2000 sin OOM -> sin cambio (EXECUTION_BS congelado)")
        env9 = U.base_env(TF_NUM_INTRAOP_THREADS=1, TF_NUM_INTEROP_THREADS=1, OMP_NUM_THREADS=1, JISA_WORKER_THREADS=1,
                          PYTHONHASHSEED=1234)
        args9 = ["--max-rows", 4000, "--data-root", fx, "--test-override",
                 "fm_grace=1000,ad_grace=1500,execution_window=1000,train_rows=2500,sequence_length=20"]
        roots9, procs9 = {}, {}
        for name, ooms in (("ref", [False]), ("bs128", [True]), ("bs32", [True, True])):
            rr = U.make_root(f"g9-{name}")
            extra_roots.append(rr)
            U.write_manifest(rr / "manifest.csv", [r_t2])
            stt = rr / "results" / ".state" / r_t2["job_id"]
            stt.mkdir(parents=True)
            for i, counted in enumerate(ooms, start=1):
                (stt / f"oom_{i}.json").write_text(json.dumps({"counted": counted, "phase": "execute", "source": "test"}))
            roots9[name] = rr
            procs9[name] = U.popen_job(rr, rr / "manifest.csv", r_t2["job_id"], args9, env=env9, log_path=logs / f"g9_{name}.log")
        rc9 = {k: p.wait(timeout=1500) for k, p in procs9.items()}
        if all(v == 0 for v in rc9.values()):
            J9 = {k: roots9[k] / "results" / r_t2["job_id"] for k in roots9}
            cf9 = {k: U.load_json(J9[k] / "config.json") for k in J9}
            eq = {k: (J9[k] / "scores.npy").read_bytes() == (J9["ref"] / "scores.npy").read_bytes() and
                  (J9[k] / "ae_errors.npy").read_bytes() == (J9["ref"] / "ae_errors.npy").read_bytes() for k in ("bs128", "bs32")}
            ck.check("g9.execution_bs_equivalence", "execution_bs_override" not in cf9["ref"] and
                     cf9["bs128"].get("execution_bs_override", {}).get("value") == 128 and
                     cf9["bs32"].get("execution_bs_override", {}).get("value") == 32 and all(eq.values()),
                     f"scores/ae_errors byte a byte iguales a EXECUTION_BS=512: {eq}")
        else:
            ck.check("g9.execution_bs_equivalence", False,
                     f"rc={rc9}\n" + "\n".join(C.tail_text(logs / f"g9_{k}.log", 1200) for k, v in rc9.items() if v != 0))

        r_kill = U.make_row(variant="_fake_kill", priority=1, rep=10)
        R10 = {}
        for name in ("alone", "concurrent"):
            rr = U.make_root(f"g10-{name}")
            extra_roots.append(rr)
            U.write_manifest(rr / "manifest.csv", [r_kill])
            (rr / "oomk").write_text("0\n")
            R10[name] = rr
        env10 = U.base_env(JISA_TEST_OOM_KILL_FILE=R10["alone"] / "oomk")
        rc, out = U.run_worker(R10["alone"], R10["alone"] / "manifest.csv", env=env10, timeout=180)
        Ra, jk = R10["alone"] / "results", r_kill["job_id"]
        ooms = [U.load_json(f) for f in sorted((Ra / ".state" / jk).glob("oom_*.json"))]
        moved = sorted((Ra / ".state" / jk).glob("lock_obsoleto_*"))
        ck.check("g10.oom_killer_counted", rc == 0 and C.read_fail_count(Ra, jk) == 3 and len(ooms) == 3 and
                 all(o.get("source") == "oom_killer" and o.get("counted") is True and o.get("oom_kill_counter_delta") ==
                     {"test_file": 1} for o in ooms) and len(moved) == 3 and all((d / "accounted").exists() for d in moved)
                 and C.account_orphan_moves(Ra, jk) == 0,
                 f"rc={rc} fail_count={C.read_fail_count(Ra, jk)} oom={len(ooms)} locks movidos={len(moved)}")
        r_sleep = U.make_row(variant="_fake_sleep", priority=2, rep=11)
        Rc = R10["concurrent"] / "results"
        U.write_manifest(R10["concurrent"] / "m_sleep.csv", [r_sleep])
        env_s = U.base_env(JISA_TEST_SLEEP_S=40)
        ps = U.popen_job(R10["concurrent"], R10["concurrent"] / "m_sleep.csv", r_sleep["job_id"], env=env_s,
                         log_path=logs / "g10_sleep.log")
        try:
            got_lock = U.wait_for(lambda: (Rc / ".locks" / r_sleep["job_id"] / "info.json").exists(), timeout=60)
            env10c = U.base_env(JISA_TEST_OOM_KILL_FILE=R10["concurrent"] / "oomk")
            rc, out = U.run_worker(R10["concurrent"], R10["concurrent"] / "manifest.csv", env=env10c, timeout=180)
        finally:
            ps.wait(timeout=120)
        oc = [U.load_json(f) for f in sorted((Rc / ".state" / jk).glob("oom_*.json"))]
        mc = sorted((Rc / ".state" / jk).glob("lock_obsoleto_*"))
        ck.check("g10.oom_killer_concurrent", got_lock and rc == C.OOM and C.read_fail_count(Rc, jk) == 0 and len(oc) == 1 and
                 oc[0].get("counted") is False and oc[0].get("concurrent_jobs_same_host") == [r_sleep["job_id"]] and
                 len(mc) == 1 and (mc[0] / "accounted").exists() and C.account_orphan_moves(Rc, jk) == 0 and
                 C.read_fail_count(Rc, jk) == 0,
                 f"rc={rc} fail_count={C.read_fail_count(Rc, jk)} oom={oc[:1]}")

        import run_job as RJ
        s_aci = C.validate_row(row(block="T1-A-ACI", dataset="ACI-IOT-2023", variant="stat"))
        reg_aci = RJ.regions_for(s_aci, 554_116, {"fm": 50_000, "ad": 22_264}, {"extra": {"regions": {"train": [0, 72_264]}}})
        s_aw = C.validate_row(t1)
        reg_aw = RJ.regions_for(s_aw, 2_278_689, {"fm": 50_000, "ad": 150_000}, {"extra": {"regions": {"train": [0, 200_000]}}})
        ins_aci = reg_aci["detector_protocol_candidates"]["calib_in_sample_train_tail"]
        ins_aw = reg_aw["detector_protocol_candidates"]["calib_in_sample_train_tail"]
        ck.check("g11.s_ins_region", ins_aci["calib"] == [50_000 + 22_264 // 3, 72_264] == [57_421, 72_264] and
                 ins_aw["calib"] == [100_000, 200_000] and ins_aw["note"].startswith("misma referencia que el post-proceso enviado")
                 and "S-INS" in ins_aci["note"] and reg_aci["fit_set_exclusion"]["required"] is True,
                 f"ACI {ins_aci['calib']} (plan §2.9: [fm + floor(ad/3), T)); AW {ins_aw['calib']}")
        cm_aux = {"extra": {"regions": {"train": [0, 200_000]}, "duplicates_train_exec": {"n": 7678, "by_calib_fraction": {"10": {"n_dup": 4693}}}},
                  "aux": {"dup_train": {"path": "/x/aux_dup_train.npy", "sha256": "abc"}}}
        s_c17 = C.validate_row(row(block="T1-A-CIC2017", dataset="CIC2017", variant="stat"))
        fse = RJ.regions_for(s_c17, 2_830_743, {"fm": 50_000, "ad": 150_000}, cm_aux)["fit_set_exclusion"]
        ck.check("g11.fit_set_exclusion", fse["path"] == "/x/aux_dup_train.npy" and fse["sha256"] == "abc" and
                 fse["by_calib_fraction"]["10"]["n_dup"] == 4693 and reg_aw["fit_set_exclusion"]["mask"] is None and
                 "KNAD" in reg_aw["fit_set_exclusion"]["status"], f"{fse}")
        ex_aw = C.t4e_eval_exclusion("KNAD:Active_Wiretap")
        cal_aw, seg_aw = C.check_protocol_regions(reg_aw, "calib_first_10pct_post_train",
                                                  extra_eval_exclude=[ex_aw["excluded_with_guards"]])
        try:
            C.t4e_eval_exclusion("KNAD:No_Existe")
            miss = False
        except C.ProtocolRegionError:
            miss = True
        ck.check("g11.t4e_reference_c0", ex_aw["excluded_with_guards"] == [1_299_200, 1_344_843] and
                 ex_aw["pool"] == [1_300_000, 1_344_043] and cal_aw == [200_000, 407_868] and
                 seg_aw == [[407_868, 1_299_200], [1_344_843, 2_278_689]] and miss and
                 C.check_protocol_regions(reg_aw, "calib_first_10pct_post_train") == ([200_000, 407_868], [407_868, 2_278_689]),
                 f"referencia c = 0 de T4-E en AW: E = {seg_aw} (label_counts.json); sin exclusion la T1-A sigue igual")
        bad_t4 = json.loads(json.dumps(cfgt["regions"])) if rcs["t4e"] == 0 else None
        if bad_t4 is not None:
            for cd in bad_t4["detector_protocol_candidates"].values():
                if isinstance(cd, dict):
                    cd.pop("eval_exclude", None)
            ck.check("g11.t4e_exclusion_from_job_metadata", rejects_plain(bad_t4, "calib_first_10pct_post_train") and
                     C.check_protocol_regions(bad_t4, "calib_first_10pct_post_train", with_exclusions=True)[1] ==
                     [[4250, 5200], [7150, T4E_ROWS]], "t4e_training_contamination basta para excluir el pool")
        rT2a = row(block="T2-min", dataset="KNAD:Active_Wiretap", variant="mha", window=200, save_ckpt=1)
        rT2b = mutate(rT2a, window=2000)
        i_other = C.tbar_info([(rT2a, 100.0)], rT2b)
        i_same = C.tbar_info([(rT2a, 100.0), (mutate(rT2a, dataset="CIC2017"), 300.0)], rT2a)
        ck.check("g11.tbar_factors", i_other == (100.0, C.TBAR_LEVELS[2], False) and i_same == (100.0, C.TBAR_LEVELS[0], True),
                 f"L=2000 con solo L=200 medido -> {i_other}; misma L -> {i_same}")
    except Exception:
        import traceback
        traceback.print_exc()
        ck.check("g.exception", False, "excepcion en el test")
    rc = ck.finish()
    for rr in extra_roots:
        U.cleanup(rr, rc == 0)
    U.cleanup(root, rc == 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
