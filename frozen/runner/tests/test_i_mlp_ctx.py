#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(i) MLP with a context window ('mlp_ctx' and 'mlp_deep_ctx' of networks_ctx.py): architecture, windowing, registration, smoke runs."""
import argparse
import hashlib
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _util as U
from _util import C

RUNNER = U.RUNNER_DIR
AW_FM = [82, 15, 8, 10]
L_PROD, HR, LR = 800, 0.22, 0.002
JOB_ARGS = ["--max-rows", 4000,
            "--test-override", "fm_grace=1000,ad_grace=1500,execution_window=1000,sequence_length=16",
            "--data-root", str(C.DATA_ROOT)]


def hand_params(variant, sizes, L=L_PROD, hr=HR):
    """Replica independiente de networks_ctx.architecture_summary (aritmetica pura, para cotejarla)."""
    aes, total = [], 0
    for nv in sizes:
        nh = math.ceil(hr * nv)
        flat = L * nv
        if variant == "mlp_ctx":
            n = (flat * nh + nh) + (nh * nv + nv)
        else:
            h1 = max(1, flat // 4)
            n = (flat * h1 + h1) + (h1 * nh + nh) + (nh * h1 + h1) + (h1 * nv + nv)
        aes.append((nv, nh, flat, n))
        total += n
    return aes, total


def print_architecture_table(ck):
    for v in ("mlp_ctx", "mlp_deep_ctx"):
        aes, tot = hand_params(v, AW_FM)
        ck.info(f"{v} | Active Wiretap FM {AW_FM} | L = {L_PROD} | hidden_ratio = {HR}")
        for i, (nv, nh, flat, n) in enumerate(aes):
            extra = f" h1={max(1, flat // 4)}" if v == "mlp_deep_ctx" else ""
            ck.info(f"   ae{i}: n_visible={nv} n_hidden={nh} flatten={flat}{extra} -> {n:,} parametros")
        ck.info(f"   TOTAL {tot:,} parametros ({tot * 4 / 2 ** 30:.2f} GiB en fp32; x4 con Adam y gradientes: "
                f"{tot * 16 / 2 ** 30:.2f} GiB)")


def in_process(ck):
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(U.CODE_ROOT))
    sys.path.insert(0, str(RUNNER))
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    src_sha_before = _src_models_sha(U.CODE_ROOT)

    import instrument as I
    import networks_ctx as NC
    import src.models.oopsie as oopsie
    import src.models.kitnet as kitnet
    import src.models.networks.conv1d as conv1d
    import src.models.utils.sequential as seq

    before = dict(oopsie.NETWORK_TYPES)
    reg = I.register_context_variants()
    reg2 = I.register_context_variants()
    ok_keys = all(oopsie.NETWORK_TYPES[k] is v for k, v in before.items())
    ck.check("i6.original_keys", ok_keys and len(before) == 7,
             f"las 7 claves originales intactas ({sorted(before)})")
    ck.check("i6.new_keys", (oopsie.NETWORK_TYPES.get("mlp_ctx") is NC.MLPContextAutoencoder and
                             oopsie.NETWORK_TYPES.get("mlp_deep_ctx") is NC.DeepMLPContextAutoencoder and
                             kitnet.NETWORK_TYPES is oopsie.NETWORK_TYPES and reg2["registered"] == reg["registered"]),
             f"NETWORK_TYPES = {sorted(oopsie.NETWORK_TYPES)}; kitnet comparte el dict; registro idempotente")
    ck.check("i6.src_intacto", _src_models_sha(U.CODE_ROOT) == src_sha_before,
             f"sha256 de src/models/** sin cambios ({src_sha_before[:12]})")

    ck.check("i3.utilidades", (NC.create_windowed_data is seq.create_windowed_data is conv1d.create_windowed_data and
                               NC.create_windowed_data_ar is seq.create_windowed_data_ar is conv1d.create_windowed_data_ar),
             "create_windowed_data{,_ar} son los objetos de src/models/utils/sequential.py, no una copia")

    L, sizes = 16, [12, 9]
    for key, cls in (("mlp_ctx", NC.MLPContextAutoencoder), ("mlp_deep_ctx", NC.DeepMLPContextAutoencoder)):
        for nv in sizes:
            ae = cls(n_visible=nv, sequence_length=L, lr=LR, hidden_ratio=HR, haste=False, ar=True,
                     name=f"{key}[{nv}]")
            summ = NC.architecture_summary(key, [nv], sequence_length=L, hidden_ratio=HR)
            want = summ["autoencoders"][0]["params"]
            got = int(ae.model.count_params())
            ck.check(f"i1.{key}.n{nv}.params", got == want, f"count_params={got:,} == calculado a mano {want:,} "
                                                            f"(n_hidden={ae.n_hidden}, flatten={L * nv})")
            ck.check(f"i1.{key}.n{nv}.n_hidden", ae.n_hidden == math.ceil(HR * nv),
                     f"n_hidden={ae.n_hidden} = ceil({HR} x {nv}) (sobre n_visible, NO sobre L x n_visible = "
                     f"{math.ceil(HR * L * nv)})")
            ck.check(f"i2.{key}.n{nv}.shapes",
                     (tuple(ae.model.input_shape) == (None, L, nv) and tuple(ae.model.output_shape) == (None, nv) and
                      tuple(ae.encoder.input_shape) == (None, L, nv) and tuple(ae.decoder.output_shape) == (None, nv)),
                     f"model {ae.model.input_shape} -> {ae.model.output_shape}; encoder {ae.encoder.input_shape}; "
                     f"decoder {ae.decoder.output_shape}")
            oc = ae.model.optimizer.get_config()
            ck.check(f"i5.{key}.n{nv}.hparams",
                     (abs(float(ae.model.optimizer.learning_rate.numpy()) - LR) < 1e-9 and
                      oc["name"].lower() == "adam" and type(ae.model.loss).__name__ == "MeanSquaredError" and
                      ae.haste == 1 and ae.sequence_length == L and ae.input_dim == nv and ae.lr == LR and
                      all(w.dtype == tf.float32 for w in ae.model.weights)),
                     f"Adam lr={float(ae.model.optimizer.learning_rate.numpy())}, loss=MSE, haste={ae.haste}, "
                     f"L={ae.sequence_length}, fp32={all(w.dtype == tf.float32 for w in ae.model.weights)}")
            ck.check(f"i5.{key}.n{nv}.error_fn", cls.ERROR_FN == ("mse" if key == "mlp_ctx" else "rmse"),
                     f"ERROR_FN={cls.ERROR_FN} (mlp_ctx = MSE como multilayer.py:86-87; mlp_deep_ctx = RMSE como "
                     f"deep_mlp_ae.py:210)")
    ae_d = NC.MLPContextAutoencoder(n_visible=12, sequence_length=L, lr=LR, hidden_ratio=HR, corruption_level=0.3,
                                    haste=False, ar=False)
    drops = [l for l in ae_d.encoder.layers if type(l).__name__ == "Dropout"]
    ck.check("i5.dropout", len(drops) == 1 and abs(drops[0].rate - 0.3) < 1e-12 and ae_d.dropout == 0.3,
             f"Dropout(corruption_level) presente en el encoder: rate={drops[0].rate if drops else None}")
    ck.check("i5.constantes", (NC.BATCH_SIZE == 1 and NC.EXECUTION_BS == 512 and NC.HASTE_VALUE == 1),
             f"BATCH_SIZE={NC.BATCH_SIZE}, EXECUTION_BS={NC.EXECUTION_BS}, HASTE_VALUE={NC.HASTE_VALUE} "
             f"(src/models/__special__.py)")

    rng = np.random.default_rng(1234)
    nv, L4, n_tr, n_ex = 6, 8, 400, 200
    base = np.cumsum(rng.normal(size=(n_tr + n_ex, nv)), axis=0)
    base = (base - base.min(0)) / (base.max(0) - base.min(0) + 1e-9)
    out = {}
    for key, cls in (("mlp_ctx", NC.MLPContextAutoencoder), ("mlp_deep_ctx", NC.DeepMLPContextAutoencoder)):
        for ar in (True, False):
            tf.keras.utils.set_random_seed(1234)
            ae = cls(n_visible=nv, sequence_length=L4, lr=LR, hidden_ratio=HR, haste=False, ar=ar, name=f"{key}-{ar}")
            tr = np.asarray(ae.train(base[:n_tr]))
            ex = np.asarray(ae.execute(base[n_tr:]))
            out[(key, ar)] = (tr, ex)
            ck.check(f"i2.{key}.ar{int(ar)}.score_len", tr.shape == (n_tr,) and ex.shape == (n_ex,),
                     f"train -> {tr.shape} (= filas del tramo), execute -> {ex.shape} (= filas del bloque)")
            ck.check(f"i4.{key}.ar{int(ar)}.finite", bool(np.isfinite(tr).all() and np.isfinite(ex).all()),
                     f"scores finitos; rango execute [{ex.min():.4g}, {ex.max():.4g}]")
    for key in ("mlp_ctx", "mlp_deep_ctx"):
        a, t = out[(key, True)][1], out[(key, False)][1]
        d = float(np.max(np.abs(a - t)))
        span = float(max(a.max(), t.max()) - min(a.min(), t.min()))
        pct = 100.0 * d / span if span else 0.0
        rk = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(t)))[0, 1])
        ck.check(f"i4.{key}.ar_vs_tsr_no_identico", d > 1e-9,
                 f"AR y TSR sobre el MISMO stream y la misma semilla NO son bit a bit iguales: "
                 f"max|AR-TSR| = {d:.6g} (el modelo lee 'ar')")
        distinguible = pct > 1.0 and rk < 0.999
        veredicto = ("paradigmas DISTINGUIBLES" if distinguible else
                     "paradigmas PRACTICAMENTE EQUIVALENTES: a este L no sostienen dos config_id; ver MR-13")
        ck.info(f"i4.{key}.ar_vs_tsr_magnitud (L={L4}): max|AR-TSR| = {pct:.2f} % del rango, "
                f"correlacion de rangos = {rk:.6f} -> {veredicto}")
    nvw, Lw, nw = 4, 5, 12
    xw = np.arange(nw * nvw, dtype=float).reshape(nw, nvw)
    pad = np.zeros((Lw, nvw))
    probs = []
    for ar, fn in ((True, NC.create_windowed_data_ar), (False, NC.create_windowed_data)):
        ds, _lw = fn(xw, Lw, pad.copy(), batch_size=1)
        full = np.concatenate([pad, xw], axis=0)
        got = [(w.numpy()[0], t.numpy()[0]) for w, t in ds]
        if len(got) != nw:
            probs.append(f"ar={ar}: {len(got)} ventanas != {nw} filas")
            continue
        for j, (w, t) in enumerate(got):
            exp_in = full[j:j + Lw] if ar else full[j + 1:j + 1 + Lw]
            if not np.array_equal(w, exp_in) or not np.array_equal(t, xw[j]):
                probs.append(f"ar={ar} fila {j}: ventana o objetivo inesperados")
                break
    ck.check("i4.semantica_ventana", not probs,
             "AR ve las L filas anteriores y predice la fila i; TSR ve las L-1 anteriores mas la i y la reconstruye "
             f"(sequential.py:14-69; {probs[:2]})")

    tf.keras.utils.set_random_seed(7)
    for key, cls in (("mlp_ctx", NC.MLPContextAutoencoder), ("mlp_deep_ctx", NC.DeepMLPContextAutoencoder)):
        ae = cls(n_visible=nvw, sequence_length=Lw, lr=LR, hidden_ratio=HR, haste=False, ar=False)
        wa = rng.random((1, Lw, nvw))
        wb = wa.copy()
        wb[0, :-1] = rng.random((Lw - 1, nvw))
        pa, pb = ae.model.predict(wa, verbose=0), ae.model.predict(wb, verbose=0)
        ck.check(f"i4.{key}.lee_la_ventana", float(np.max(np.abs(pa - pb))) > 1e-6,
                 f"cambiar solo las L-1 filas anteriores cambia la salida: max|d| = {float(np.max(np.abs(pa - pb))):.6g}")

    nvc, Lc, n_trc, P = 20, 8, 400, 20
    bc = np.cumsum(rng.normal(size=(600, nvc)), axis=0)
    bc = (bc - bc.min(0)) / (bc.max(0) - bc.min(0) + 1e-9)
    bounded = []
    for key, cls in (("mlp_ctx", NC.MLPContextAutoencoder), ("mlp_deep_ctx", NC.DeepMLPContextAutoencoder)):
        for ar in (True, False):
            tf.keras.utils.set_random_seed(1234)
            ae = cls(n_visible=nvc, sequence_length=Lc, lr=LR, hidden_ratio=HR, haste=False, ar=ar)
            ae.train(bc[:n_trc])
            w0 = ae.window.copy()
            blk = bc[n_trc:n_trc + 60].copy()
            e1 = ae.execute(blk)
            ae.window = w0.copy()
            alt = blk.copy()
            alt[:P] = blk[:P][rng.permutation(P)]
            e2 = ae.execute(alt)
            safe = P + Lc + 1
            if np.any(np.abs(e1[safe:] - e2[safe:]) > 0):
                bounded.append(f"{key} ar={ar}")
    ck.check("i4.contexto_acotado", not bounded,
             f"permutar las {P} primeras filas del bloque no cambia NINGUNA fila a partir de la {P + Lc + 1}: la "
             f"influencia del contexto esta acotada por L ({bounded})")

    class _Spec:
        variant, ar = "mlp_ctx", True
    cfg = {"k": 2, "hidden_ratio": HR, "lr": LR, "L": 16}

    class _KN:
        pass
    kn = _KN()
    kn.feature_map = [list(range(12)), list(range(9))]
    kn.autoencoders = [NC.MLPContextAutoencoder(n_visible=len(f), sequence_length=16, lr=LR, hidden_ratio=HR,
                                                haste=False, ar=True) for f in kn.feature_map]
    from src.models.networks import StatisticalAnomaly
    kn.output_autoencoder = StatisticalAnomaly(2)
    try:
        I.check_aes(kn, cfg, _Spec())
        ck.check("i6.check_aes", True, "instrument.check_aes acepta mlp_ctx como variante con ventana")
    except Exception as e:
        ck.check("i6.check_aes", False, f"{type(e).__name__}: {e}")


def _src_models_sha(code_root):
    h = hashlib.sha256()
    for p in sorted(Path(code_root, "src", "models").rglob("*.py")):
        h.update(p.relative_to(code_root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def manifest_tests(ck):
    import make_manifest as MM
    cur_p = RUNNER / "manifest.csv"
    cand_p = RUNNER / "manifest_v3_candidate.csv"
    rows = MM.generate_v3(MM.ACTIVE_SECOND)
    text = MM.to_csv(rows)
    cur_text = cur_p.read_text()
    errs = MM.check_rows_preserved(cur_text, rows, text, "vigente")
    n_cur = len(C.read_manifest(cur_p))
    ck.check("i7.463_identicas", not errs and n_cur == 463,
             f"{n_cur} filas vigentes conservadas byte a byte en el candidato ({errs[:2]})")
    ck.check("i7.candidato_en_disco", cand_p.exists() and cand_p.read_text() == text,
             f"{cand_p.name} coincide con lo generado")
    ck.check("i7.manifest_intacto", cur_p.read_text() == cur_text and len(C.read_manifest(cur_p)) == 463,
             "manifest.csv no se ha tocado (lo esta leyendo un worker)")
    t7 = [r for r in rows if r["block"].startswith("T7-")]
    w = [r for r in t7 if r["block"] == "T7-MLPW"]
    dp = [r for r in t7 if r["block"] == "T7-MLPDEEP"]
    ck.check("i7.recuentos", len(rows) == 485 and len(w) == 18 and len(dp) == 4,
             f"{len(rows)} filas = 463 + 18 (T7-MLPW) + 4 (T7-MLPDEEP)")
    pw = sorted(int(r["priority"]) for r in w)
    pd_ = sorted(int(r["priority"]) for r in dp)
    t4e = max(int(r["priority"]) for r in rows if r["block"] == "T4-E")
    t1c = min(int(r["priority"]) for r in rows if r["block"].startswith("T1-C"))
    ck.check("i7.prioridades", pw == list(range(3500, 3518)) and pd_ == list(range(3600, 3604)) and
             t4e < pw[0] and pd_[-1] < t1c,
             f"T7-MLPW {pw[0]}-{pw[-1]}, T7-MLPDEEP {pd_[0]}-{pd_[-1]}; T4-E max {t4e} < 3500 y 3603 < T1-C min {t1c}")
    ck.check("i7.campos", all(r["window"] == "800" and r["k_clusters"] == "4" and r["grouping"] == "dbscan" and
                              r["partition_seed"] == "NA" and r["contamination"] == "0" and r["seed"] == "A" and
                              r["pipeline"] == "corrected" and r["warmup_frac"] == "NA" for r in t7),
             "L=800, k=4, dbscan, sin contaminacion, semilla A, pipeline corrected en las 22 filas")
    ck.check("i7.datasets", sorted({r["dataset"] for r in w}) == sorted(C.ALL9_DATASETS) and
             sorted({r["dataset"] for r in dp}) == ["CIC2017", "KNAD:Active_Wiretap"] and
             sorted({(r["variant"], r["paradigm"]) for r in w}) == [("mlp_ctx", "AR"), ("mlp_ctx", "TSR")],
             "T7-MLPW: 9 datasets x {AR, TSR}; T7-MLPDEEP: AW y CIC2017 x {AR, TSR}")
    ck.check("i7.ckpt", all(r["save_ckpt"] == "1" for r in w) and all(r["save_ckpt"] == "0" for r in dp),
             "save_ckpt = 1 en mlp_ctx y 0 en mlp_deep_ctx (4,3 GiB de pesos por job)")
    bad = [("mlp_ctx fuera de su bloque", U.make_row(block="T1-A-KNAD", variant="mlp_ctx", ar=True,
                                                     pipeline="corrected", seed="A")),
           ("T7-MLPW con mlp", U.make_row(block="T7-MLPW", variant="mlp", ar=True, pipeline="corrected", seed="A")),
           ("T7-MLPW con semilla B", U.make_row(block="T7-MLPW", variant="mlp_ctx", ar=True, pipeline="corrected",
                                                seed="B")),
           ("T7-MLPDEEP con dataset ajeno", U.make_row(block="T7-MLPDEEP", variant="mlp_deep_ctx", ar=True,
                                                       pipeline="corrected", seed="A", dataset="KNAD:Fuzzing")),
           ("T7 con L != 800", U.make_row(block="T7-MLPW", variant="mlp_ctx", ar=True, pipeline="corrected",
                                          seed="A", window=200)),
           ("T7 con pipeline orig", U.make_row(block="T7-MLPW", variant="mlp_ctx", ar=True, pipeline="orig",
                                               seed="A"))]
    rejected = []
    for name, r in bad:
        try:
            C.validate_row(r, test_mode=False)
            rejected.append(f"ACEPTADA {name}")
        except C.InvalidJob:
            pass
    ck.check("i7.validaciones_negativas", not rejected, f"6 filas mal formadas rechazadas ({rejected})")


def postproc_tests(ck):
    import metrics_rev as M
    import aggregate as A
    row = {"job_id": "x", "block": "T7-MLPW", "dataset": "KNAD:Active_Wiretap", "variant": "mlp_ctx", "ar": "True",
           "paradigm": "AR", "seed": "A", "pipeline": "corrected", "pipeline_version": "v1", "window": "800",
           "k_clusters": "4", "grouping": "dbscan", "partition_seed": "NA", "contamination": "0", "rep": "1"}
    d = M.job_descriptor({"job_id": "x", "manifest_row": row})
    ck.check("i8.config_id", d["config_id"] == "C12" and d["family"] == "T7" and d["exploratory_arm"] is True,
             f"config_id={d['config_id']} family={d['family']} exploratory_arm={d['exploratory_arm']}")
    dconf = M.job_descriptor({"job_id": "c", "manifest_row": dict(row, block="T1-A-KNAD", variant="mlp")})
    ck.check("i8.exploratory_arm_nombre",
             "exploratory" not in d and dconf["exploratory_arm"] is False and dconf["exploratory_arm_rule"] is None,
             "el descriptor NO define la clave ambigua 'exploratory'; un job confirmatorio lleva "
             f"exploratory_arm=False (config_id={dconf['config_id']})")
    d2 = M.job_descriptor({"job_id": "y", "manifest_row": dict(row, ar="False", paradigm="TSR")})
    d3 = M.job_descriptor({"job_id": "z", "manifest_row": dict(row, block="T7-MLPDEEP", variant="mlp_deep_ctx")})
    ck.check("i8.config_ids_4", (d2["config_id"], d3["config_id"]) == ("C13", "C14") and
             sorted(M.CTX_CONFIG_IDS.values()) == ["C12", "C13", "C14", "C15"],
             "C12/C13 = mlp_ctx AR/TSR, C14/C15 = mlp_deep_ctx AR/TSR (AR y TSR NO se promedian como en C9)")
    ck.check("i8.mlp_replica", d["mlp_replica"] is None,
             "mlp_replica = None: la regla AG-1 de promediar replicas (C9) no se aplica a las variantes con ventana")
    ck.check("i8.aggregate_fuera",
             ("C12" not in A.CONFIGS and "C12" not in A.ROWCFG_TO_CONFIG.values() and
              "mlp_ctx-AR" not in A.ROWCFGS and set(A.CTX_CONFIGS) == {"C12", "C13", "C14", "C15"}),
             f"CONFIGS = {A.CONFIGS[0]}..{A.CONFIGS[-1]} (sin C12-C15); ROWCFGS sin mlp_ctx: quedan fuera de cells, "
             f"S*_d, SD_pool, value_sets y de todo lo que lee stats.py")
    import stats as S
    ck.check("i8.stats_fuera", S.CONFIGS == [f"C{i}" for i in range(1, 12)] and
             S.EXCLUDED_CONFIGS == ["C12", "C13", "C14", "C15"] and "C12" not in S.CTX8 + S.SIMPLE,
             "stats.py: CONFIGS = C1..C11, EXCLUDED_CONFIGS = C12..C15 (guarda en main)")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        J = Path(td)
        for f in ("scores.npy", "labels.npy", "config.json", "DONE"):
            (J / f).write_bytes(b"")
        try:
            M.applicable({"manifest_row": row}, J)
            ok_t7 = True
        except M.NotApplicable as e:
            ok_t7 = f"{e}"
        bad = []
        for b in ("X2", "S-w1"):
            try:
                M.applicable({"manifest_row": dict(row, block=b)}, J)
                bad.append(b)
            except M.NotApplicable:
                pass
    ck.check("i8.applicable", ok_t7 is True and not bad, f"T7 aplicable ({ok_t7}); X2 y S siguen fuera ({bad})")


def adversarial_fixes_tests(ck):
    """Each check corresponds to one fix applied to the context-window variants."""
    import json
    import tempfile
    import aggregate as AG
    import metrics_rev as M
    import common as CC

    with tempfile.TemporaryDirectory() as td:
        R = Path(td)
        M.record_chronology(R, {"created": "2026-09-17T08:00:00", "job_id": "T1-A-KNAD__a", "dataset_id": "D3",
                                "family": "T1", "exploratory_arm": False})
        M.record_chronology(R, {"created": "2026-09-01T08:00:00", "job_id": "T7-MLPW__b", "dataset_id": "D3",
                                "family": "T7", "exploratory_arm": True})
        first, log = M.read_chronology(R)
        fams = sorted({e.get("family") for e in log})
        ck.check("i10.cronologia_family", fams == ["T1", "T7"] and all("exploratory_arm" in e for e in log),
                 f"MR-12 escribe family y exploratory_arm en cada entrada del log ({fams})")
    ck.check("i10.cronologia_filtro", "T7" in CC.EXPLORATORY_FAMILIES and "T1" not in CC.EXPLORATORY_FAMILIES,
             f"EXPLORATORY_FAMILIES = {CC.EXPLORATORY_FAMILIES}: aggregate descarta T7 de chronology_R7")

    class _A:
        pass
    A = _A()
    A.ctx_jobs = [{"job_id": "j", "block": "T7-MLPW", "config_id": "C12", "config_name": "x", "dataset": "d",
                   "dataset_id": "D1", "variant": "mlp_ctx", "ar": True, "seed": "A", "state": "DONE",
                   "metrics_state": "ok", "reference_block": "T1-A-KNAD", "tag": "[EXPLORATORIO]",
                   "flat": {"auroc_FUS": 0.80}}]
    A.cells = {"D1": {"C9": {"mean": {"auroc_FUS": 0.70}, "auroc_FUS_by_seed": {"A": 0.72, "B": 0.68}}}}
    A.datasets = {"D1": {"sd_pool": {"threshold": 0.03}}}
    A.mlp_noise = {"D1": {"A": {"delta_AR_minus_TSR": -0.02}}}
    A.idx = None
    S = AG.ctx_summary(A)
    it = S["jobs"][0]
    ck.check("i10.delta_semilla_A",
             abs(it["delta_auroc_FUS_vs_C9_seedA"] - 0.08) < 1e-12 and
             abs(it["delta_auroc_FUS_vs_C9_mean"] - 0.10) < 1e-12 and
             it["reference_auroc_FUS_seedA"] == 0.72 and "delta_auroc_FUS_vs_C9" not in it,
             "la referencia principal es C9 de la semilla A (0,72), no la media sobre S*_d (0,70); ya no existe el "
             "campo ambiguo delta_auroc_FUS_vs_C9")
    ck.check("i10.suelo_de_ruido",
             it["noise_threshold"] == 0.03 and abs(it["abs_delta_MLP_seedA"] - 0.02) < 1e-12 and
             it["exceeds_noise_threshold"] is True and it["reference_rowcfg"] == "mlp-AR" and it["delta_rule"],
             "cada delta viaja con noise_threshold, |Delta_MLP| de la semilla A y la replica de referencia")
    ck.check("i10.inventario_y_declaraciones",
             S["exploratory_inventory"]["n_arms"] == 1 and S["exploratory_inventory"]["adjusted"] is False and
             "etiquetas" in S["disclosure_required"] and "No pre-registered rule" in S["headline_rule"],
             "exploratory_ctx lleva inventario, la declaracion de etiquetas y el aviso de que no hay regla de cifra "
             "titular")

    st = (Path(__file__).resolve().parents[1] / "status.py").read_text()
    ck.check("i10.eta_separada",
             "pending_hours_confirmatory" in st and "pending_hours_exploratory" in st and
             "EXPLORATORY_FAMILIES" in st,
             "status.py separa las horas de T7 de las que usa la escalera de §13.3")

    wk = (Path(__file__).resolve().parents[1] / "worker.py").read_text()
    ck.check("i10.exclusividad",
             CC.EXCLUSIVE_VARIANTS.get("mlp_deep_ctx") == 1 and "EXCLUSIVE_VARIANTS" in wk and
             "live_variants" in wk,
             f"EXCLUSIVE_VARIANTS = {CC.EXCLUSIVE_VARIANTS}; worker.py no reclama una segunda fila de la misma "
             "variante exclusiva viva en el host")

    sys.path.insert(0, str(U.CODE_ROOT))
    import instrument as I
    import src.models.networks as networks
    reg = I.register_context_variants()
    ck.check("i10.solo_network_types",
             not hasattr(networks, "MLPContextAutoencoder") and
             not hasattr(networks, "DeepMLPContextAutoencoder") and
             all(r.get("sha256") for r in reg["registered"].values()),
             "no se hace setattr sobre src.models.networks (nadie lo lee) y el registro guarda el sha256 del modulo")

    ck.check("i10.version_metrics_rev", M.METRICS_REV_VERSION == "1.2.0" and AG.AGGREGATE_VERSION == "1.2.0",
             f"metrics_rev {M.METRICS_REV_VERSION} y aggregate {AG.AGGREGATE_VERSION}: las dos versiones suben "
             "juntas, si no AG-10 vera versiones mezcladas")

    gate = Path(__file__).resolve().parent / "check_metrics_rev_regression.py"
    ck.check("i10.puerta_regresion", gate.exists() and "IGNORED_TOP" in gate.read_text(),
             f"existe {gate.name}: regresion numerica sobre los metrics.json REALES antes del tag")

    gi = (Path(__file__).resolve().parents[3] / ".gitignore").read_text().splitlines()
    ck.check("i10.gitignore_results", any(l.strip() == "results/" for l in gi),
             "results/ (55 GB, ~1300 ficheros) esta entero en .gitignore: un git add -A en la promocion no puede "
             "versionarlo")


def job_smoke(ck):
    root = U.make_root("i")
    R = root / "results"
    try:
        fam = [("T7-MLPW", "mlp_ctx", True, 1), ("T7-MLPW", "mlp_ctx", False, 1),
               ("T7-MLPDEEP", "mlp_deep_ctx", True, 0)]
        rows = [U.make_row(block=b, variant=v, ar=ar, seed="A", save_ckpt=ck_, priority=i, dataset="KNAD:OS_Scan")
                for i, (b, v, ar, ck_) in enumerate(fam)]
        man = root / "manifest.csv"
        U.write_manifest(man, rows)
        env = U.base_env(PYTHONHASHSEED=1234, OMP_NUM_THREADS="4")
        t0 = time.time()
        procs = {f"{v}-{ar}": U.popen_job(root, man, r["job_id"], JOB_ARGS, env=env,
                                          log_path=root / "logs" / f"{v}-{ar}.log")
                 for (b, v, ar, _c), r in zip(fam, rows)}
        rcs = {k: p.wait(timeout=1800) for k, p in procs.items()}
        ck.info(f"rc={rcs} en {time.time() - t0:.0f} s")
        scores = {}
        for (b, v, ar, ck_), r in zip(fam, rows):
            key = f"{v}-{ar}"
            J = R / r["job_id"]
            if rcs[key] != 0:
                ck.check(f"i9.{key}", False, f"rc={rcs[key]}\n{C.tail_text(root / 'logs' / f'{key}.log', 3000)}")
                continue
            arr = U.load_json(J / "arrays.json")
            cfg = U.load_json(J / "config.json")
            s, e = arr["scores.npy"], arr.get("ae_errors.npy", {})
            ck.check(f"i9.{key}.arrays", (s["shape"] == [4000] and s["dtype"] == "float64" and
                                          e.get("shape") == [4000, 4] and s.get("n_nan") == 0),
                     f"scores {s['shape']} {s['dtype']} n_nan={s.get('n_nan')}; ae_errors {e.get('shape')}")
            hp = cfg["hyperparameters_effective"]["autoencoders"][0]
            exp_w = "create_windowed_data_ar" if ar else "create_windowed_data"
            ck.check(f"i9.{key}.hparams",
                     (hp.get("create_windowed") == exp_w and hp.get("sequence_length") == 16 and
                      hp.get("error_fn") == ("mse" if v == "mlp_ctx" else "rmse") and abs(hp.get("lr") - LR) < 1e-12 and
                      hp.get("haste") == 1 and hp.get("class") == ("MLPContextAutoencoder" if v == "mlp_ctx"
                                                                   else "DeepMLPContextAutoencoder")),
                     f"clase={hp.get('class')} ventana={hp.get('create_windowed')} L={hp.get('sequence_length')} "
                     f"lr={hp.get('lr')} error_fn={hp.get('error_fn')} params={hp.get('count_params')}")
            reg = (cfg.get("context_variants") or {}).get("registered") or {}
            ck.check(f"i9.{key}.registro", v in reg and reg[v]["module"].endswith("networks_ctx.py") and
                     len((cfg["context_variants"].get("keys_after") or [])) == 9,
                     f"config.json registra las variantes: {sorted(reg)}; NETWORK_TYPES con "
                     f"{len(cfg['context_variants'].get('keys_after') or [])} claves")
            import csv as _csv
            tl = list(_csv.DictReader((J / "train_log.csv").open()))
            ck.check(f"i9.{key}.train_log",
                     len(tl) == 4 and all(r_["train_steps"] == "1500" and r_["epochs"] == "1" and
                                          r_["loss_epoch_1"] not in ("NA", "") for r_ in tl),
                     f"4 AEs, pasos={[r_['train_steps'] for r_ in tl]}, epocas={[r_['epochs'] for r_ in tl]}")
            if ck_:
                need = [J / "ckpt" / f"ae{i}" / sub / "saved_model.pb" for i in range(4)
                        for sub in ("model", "encoder", "decoder")]
                ck.check(f"i9.{key}.ckpt", all(p.exists() for p in need) and "error" not in (cfg.get("ckpt") or {}),
                         f"{sum(p.exists() for p in need)}/{len(need)} ficheros de checkpoint")
                sc = U.load_json(J / "ckpt" / "networks_ctx_provenance.json")
                ck.check(f"i9.{key}.ckpt_procedencia",
                         bool(sc) and sc.get("pickled_module") == "networks_ctx" and
                         sc.get("module_sha256") == C.sha256_file(RUNNER / "networks_ctx.py") and
                         sc.get("variant") == v and "PYTHONPATH" in (sc.get("how_to_load") or ""),
                         f"ckpt/networks_ctx_provenance.json: modulo={sc.get('pickled_module')} "
                         f"clase={sc.get('class')} sha256 coincide con networks_ctx.py")
            else:
                ck.check(f"i9.{key}.sin_ckpt", not (J / "ckpt").exists(), "sin checkpoint (save_ckpt = 0)")
            import numpy as np
            scores[key] = np.load(J / "scores.npy")
        if "mlp_ctx-True" in scores and "mlp_ctx-False" in scores:
            import numpy as np
            a, t = scores["mlp_ctx-True"], scores["mlp_ctx-False"]
            d = float(np.max(np.abs(a - t)))
            span = float(max(a.max(), t.max()) - min(a.min(), t.min()))
            rk = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(t)))[0, 1])
            ck.check("i9.ar_vs_tsr_job", d > 1e-9,
                     f"AR y TSR del MISMO job (mismo stream, misma semilla) dan scores distintos: max|d| = {d:.6g}")
            ck.info(f"i9.ar_vs_tsr_magnitud (L reducida del smoke): max|AR-TSR| = "
                    f"{(100.0 * d / span if span else 0.0):.2f} % del rango, correlacion de rangos = {rk:.6f}; "
                    f"a L = 800 sobre datos reales se midio 16,5 % del rango con rangos 0,99973 (MR-13)")
    except Exception:
        import traceback
        traceback.print_exc()
        ck.check("i9.exception", False, "excepcion en el smoke")
    finally:
        U.cleanup(root, ck.ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-jobs", action="store_true", help="Omite i9 (los tres jobs por run_job).")
    a = ap.parse_args()
    ck = U.Checker("test_i_mlp_ctx")
    sys.path.insert(0, str(RUNNER))
    print_architecture_table(ck)
    try:
        manifest_tests(ck)
        postproc_tests(ck)
        in_process(ck)
        adversarial_fixes_tests(ck)
    except Exception:
        import traceback
        traceback.print_exc()
        ck.check("i.exception", False, "excepcion fuera del smoke")
    if not a.no_jobs:
        job_smoke(ck)
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
