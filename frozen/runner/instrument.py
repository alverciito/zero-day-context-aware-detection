#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""instrument.py - observation hooks of the job runner (seeds, DBSCAN recorder, loop observer, captures) that leave the numerics unchanged."""
import csv
import math
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

EXPECTED_CONFIG = {"train_period": 150, "clustering_period": 50, "sequence_length": 800, "hidden_ratio": 0.22,
                   "autoencoder_size": 4, "clustering": "dbscan", "output_ae_type": "stat", "execution_window": 400}
EXPECTED_SPECIAL = {"HASTE_VALUE": 1, "BATCH_SIZE": 1, "EXECUTION_BS": 512}
EXPECTED_LR = 0.002
ELM_GRACE = 10_000
DBSCAN_MAX_ITERS = 100
HOTFIX_MIN_SIZE = 8
T4E_GUARD_ROWS = 800
T4E_CALIB_FRAC = C.PRIMARY_CALIB_PCT / 100.0
T4E_POOL_FRAC = 0.20
LABEL_COUNTS_REL = C.LABEL_COUNTS_REL
EXECUTION_BS_OOM_STEPS = (128, 32)
AE_CLASSES = {"mha": "MHAAutoencoder", "original": "DenoisingAutoencoder", "lstm": "LSTMAutoencoder",
              "original_eps": "DenoisingAutoencoderEps16",
              "mlp_ctx_fix_e5": "MLPContextFixedE5Autoencoder",
              "mlp_fix_e5": "MLPPerRecordFixedE5Autoencoder",
              "mlp": "MLPAutoencoder", "conv1d": "Conv1DAutoencoder", "conv2d": "Conv2DAutoencoder",
              "stat": "StatisticalAnomaly",
              "mlp_ctx": "MLPContextAutoencoder", "mlp_deep_ctx": "DeepMLPContextAutoencoder",
              "mlp_ctx_fix": "MLPContextFixedAutoencoder", "mlp_fix": "MLPPerRecordFixedAutoencoder"}
ORIGINAL_NETWORK_TYPES = {"mha": "MHAAutoencoder", "original": "DenoisingAutoencoder", "lstm": "LSTMAutoencoder",
                          "mlp": "MLPAutoencoder", "conv1d": "Conv1DAutoencoder", "conv2d": "Conv2DAutoencoder",
                          "stat": "StatisticalAnomaly"}
TEST_OVERRIDE_KEYS = {"fm_grace", "ad_grace", "execution_window", "sequence_length", "train_rows", "probe_train_s",
                      "probe_predict_s", "probe_warmup_steps", "probe_warmup_batches", "probe_loop_rows",
                      "probe_barrier_timeout_s"}
S17_NOTES = {
    "mlp_fit_shuffle": "MLPAutoencoder.train llama a model.fit(x, x, epochs=1, batch_size=1, verbose=1) con arrays "
                       "numpy y shuffle=True por defecto de Keras (multilayer.py:84): consume RNG de TF",
    "predict_internal_batch": "las redes con ventana llaman a model.predict(x_batch, verbose=0) con batch_size por "
                              "defecto (32) dentro de cada lote de EXECUTION_BS=512 ventanas (conv1d.py:89,104; "
                              "conv2d.py:90,105; lstm.py:87,102; mha.py:91,106)",
    "circular_padding": "en train() la ventana de relleno inicial es de ceros para fit; para el error de entrenamiento "
                        "se reutiliza self.window = ultimas L filas del propio tramo (relleno circular; conv1d.py:82,86;"
                        " sequential.py:14-69); de ahi pasa a ejecucion y entre bloques",
    "windowed_dtypes": "y_batch sale de tf.data en float64 y predicted en float32; el RMSE se calcula en float64 "
                       "(conv1d.py:90,105); mlp devuelve float32 (multilayer.py:98-102); original/stat float64",
    "haste": "haste=False en __build_daes__ -> epochs = HASTE_VALUE = 1 (kitnet.py:297-305; __special__.py:16)",
    "score_row_semantics": "score[i] corresponde a la fila i del stream; AR usa las L filas previas y predice la i; "
                           "TSR usa las L-1 previas mas la i (sequential.py:14-69)",
}


def register_context_variants(spec=None):
    """Register the context-window variants of networks_ctx.py in NETWORK_TYPES at run time, only adding keys; returns what was registered."""
    import src.models.networks as networks
    import src.models.oopsie as oopsie
    import networks_ctx
    for key, want in ORIGINAL_NETWORK_TYPES.items():
        got = oopsie.NETWORK_TYPES.get(key)
        if got is None or got.__name__ != want:
            raise C.FrozenConfigError(f"NETWORK_TYPES[{key!r}] = {got} != {want} (oopsie.py:10-18): la tabla "
                                      "original ha cambiado; el registro de variantes solo puede anadir claves")
    reg = {}
    for key, cls in networks_ctx.CONTEXT_NETWORK_TYPES.items():
        cur = oopsie.NETWORK_TYPES.get(key)
        if cur is not None and cur is not cls:
            raise C.FrozenConfigError(f"NETWORK_TYPES[{key!r}] ya estaba ocupada por {cur}")
        if hasattr(networks, cls.__name__):
            raise C.FrozenConfigError(f"src.models.networks ya tiene un atributo {cls.__name__!r}: el registro de "
                                      "variantes solo puede anadir claves a NETWORK_TYPES")
        oopsie.NETWORK_TYPES[key] = cls
        reg[key] = {"class": cls.__name__, "module": networks_ctx.__file__, "error_fn": cls.ERROR_FN,
                    "sha256": C.sha256_file(networks_ctx.__file__)}
    import src.models.kitnet as kitnet
    if kitnet.NETWORK_TYPES is not oopsie.NETWORK_TYPES:
        raise C.FrozenConfigError("kitnet.NETWORK_TYPES no es el mismo objeto que oopsie.NETWORK_TYPES: el registro "
                                  "en caliente no llegaria a __build_daes__")
    return {"registered": reg, "keys_after": sorted(oopsie.NETWORK_TYPES),
            "networks_ctx_sha256": C.sha256_file(networks_ctx.__file__),
            "rule": "src/ intacto; las clases viven en revision/experiments/networks_ctx.py y solo se usan en los "
                    "bloques T7-* ([EXPLORATORIO])",
            "variant_of_this_job": getattr(spec, "variant", None)}


def apply_seeds(label):
    if label is None:
        return []
    s = C.SEEDS[label]
    import random
    import tensorflow as tf
    random.seed(s)
    np.random.seed(s)
    tf.keras.utils.set_random_seed(s)
    return [f"random.seed({s})", f"numpy.random.seed({s})", f"tf.keras.utils.set_random_seed({s})"]


def frozen_config(spec, overrides, test_mode):
    import experiments.frames.launch_all as la
    import src.models.__special__ as sp
    from src.models import KitNET
    import inspect
    checks = {}
    if dict(la.CONFIG) != EXPECTED_CONFIG:
        raise C.FrozenConfigError(f"launch_all.CONFIG {la.CONFIG} != congelado {EXPECTED_CONFIG}.")
    checks["launch_all_CONFIG"] = dict(la.CONFIG)
    special = {k: getattr(sp, k) for k in EXPECTED_SPECIAL}
    if special != EXPECTED_SPECIAL:
        raise C.FrozenConfigError(f"__special__ {special} != {EXPECTED_SPECIAL}")
    checks["__special__"] = special
    lr_default = inspect.signature(KitNET.__init__).parameters["learning_rate"].default
    if lr_default != EXPECTED_LR:
        raise C.FrozenConfigError(f"KitNET learning_rate por defecto {lr_default} != {EXPECTED_LR}")
    checks["kitnet_default_learning_rate"] = lr_default
    cfg = {"fm": 1000 * la.CONFIG["clustering_period"], "ad": 1000 * la.CONFIG["train_period"],
           "ew": 1000 * la.CONFIG["execution_window"], "L": la.CONFIG["sequence_length"],
           "hidden_ratio": la.CONFIG["hidden_ratio"], "k": la.CONFIG["autoencoder_size"],
           "clustering": la.CONFIG["clustering"], "output_ae_type": la.CONFIG["output_ae_type"], "lr": EXPECTED_LR}
    abl = spec.ablation()
    factor = (abl or {}).get("factor")
    if spec.mode == "probe":
        cfg["L"] = spec.window
    elif spec.window != cfg["L"] and not spec.is_fake:
        if factor != "window":
            raise C.InvalidJob(f"window={spec.window} != CONFIG sequence_length={cfg['L']} fuera de T2")
        cfg["L"] = spec.window
    if spec.k != cfg["k"]:
        if factor != "k" or spec.mode == "probe":
            raise C.InvalidJob(f"k_clusters={spec.k} != CONFIG autoencoder_size={cfg['k']} fuera de T3")
        cfg["k"] = spec.k
    if spec.grouping != "dbscan" and factor != "grouping":
        raise C.InvalidJob(f"grouping={spec.grouping} fuera de T3-min-b/T3-ext-a")
    if spec.contamination_pct and factor != "contamination":
        raise C.InvalidJob(f"contamination={spec.contamination_pct} fuera de T4-E")
    if spec.pipeline == "corrected":
        t_rows = int(C.corrected_train_rows(spec))
        if t_rows - cfg["fm"] != cfg["ad"]:
            if spec.dataset not in C.CORRECTED_AE_DEVIATIONS:
                raise C.FrozenConfigError(f"{spec.dataset}: T={t_rows} no es fm + ad congelados ({cfg['fm']} + {cfg['ad']}) "
                                          f"y no hay desviacion declarada")
            cfg["ad"] = t_rows - cfg["fm"]
            checks["ae_rows_dev006"] = {"dataset": spec.dataset, "train_rows": t_rows, "ad": cfg["ad"],
                                        "note": C.CORRECTED_AE_DEVIATIONS[spec.dataset]}
    cfg.update(ablation=abl, grouping=spec.grouping, partition_seed=spec.partition_seed,
               contamination_pct=spec.contamination_pct)
    ov = dict(overrides or {})
    bad = set(ov) - TEST_OVERRIDE_KEYS
    if bad:
        raise C.InvalidJob(f"overrides no permitidos: {sorted(bad)}")
    if ov and not test_mode:
        raise C.InvalidJob("overrides solo con --test-mode")
    for k_ov, k_cfg in (("fm_grace", "fm"), ("ad_grace", "ad"), ("execution_window", "ew"), ("sequence_length", "L")):
        if k_ov in ov:
            cfg[k_cfg] = int(ov[k_ov])
    if spec.variant in C.NUMPY_VARIANTS and cfg["ad"] < ELM_GRACE:
        raise C.InvalidJob(f"ad_grace={cfg['ad']} < {ELM_GRACE}: el ELM devuelve 0.0 escalar y rompe tracking "
                           "(original_kitsune.py:126-127; kitnet.py:204)")
    cfg["overrides"] = ov
    cfg["checks"] = checks
    return cfg


def kitnet_kwargs(x_shape_last, cfg, spec):
    """Mismos argumentos y orden que launch_all.py:54-65."""
    return dict(feature_map=None, ad_grace_period=cfg["ad"], execution_window=cfg["ew"], fm_grace_period=cfg["fm"],
                sequence_length=cfg["L"], hidden_ratio=cfg["hidden_ratio"], autoencoder_size=cfg["k"],
                clustering=cfg["clustering"], ar=spec.ar, ae_type=spec.variant, output_ae_type=cfg["output_ae_type"])


def check_kitnet_config(kn, cfg, spec):
    exp_exec = {"execution_window": cfg["ew"], "fm_grace_period": cfg["fm"], "ad_grace_period": cfg["ad"]}
    exp_ae = {"type": spec.variant, "output_type": cfg["output_ae_type"], "hidden_ratio": cfg["hidden_ratio"],
              "learning_rate": cfg["lr"], "sequence_length": cfg["L"], "autoencoder_size": cfg["k"], "is_ar": spec.ar}
    if kn.execution_config != exp_exec:
        raise C.FrozenConfigError(f"kn.execution_config {kn.execution_config} != {exp_exec}")
    if kn.autoencoder_config != exp_ae:
        raise C.FrozenConfigError(f"kn.autoencoder_config {kn.autoencoder_config} != {exp_ae}")
    if kn.clustering_model_name != cfg["clustering"]:
        raise C.FrozenConfigError(f"clustering {kn.clustering_model_name} != {cfg['clustering']}")
    return {"execution_config": dict(kn.execution_config), "autoencoder_config": dict(kn.autoencoder_config),
            "clustering": kn.clustering_model_name, "n_features": kn.n_features}


def check_aes(kn, cfg, spec):
    """Tras construir los AEs (fila fm-1): clase, n_hidden, lr, L, ventana, haste. FrozenConfigError si no casa."""
    probs = []
    if len(kn.autoencoders) != cfg["k"] or len(kn.feature_map) != cfg["k"]:
        probs.append(f"{len(kn.autoencoders)} AEs / {len(kn.feature_map)} grupos != {cfg['k']}")
    exp_cls = AE_CLASSES[spec.variant]
    for i, (ae, fmap) in enumerate(zip(kn.autoencoders, kn.feature_map)):
        nvis = len(fmap)
        nh = int(np.ceil(cfg["hidden_ratio"] * nvis))
        cls = type(ae).__name__
        if cls != exp_cls:
            probs.append(f"ae{i}: clase {cls} != {exp_cls}")
            continue
        if spec.variant in C.KERAS_VARIANTS:
            exp_haste = int(getattr(type(ae), "EPOCHS", None) or 1)
            if ae.input_dim != nvis or ae.n_hidden != nh or ae.lr != cfg["lr"] or ae.haste != exp_haste:
                probs.append(f"ae{i}: input_dim={ae.input_dim}/{nvis} n_hidden={ae.n_hidden}/{nh} lr={ae.lr} "
                             f"haste={ae.haste}/{exp_haste}")
            if spec.variant in C.FIXED_VARIANTS and hasattr(ae.model, "opt_enc"):
                try:
                    lr_enc = float(ae.model.opt_enc.learning_rate.numpy())
                    lr_dec = float(ae.model.opt_dec.learning_rate.numpy())
                    want_enc = cfg["lr"] * nvis / float(ae.fan_in_encoder)
                    if abs(lr_dec - cfg["lr"]) > 1e-6 * max(1.0, cfg["lr"]):
                        probs.append(f"ae{i}: lr del decodificador {lr_dec} != {cfg['lr']}")
                    if abs(lr_enc - want_enc) > 1e-5 * max(want_enc, 1e-12):
                        probs.append(f"ae{i}: lr del codificador {lr_enc} != {want_enc} (regla Pi)")
                except Exception as e:
                    probs.append(f"ae{i}: lr de los optimizadores no legible: {e}")
            else:
                try:
                    olr = float(ae.model.optimizer.learning_rate.numpy())
                    if abs(olr - cfg["lr"]) > 1e-9:
                        probs.append(f"ae{i}: optimizer lr {olr} != {cfg['lr']}")
                except Exception as e:
                    probs.append(f"ae{i}: lr del optimizador no legible: {e}")
            if spec.variant in C.WINDOWED_VARIANTS:
                exp_w = "create_windowed_data_ar" if spec.ar else "create_windowed_data"
                if ae.sequence_length != cfg["L"] or ae.create_windowed.__name__ != exp_w:
                    probs.append(f"ae{i}: L={ae.sequence_length}/{cfg['L']} ventana={ae.create_windowed.__name__}/{exp_w}")
        elif spec.variant in C.NUMPY_VARIANTS:
            if spec.variant == "original_eps":
                eps = getattr(ae, "EPS_TRAIN", None)
                if eps != 1e-16:
                    probs.append(f"ae{i}: original_eps con EPS_TRAIN={eps!r} != 1e-16")
            if (ae.n_visible != nvis or ae.n_hidden != nh or ae.lr != cfg["lr"] or ae.grace_period != ELM_GRACE
                    or ae.corruption_level != 0.0):
                probs.append(f"ae{i}: ELM n_visible={ae.n_visible} n_hidden={ae.n_hidden} lr={ae.lr} "
                             f"grace={ae.grace_period} corruption={ae.corruption_level}")
        elif spec.variant == "stat":
            if np.asarray(ae.mean).shape != (nvis,):
                probs.append(f"ae{i}: stat mean shape {np.asarray(ae.mean).shape} != ({nvis},)")
    out = kn.output_autoencoder
    if type(out).__name__ != "StatisticalAnomaly" or np.asarray(out.mean).shape != (cfg["k"],):
        probs.append(f"capa de salida {type(out).__name__} con mean {np.asarray(out.mean).shape}")
    if probs:
        raise C.FrozenConfigError("configuracion efectiva distinta de la congelada: " + "; ".join(probs))
    return True


def hyperparameters_effective(kn, cfg, spec):
    import src.models.__special__ as sp
    res = {"kitnet": {"execution_config": dict(kn.execution_config), "autoencoder_config": dict(kn.autoencoder_config),
                      "clustering_model_name": kn.clustering_model_name, "n_features": kn.n_features},
           "__special__": {k: getattr(sp, k) for k in EXPECTED_SPECIAL}, "autoencoders": [], "notes": S17_NOTES}
    for i, (ae, fmap) in enumerate(zip(kn.autoencoders, kn.feature_map)):
        d = {"index": i, "class": type(ae).__name__, "n_features_in_group": len(fmap)}
        for a in ("input_dim", "n_visible", "n_hidden", "lr", "sequence_length", "haste", "dropout", "seed",
                  "grace_period", "corruption_level", "hidden_ratio", "ar", "flat_dim", "h1"):
            if hasattr(ae, a):
                v = getattr(ae, a)
                d[a] = v if isinstance(v, (int, float, str, type(None))) else str(v)
        if getattr(type(ae), "ERROR_FN", None) is not None:
            d["error_fn"] = type(ae).ERROR_FN
        if hasattr(ae, "create_windowed"):
            d["create_windowed"] = ae.create_windowed.__name__
        if getattr(ae, "model", None) is not None:
            try:
                oc = ae.model.optimizer.get_config()
                d["optimizer"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                  for k, v in oc.items()}
            except Exception as e:
                d["optimizer"] = f"no legible: {e}"
            try:
                d["count_params"] = int(ae.model.count_params())
                d["layers"] = [{"name": l.name, "class": type(l).__name__, "params": int(l.count_params())}
                               for l in ae.model.layers]
            except Exception as e:
                d["count_params"] = f"no legible: {e}"
            try:
                d["loss"] = type(ae.model.loss).__name__
            except Exception:
                pass
        if hasattr(ae, "W"):
            d["W_shape"] = list(np.asarray(ae.W).shape)
        res["autoencoders"].append(d)
    out = kn.output_autoencoder
    res["output_layer"] = {"class": type(out).__name__, "n_visible": int(np.asarray(out.mean).shape[0])}
    return res


def hotfix_outcome(labels, k):
    """Prediccion exacta del hot-fix de dbscan.py:50-63 sobre las etiquetas del ultimo ajuste. La dinamica solo depende
    de los tamanos de los grupos (argmax/argmin del primer indice y 8 pop/append), asi que se simula sobre ellos:
    'ok' (termina), 'no_termina' (estado repetido: bucle infinito) o 'crash_pop_empty' (pop de una lista vacia)."""
    lp1 = np.asarray(labels) + 1
    lens = [int((lp1 == i).sum()) for i in range(k)]
    before, seen, moves = list(lens), set(), 0
    while min(lens) < HOTFIX_MIN_SIZE:
        st = tuple(lens)
        imax, imin = int(np.argmax(lens)), int(np.argmin(lens))
        if imax == imin or st in seen:
            status = "crash_pop_empty" if lens[imax] == 0 else "no_termina"
            return {"status": status, "moves": moves, "sizes_before": before, "sizes_at_stop": list(lens), "k": int(k)}
        seen.add(st)
        if lens[imax] < HOTFIX_MIN_SIZE:
            return {"status": "crash_pop_empty", "moves": moves, "sizes_before": before, "sizes_at_stop": list(lens),
                    "k": int(k)}
        lens[imax] -= HOTFIX_MIN_SIZE
        lens[imin] += HOTFIX_MIN_SIZE
        moves += 1
    return {"status": "ok", "moves": moves, "sizes_before": before, "sizes_after": list(lens), "k": int(k)}


class DBSCANRecorder:
    def __init__(self, target_k=None):
        self.trajectory = []
        self.last_labels = None
        self.target_k = target_k
        self.hotfix_prediction = None
        self._mod = None
        self._orig = None

    def __enter__(self):
        import src.models.cluster.dbscan as mod
        from sklearn.cluster import DBSCAN as RealDBSCAN
        if mod.skDBSCAN is not RealDBSCAN:
            raise C.FrozenConfigError(f"dbscan.skDBSCAN es {mod.skDBSCAN}, no sklearn.cluster.DBSCAN")
        rec = self

        class _RecordingDBSCAN:
            def __init__(self, eps, min_samples):
                self._real = RealDBSCAN(eps=eps, min_samples=min_samples)
                self._eps = eps
                self._min_samples = min_samples

            def fit(self, X):
                est = self._real.fit(X)
                labels = np.asarray(est.labels_).copy()
                rec.last_labels = labels
                n_labels = int(len(set(labels.tolist())))
                rec.trajectory.append({"iter": len(rec.trajectory) + 1, "eps": float(self._eps),
                                       "min_samples": int(self._min_samples), "n_points": int(np.asarray(X).shape[0]),
                                       "n_labels_incl_noise": n_labels,
                                       "n_noise": int((labels == -1).sum())})
                if rec.target_k is not None and (n_labels == rec.target_k or len(rec.trajectory) == DBSCAN_MAX_ITERS):
                    rec.hotfix_prediction = hotfix_outcome(labels, rec.target_k)
                    if rec.hotfix_prediction["status"] != "ok":
                        exc = C.NonExecutableCell(
                            f"hot-fix de dbscan.py:58-63 con k={rec.target_k}: {rec.hotfix_prediction['status']} "
                            f"(tamanos {rec.hotfix_prediction['sizes_before']} tras {len(rec.trajectory)} ajustes, "
                            f"{n_labels} etiquetas con ruido); el codigo original no puede completar la celda")
                        exc.detail = {"hotfix_prediction": rec.hotfix_prediction, "trajectory": list(rec.trajectory),
                                      "labels_plus1": (labels + 1).tolist()}
                        raise exc
                return est

        self._mod, self._orig = mod, mod.skDBSCAN
        mod.skDBSCAN = _RecordingDBSCAN
        return self

    def __exit__(self, *exc):
        if self._mod is not None:
            self._mod.skDBSCAN = self._orig
        return False


def fm_stats(rec, kn, k, fm_rows, grouping=None):
    """Replica de dbscan.py:31-63 sobre la trayectoria registrada; consistencia con kn.feature_map.
    Con agrupacion aleatoria (grouping = RandomGroupingClusterer, T3) la replica se compara con el mapa DBSCAN que
    devolvio el original y kn.feature_map con la particion aleatoria inyectada (mismos tamanos)."""
    traj = rec.trajectory
    res = {"n_features": int(kn.n_features), "rows_used": [0, int(fm_rows)],
           "normalization": "min-max por feature sobre el buffer (dbscan.py:25-26)",
           "points": "features (fit(x.T), dbscan.py:36)", "eps0": 1.4, "min_samples": 8, "k": int(k),
           "trajectory": traj, "iterations": len(traj), "noise_counts_as_cluster": True}
    eps, momentum, eps_ok = 1.4, 1.0, True
    for t in traj:
        if abs(t["eps"] - eps) > 1e-12:
            eps_ok = False
        diff = k - t["n_labels_incl_noise"]
        eps += 0.1 * diff * momentum
        momentum *= 0.99
        if eps < 0:
            eps = 0.05
    res.update(eps_trajectory_replicated=eps_ok, eps_last_fit=traj[-1]["eps"] if traj else None,
               eps_after_update=eps, converged=bool(traj and traj[-1]["n_labels_incl_noise"] == k),
               warning_iter100=len(traj) == 100)
    consistency = {"feature_map_equal": None}
    if rec.hotfix_prediction is not None:
        res["hotfix_prediction"] = rec.hotfix_prediction
    if rec.last_labels is not None and hotfix_outcome(rec.last_labels, k)["status"] != "ok":
        res["consistency"] = {"feature_map_equal": None, "note": "hot-fix no terminaria: sin replica"}
        return res
    if rec.last_labels is not None:
        lp1 = rec.last_labels + 1
        fmap = [np.where(lp1 == i)[0].tolist() for i in range(k)]
        sizes_before = [len(c) for c in fmap]
        dropped = np.where(lp1 >= k)[0].tolist()
        moves, lens = [], [len(c) for c in fmap]
        while min(lens) < 8:
            imax, imin = int(np.argmax(lens)), int(np.argmin(lens))
            moved = []
            for _ in range(8):
                v = fmap[imax].pop(-1)
                fmap[imin].append(v)
                moved.append(int(v))
            moves.append({"from": imax, "to": imin, "features": moved})
            lens = [len(c) for c in fmap]
        kn_map = [[int(v) for v in c] for c in (kn.feature_map or [])]
        if grouping is None:
            consistency = {"feature_map_equal": fmap == kn_map}
        else:
            dbm = grouping.dbscan_map
            consistency = {"feature_map_equal": fmap == dbm, "random_map_is_kn_feature_map": kn_map == grouping.feature_map,
                           "sizes_equal_dbscan": [len(c) for c in kn_map] == [len(c) for c in (dbm or [])]}
            res.update(feature_map_dbscan=dbm, random_grouping=grouping.info)
        res.update(labels_plus1=lp1.tolist(), sizes_before_hotfix=sizes_before, dropped_features=dropped,
                   hotfix_moves=moves, hotfix_applied=bool(moves), sizes_final=[len(c) for c in kn_map],
                   feature_map=kn_map, features_without_ae=sorted(set(range(kn.n_features)) - {v for c in kn_map for v in c}))
    res["consistency"] = consistency
    return res


def random_partition(n_features, sizes, partition_seed):
    """Permutacion numpy.random.default_rng(partition_seed).permutation(n_features) repartida en grupos consecutivos
    con `sizes` (orden de los grupos del mapa DBSCAN; dentro de cada grupo, el orden de la permutacion). Si el mapa
    DBSCAN deja features sin AE, quedan sin AE las ultimas n - sum(sizes) de la permutacion (mismos tamanos).
    default_rng es un Generator propio: no toca el RNG global de numpy ni el de TF."""
    perm = np.random.default_rng(int(partition_seed)).permutation(int(n_features))
    if sum(sizes) > n_features:
        raise C.FrozenConfigError(f"tamanos {sizes} suman mas que n_features={n_features}")
    groups, s = [], 0
    for size in sizes:
        groups.append([int(v) for v in perm[s:s + size]])
        s += size
    return groups, perm


class RandomGroupingClusterer:
    """Stand-in for kn.clustering_model during the feature-map phase: a random partition with the group sizes of the DBSCAN map."""

    def __init__(self, real, partition_seed, n_features):
        self.real = real
        self.partition_seed = int(partition_seed)
        self.n_features = int(n_features)
        self.dbscan_map = None
        self.feature_map = None
        self.info = None
        self.calls = 0

    def update(self, x):
        self.real.update(x)

    def cluster(self, n_clusters):
        self.calls += 1
        dbm = [[int(v) for v in g] for g in self.real.cluster(n_clusters)]
        sizes = [len(g) for g in dbm]
        groups, perm = random_partition(self.n_features, sizes, self.partition_seed)
        self.dbscan_map, self.feature_map = dbm, groups
        self.info = {"method": "numpy.random.default_rng(partition_seed).permutation(n_features) en grupos consecutivos "
                               "con los tamanos finales del mapa DBSCAN de este mismo FM (analysis_plan.md §7.3)",
                     "partition_seed": self.partition_seed, "n_features": self.n_features, "sizes": sizes,
                     "permutation": [int(v) for v in perm], "feature_map_random": groups,
                     "features_without_ae": sorted(set(range(self.n_features)) - {v for g in groups for v in g}),
                     "injection": "kn.clustering_model.cluster en la fila fm-1 (kitnet.py:149-154); no KitNET(feature_map=...)"}
        return groups


def t1a_feature_map_reference(results_root, spec):
    """fm_stats.json de los jobs T1-A DONE del mismo dataset y pipeline (k = 4, DBSCAN): el plan fija los tamanos de la
    agrupacion aleatoria con 'el mapa DBSCAN de T1-A de ese dataset'. DBSCAN es determinista con los mismos datos, asi
    que deben coincidir con el de este FM. Devuelve lista de (job_id, feature_map)."""
    R = Path(results_root)
    pat = f"T1-A-*__{C.ds_slug(spec.dataset)}__*__{spec.pipeline}.{spec.pipeline_version}__L800__k4__s*__r*"
    out = []
    for d in sorted(R.glob(pat)):
        f = d / "fm_stats.json"
        if (d / "DONE").exists() and f.exists():
            fs = C.read_json(f, {}) or {}
            if fs.get("feature_map") is not None:
                out.append((d.name, fs["feature_map"]))
    return out


def t4e_inject(spec, x, y, cfg, code_root, test_mode):
    """Sustituye EN MEMORIA (nunca en la cache) un bloque contiguo de B_c = round(c/100 * ad) filas centrado en
    [fm, T) por las primeras B_c filas de ataque (features y etiquetas) del pool de §8.3: las primeras
    ceil(0,20 |F_cal|) filas de ataque de E = [T + floor(0,10 (N - T)), N) en orden del stream. El FM [0, fm) no cambia.
    El pool con 800 filas de guarda a cada lado se declara para excluirlo de E aguas abajo (en estas filas y en su
    referencia c = 0). En produccion |F_cal|, el pool y las guardas se toman de plan_support/label_counts.json (plan
    congelado) y se exige que coincidan con los recalculados aqui; si no, PipelineError (HALT).
    Devuelve (x modificada en su sitio, y nueva, info, donor_idx)."""
    import hashlib
    fm, ad = int(cfg["fm"]), int(cfg["ad"])
    T, n = fm + ad, int(len(y))
    c = int(spec.contamination_pct)
    y = np.asarray(y)
    if n <= T:
        raise C.PipelineError(f"T4-E: el stream ({n} filas) no tiene filas tras el entrenamiento ({T})")
    if int(np.asarray(y[:T]).sum()) != 0:
        raise C.PipelineError(f"T4-E: el entrenamiento [0, {T}) ya contiene ataques antes de inyectar")
    n_calib = int(math.floor(T4E_CALIB_FRAC * (n - T)))
    es, ee = T + n_calib, n
    key = ("knad_" + spec.attack.lower()) if spec.kind == "knad" else spec.kind
    ref, lc_sha = None, None
    if test_mode:
        fit_rows, fit_src = n_calib, "test-mode: |F_cal| = filas de calibracion (sin deduplicar)"
    else:
        lc_path = Path(code_root) / LABEL_COUNTS_REL
        lc = C.read_json(lc_path)
        if not lc or key not in lc:
            raise C.PipelineError(f"T4-E: {lc_path} sin la entrada {key!r} (insumo del plan congelado)")
        lc_sha = C.sha256_file(lc_path)
        ent = lc[key]
        if int(ent["calib"]["rows"]) != n_calib:
            raise C.PipelineError(f"T4-E: calibracion de {key} con {n_calib} filas != {ent['calib']['rows']} de label_counts")
        fit_rows, fit_src = int(ent["calib"]["fit_rows_dedup"]), f"{LABEL_COUNTS_REL} [{key}].calib.fit_rows_dedup"
        ref = ent.get("t4p_donor_pool")
        if not isinstance(ref, dict) or any(k not in ref for k in ("pool", "excluded_with_guards", "attack_rows_needed")):
            raise C.PipelineError(f"T4-E: {lc_path} [{key}] sin t4p_donor_pool completo (pool, excluded_with_guards, "
                                  f"attack_rows_needed): insumo del plan congelado (§8.3)")
    need = int(math.ceil(T4E_POOL_FRAC * fit_rows))
    att = es + np.flatnonzero(np.asarray(y[es:ee]) == 1)
    if len(att) < need:
        raise C.PipelineError(f"T4-E: E tiene {len(att)} filas de ataque < {need} del pool de §8.3")
    a, b = int(att[0]), int(att[need - 1]) + 1
    xs, xe = max(es, a - T4E_GUARD_ROWS), min(ee, b + T4E_GUARD_ROWS)
    if ref is not None and (list(ref.get("pool", [])) != [a, b] or list(ref.get("excluded_with_guards", [])) != [xs, xe]
                            or int(ref.get("attack_rows_needed", -1)) != need):
        raise C.PipelineError(f"T4-E: pool recalculado {[a, b]} guardas {[xs, xe]} need {need} != label_counts {ref}")
    B = int(round(c * ad / 100.0))
    if not 1 <= B <= need:
        raise C.PipelineError(f"T4-E: B_c={B} fuera de [1, {need}] (pool de §8.3)")
    start = fm + (ad - B) // 2
    if start < fm or start + B > T:
        raise C.PipelineError(f"T4-E: bloque [{start}, {start + B}) fuera de [fm, T) = [{fm}, {T})")
    donor = att[:B].astype(np.int64)

    def h(arr):
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

    y_clean_sha, x_block_before = h(y), h(x[start:start + B])
    x[start:start + B] = x[donor]
    y2 = y.copy()
    y2[start:start + B] = y[donor]
    info = {"plan": "analysis_plan.md §8.5 T4-E", "c_pct": c, "ae_rows": [fm, T], "B_c": B,
            "B_c_rule": "round(c/100 * ad)", "injected_rows": [int(start), int(start + B)],
            "center_rule": "start = fm + floor((ad - B_c) / 2)", "fm_unchanged": [0, fm],
            "calib": [T, es], "eval": [es, ee], "fit_rows": fit_rows, "fit_rows_source": fit_src,
            "pool_attack_rows_needed": need, "donor_pool": [a, b], "guard_rows": T4E_GUARD_ROWS,
            "eval_exclude_with_guards": [int(xs), int(xe)],
            "donor_rows": {"n": int(B), "first": int(donor[0]), "last": int(donor[-1]), "sha256_int64": h(donor),
                           "file": "t4e_donor_rows.npy"},
            "label_counts_sha256": lc_sha, "label_counts_crosscheck": ref is not None,
            "labels_clean_sha256": y_clean_sha, "labels_injected_sha256": h(y2),
            "x_block_sha256_before": x_block_before, "x_block_sha256_after": h(x[start:start + B]),
            "n_attack_rows_in_train_after": int(y2[:T].sum()),
            "post_process_contract": "excluir eval_exclude_with_guards de E en estas filas y en su referencia c = 0 "
                                     "(celdas de T1-A semilla A); features AfterImage de las filas movidas fuera de su "
                                     "contexto temporal (aproximacion declarada, §8.5)"}
    return x, y2, info, donor


def execution_bs_override(spec, results_root, steps=EXECUTION_BS_OOM_STEPS):
    """Si la fila es T2 (factor window) con MHA y L = 2000, y la celda ya tiene OOM CONTADOS (.state/X/oom_*.json con
    counted = true: sin concurrencia; en CPU, SIGKILL confirmado por el OOM killer, worker.py), fija
    src.models.networks.mha.EXECUTION_BS = steps[min(n, len(steps)) - 1] en este proceso (mha.py:7 lo importa por
    nombre; se lee en cada llamada a create_windowed, mha.py:88,101) y devuelve el registro para config.json. El
    entrenamiento (batch 1) no cambia. Cualquier otra fila o sin OOM contados -> None (comportamiento congelado)."""
    abl = spec.ablation() or {}
    if abl.get("factor") != "window" or spec.variant != "mha" or spec.window != 2000 or spec.mode != "full":
        return None
    st = Path(results_root) / ".state" / spec.job_id
    ooms = []
    for f in sorted(st.glob("oom_*.json")) if st.exists() else []:
        d = C.read_json(f, {}) or {}
        if d.get("counted"):
            ooms.append({"file": f.name, "phase": d.get("phase"), "ts": d.get("ts"), "source": d.get("source", "run_job")})
    if not ooms:
        return None
    import src.models.__special__ as sp
    import src.models.networks.mha as mha_mod
    value = int(steps[min(len(ooms), len(steps)) - 1])
    frozen = int(sp.EXECUTION_BS)
    if value % 32 or value > frozen:
        raise C.FrozenConfigError(f"EXECUTION_BS reducido {value} no es multiplo de 32 <= {frozen}")
    mha_mod.EXECUTION_BS = value
    return {"module": "src.models.networks.mha", "value": value, "frozen": frozen, "previous_counted_ooms": ooms,
            "plan": "analysis_plan.md §7.2 (OOM de MHA con L = 2000: se reduce el lote de inferencia, desviacion declarada)",
            "reason": f"{len(ooms)} OOM contados previos de esta celda",
            "equivalence": "model.predict trocea cada lote en lotes de 32 (S17_NOTES.predict_internal_batch); con un "
                           "multiplo de 32 cada llamada ve los mismos lotes de 32 filas y el RMSE es por fila "
                           "(mha.py:92-94,105-107): scores identicos (test g9 en CPU, byte a byte)"}


def gpu_peak_and_reset(device):
    if not str(device).upper().startswith("/GPU"):
        return None
    try:
        import tensorflow as tf
        peak = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
        tf.config.experimental.reset_memory_stats("GPU:0")
        return peak
    except Exception:
        return None


def rss_peak_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class LoopObserver:
    def __init__(self, n, cfg, hb, device, on_fm_end=None, on_train_end=None):
        self.n, self.fm, self.ad, self.ew = n, cfg["fm"], cfg["ad"], cfg["ew"]
        self.hb, self.device = hb, device
        self.on_fm_end, self.on_train_end = on_fm_end, on_train_end
        ev = set()
        if self.fm - 1 < n:
            ev.add(self.fm - 1)
        if self.fm + self.ad - 1 < n:
            ev.add(self.fm + self.ad - 1)
        b = self.fm + self.ad + self.ew - 1
        while b < n - 1:
            ev.add(b)
            b += self.ew
        ev.add(n - 1)
        self.events = sorted(e for e in ev if e >= 0)
        self._k = 0
        self.next_event = self.events[0] if self.events else -1
        self.t = {}
        self.exec_blocks = []
        self.gpu_peak = {}
        self.anomalies = []
        self._t_last = None
        self._block_start = self.fm + self.ad
        self.row = 0

    def start(self):
        self._t0 = self._t_last = time.perf_counter()
        gpu_peak_and_reset(self.device)
        if self.hb:
            self.hb.set_phase("fm", self.n)

    def after(self, i, kn):
        self.row = i
        if self.hb is not None:
            self.hb.row = i
        if i == self.next_event:
            self._event(i, kn)

    def _event(self, i, kn):
        now = time.perf_counter()
        seg = now - self._t_last
        if i == self.fm - 1:
            self.t["fm_s"] = seg
            self.gpu_peak["fm"] = gpu_peak_and_reset(self.device)
            if self.on_fm_end:
                self.on_fm_end(kn)
            if self.hb:
                self.hb.set_phase("train")
        elif i == self.fm + self.ad - 1:
            self.t["train_s"] = seg
            self.gpu_peak["train"] = gpu_peak_and_reset(self.device)
            if self.on_train_end:
                self.on_train_end(kn)
            if self.hb:
                self.hb.set_phase("exec")
        elif i >= self.fm + self.ad:
            if kn.execution_stack_count != 0:
                self.anomalies.append(f"fila {i}: execution_stack_count={kn.execution_stack_count} en frontera esperada")
            self.exec_blocks.append({"start_row": self._block_start, "n_rows": i + 1 - self._block_start,
                                     "seconds": seg, "gpu_peak_bytes": gpu_peak_and_reset(self.device)})
            self._block_start = i + 1
        elif i == self.n - 1:
            self.t["partial_last_phase_s"] = seg
        self._t_last = time.perf_counter()
        self._k += 1
        self.next_event = self.events[self._k] if self._k < len(self.events) else -1

    def timing(self):
        ex = sum(b["seconds"] for b in self.exec_blocks)
        rows = sum(b["n_rows"] for b in self.exec_blocks)
        return {"fm_s": self.t.get("fm_s"), "train_s": self.t.get("train_s"), "exec_s": ex if self.exec_blocks else None,
                "exec_rows": rows, "exec_rows_per_s": (rows / ex) if ex else None, "exec_blocks": self.exec_blocks,
                "loop_total_s": time.perf_counter() - self._t0, "gpu_peak_bytes": self.gpu_peak}


def save_scores(kn, tmp):
    """Esencial. Misma expresion y mismo np.save que show_stats (kitnet.py:323-324,345)."""
    mse = np.array(kn.current_error)
    mse = np.log10(mse + 1)
    np.save(str(Path(tmp) / "scores") + ".npy", mse)
    return mse


def save_labels(y, tmp):
    """Save the aligned labels of the stream in the job directory (int8), after scores.npy; they are not read inside the job."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"etiquetas con forma {y.shape}")
    y8 = y.astype(np.int8)
    if not np.array_equal(y8.astype(y.dtype), y):
        raise ValueError(f"etiquetas {y.dtype} no representables en int8 sin perdida")
    np.save(Path(tmp) / "labels.npy", y8)
    info = array_info(Path(tmp) / "labels.npy")
    import hashlib
    info.update(source_dtype=str(y.dtype), source_sha256_tobytes=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
                values=sorted(int(v) for v in np.unique(y8)))
    return info


def array_info(path):
    a = np.load(path, mmap_mode="r")
    info = {"path": Path(path).name, "shape": list(a.shape), "dtype": str(a.dtype), "sha256": C.sha256_file(path)}
    if a.dtype.kind == "f":
        info["n_nan"] = int(np.isnan(a).sum())
        info["n_inf"] = int(np.isinf(a).sum())
    return info


def capture_after_loop(kn, rec, obs, tmp, spec, cfg, fm_stats_early, anomalies):
    """Capturas no esenciales (M5 ii): cada una en su try; los fallos van a anomalies."""
    tmp = Path(tmp)
    arrays = {"scores.npy": array_info(tmp / "scores.npy")}
    outputs = ["scores.npy"]
    fm, ad = cfg["fm"], cfg["ad"]
    try:
        E = np.stack([np.asarray(t) for t in kn.tracking["exec_loss"]], axis=1)
        np.save(tmp / "ae_errors.npy", E)
        del E
        E2 = np.load(tmp / "ae_errors.npy")
        info = array_info(tmp / "ae_errors.npy")
        segs = []
        if kn.trained_instances >= fm + ad:
            segs.append((fm, fm + ad, "train"))
        segs += [(b["start_row"], b["start_row"] + b["n_rows"], "exec") for b in obs.exec_blocks]
        cur = kn.current_error
        results, all_ok = [], True
        for a, b, kind in segs:
            out = np.asarray(kn.output_autoencoder.execute(E2[a:b]))
            ref = np.asarray(cur[a:b])
            same_shape = out.shape == ref.shape
            eq = bool(same_shape and np.array_equal(out, ref, equal_nan=out.dtype.kind == "f" and ref.dtype.kind == "f"))
            md = None
            if same_shape and not eq:
                with np.errstate(invalid="ignore"):
                    d = np.abs(out.astype(np.float64) - ref.astype(np.float64))
                fin = d[np.isfinite(d)]
                md = float(fin.max()) if fin.size else None
            results.append({"rows": [a, b], "kind": kind, "array_equal": eq, "equal_nan": True,
                            "max_abs_diff_finite": md,
                            "n_nan_current_error": int(np.isnan(ref).sum()) if ref.dtype.kind == "f" else 0,
                            "n_nan_recomputed": int(np.isnan(out).sum()) if out.dtype.kind == "f" else 0})
            all_ok &= eq
        info["selfcheck_output_layer_from_saved_file"] = {"all_equal": all_ok, "segments": results,
                                                          "zeros_rows_0_fm": bool(np.all(E2[:fm] == 0))}
        if not all_ok:
            anomalies.append("autochequeo ae_errors: la capa stat sobre ae_errors.npy no reproduce current_error")
        arrays["ae_errors.npy"] = info
        outputs.append("ae_errors.npy")
        del E2
    except Exception:
        anomalies.append("ae_errors.npy: " + traceback.format_exc()[-1500:])
    try:
        d = {}
        for i, ae in enumerate(kn.autoencoders):
            for a in ("norm_min", "norm_max", "mean", "std", "n"):
                if hasattr(ae, a) and getattr(ae, a) is not None:
                    d[f"ae{i}_{a}"] = np.asarray(getattr(ae, a), dtype=np.float64)
        out = kn.output_autoencoder
        for a in ("mean", "std", "norm_min", "norm_max"):
            d[f"output_{a}"] = np.asarray(getattr(out, a), dtype=np.float64)
        np.savez(tmp / "model_stats.npz", **d)
        outputs.append("model_stats.npz")
    except Exception:
        anomalies.append("model_stats.npz: " + traceback.format_exc()[-1500:])
    try:
        fs = fm_stats_early if fm_stats_early is not None else fm_stats(rec, kn, cfg["k"], fm)
        if fs.get("consistency", {}).get("feature_map_equal") is False:
            anomalies.append("fm_stats: la replica de dbscan.py no coincide con kn.feature_map")
        C.write_json_atomic(tmp / "fm_stats.json", fs)
        outputs.append("fm_stats.json")
    except Exception:
        anomalies.append("fm_stats.json: " + traceback.format_exc()[-1500:])
    try:
        with open(tmp / "train_log.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ae_index", "class", "n_visible", "n_hidden", "epochs", "train_steps", "optimizer_iterations",
                        "loss_epoch_1", "loss_source", "train_phase_seconds"])
            for i, (ae, fmap) in enumerate(zip(kn.autoencoders, kn.feature_map)):
                loss, src, iters, steps = "NA", "NA", "NA", "NA"
                model = getattr(ae, "model", None)
                hist = getattr(getattr(model, "history", None), "history", None)
                if isinstance(hist, dict) and hist.get("loss"):
                    loss, src = repr(float(hist["loss"][0])), "model.history"
                elif model is not None:
                    try:
                        tracker = {m.name: m for m in model.metrics}.get("loss")
                        if tracker is not None:
                            loss = repr(float(tracker.result().numpy()))
                            src = ("model.metrics['loss'] (media de la epoca; model.predict sustituye model.history, "
                                   "keras training.py:2637, y no reinicia metricas)")
                    except Exception:
                        pass
                if model is not None:
                    try:
                        iters = int(model.optimizer.iterations.numpy())
                    except Exception:
                        pass
                    try:
                        steps = int(model._train_counter.numpy())
                    except Exception:
                        pass
                w.writerow([i, type(ae).__name__, len(fmap), getattr(ae, "n_hidden", "NA"), getattr(ae, "haste", "NA"),
                            steps, iters, loss, src, obs.t.get("train_s", "NA")])
        outputs.append("train_log.csv")
    except Exception:
        anomalies.append("train_log.csv: " + traceback.format_exc()[-1500:])
    try:
        recs, variant = [], getattr(spec, "variant", None)
        for i, (ae, fmap) in enumerate(zip(kn.autoencoders, kn.feature_map)):
            conv = getattr(ae, "convergence", None)
            if conv is None:
                continue
            recs.append({"ae_index": i, "class": type(ae).__name__, "n_visible": len(fmap),
                         "n_hidden": getattr(ae, "n_hidden", None), "features": [int(f) for f in fmap],
                         **{k: v for k, v in conv.items()}})
        if recs:
            g = [r.get("gates", {}) for r in recs]
            rep = {"job_id": getattr(spec, "job_id", None), "variant": variant,
                   "ar": getattr(spec, "ar", None), "block": getattr(spec, "block", None),
                   "dataset": getattr(spec, "dataset", None), "seed": getattr(spec, "seed", None),
                   "rep": getattr(spec, "rep", None),
                   "decision": "par re-entrenado",
                   "gate_thresholds": {"CG1_pass": "mse_val < 0.9 * suelo del predictor constante",
                                       "CG1_degenerate": "ratio > 1.15",
                                       "CG4_pass": "output_std_between_windows > 1e-3",
                                       "CG0_pass": "paso de Adam sobre la preactivacion < 1.0",
                                       "CG2": "se comprueba emparejando los dos jobs de la celda "
                                              "(revision/analysis/containment_t8.py); aqui solo van los datos"},
                   "summary": {"n_aes": len(recs),
                               "CG1_pass": int(sum(1 for x in g if x.get("CG1_pass"))),
                               "CG3_pass": int(sum(1 for x in g if x.get("CG3_pass"))),
                               "CG4_pass": int(sum(1 for x in g if x.get("CG4_pass"))),
                               "CG0_pass": int(sum(1 for x in g
                                                   if isinstance(x.get("CG0_first_step"), dict)
                                                   and x["CG0_first_step"].get("pass"))),
                               "all_aes_trained": bool(recs and all(x.get("CG1_pass") for x in g)),
                               "degenerate_aes": [r["ae_index"] for r, x in zip(recs, g)
                                                  if x.get("CG1_verdict") == "degenerado"]},
                   "autoencoders": recs}
            C.write_json_atomic(tmp / "convergence_report.json", rep)
            outputs.append("convergence_report.json")
    except Exception:
        anomalies.append("convergence_report.json: " + traceback.format_exc()[-1500:])
    return arrays, outputs


def save_checkpoint(kn, path, variant):
    path = Path(path)
    t0 = time.time()
    if variant == "original":
        path.mkdir()
        kn.save(str(path / "kitnet"))
        method = "KitNET.save(ckpt/kitnet) -> ckpt/kitnet_model.pkl (kitnet.py:219-221)"
    else:
        kn.save(str(path))
        method = "KitNET.save(ckpt) -> ae{i}/{model,encoder,decoder} SavedModel + class.pkl, output.pkl, model.pkl (kitnet.py:222-251)"
    sidecar = None
    if variant in C.CONTEXT_VARIANTS:
        import networks_ctx
        cls = networks_ctx.CONTEXT_NETWORK_TYPES[variant]
        sidecar = {"decision": "MLP con ventana", "variant": variant, "class": cls.__name__, "pickled_module": cls.__module__,
                   "module_file": networks_ctx.__file__, "module_sha256": C.sha256_file(networks_ctx.__file__),
                   "error_fn": cls.ERROR_FN,
                   "how_to_load": ("PYTHONPATH=<repo>/revision/experiments <repo>/envs/tf215/bin/python -c "
                                   "\"import networks_ctx, src.models.kitnet as k; ...\"  (el modulo 'networks_ctx' "
                                   "debe ser importable con ESE nombre antes de despicklear)"),
                   "rule": "las clases no estan en src/; viven en revision/experiments/networks_ctx.py"}
        C.write_json_atomic(path / "networks_ctx_provenance.json", sidecar)
    files = [str(p.relative_to(path)) for p in sorted(path.rglob("*")) if p.is_file()]
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return {"method": method, "path": "ckpt", "seconds": time.time() - t0, "n_files": len(files), "bytes": size,
            "top_level": sorted(p.name for p in path.iterdir()),
            "context_variant_provenance": sidecar,
            "load_caveat": "KitNET.load anade los AEs en el orden de os.listdir (kitnet.py:274), no ordenado: "
                           "ordenar ae0..ae3 antes de reconstruir" +
                           ("; ademas el pickle referencia el modulo 'networks_ctx': ver "
                            "ckpt/networks_ctx_provenance.json" if sidecar else "")}


def collect_env(code_root, device, test_mode):
    env = {"host": C.hostname(), "user": os.environ.get("USER"), "date": C.now_iso(), "python": sys.version,
           "platform": platform.platform(), "device": device, "test_mode": test_mode}
    vers = {}
    for mod in ("tensorflow", "keras", "numpy", "pandas", "sklearn", "scipy", "matplotlib", "tqdm"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            vers[mod] = f"no importable: {e}"
    env["versions"] = vers
    try:
        import tensorflow as tf
        env["tf_build_info"] = {k: str(v) for k, v in tf.sysconfig.get_build_info().items()}
        env["tf_physical_gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
        env["tf_logical_gpus"] = [g.name for g in tf.config.list_logical_devices("GPU")]
        det = []
        for g in tf.config.list_physical_devices("GPU"):
            try:
                det.append({k: str(v) for k, v in tf.config.experimental.get_device_details(g).items()})
            except Exception as e:
                det.append(str(e))
        env["tf_gpu_details"] = det
        env["threads"] = thread_env()
        env["threads"].update({"intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
                               "inter_op": tf.config.threading.get_inter_op_parallelism_threads()})
        from tensorflow.python.framework import config as fcfg
        env["determinism_as_is"] = {"is_op_determinism_enabled": bool(fcfg.is_op_determinism_enabled()),
                                    "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
                                    "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
                                    "note": "no se modifican"}
    except Exception as e:
        env["tf_error"] = str(e)
    env.setdefault("threads", thread_env())
    env["nvidia_smi"] = _nvidia_smi_gpu()
    try:
        cpu = [ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
               if ln.startswith("model name")]
        env["cpu_model"] = cpu[0] if cpu else None
        mem = {ln.split(":")[0]: ln.split(":")[1].strip() for ln in Path("/proc/meminfo").read_text().splitlines()[:3]}
        env["meminfo"] = mem
    except Exception:
        pass
    env["env_vars"] = {k: os.environ.get(k) for k in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED",
                                                      "TF_FORCE_GPU_ALLOW_GROWTH", "TF_CPP_MIN_LOG_LEVEL", "MPLBACKEND",
                                                      "PYTHONDONTWRITEBYTECODE") + THREAD_ENV_VARS}
    env["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED")
    env["flush_denormal"] = "N/A (TF; torch.set_flush_denormal no aplica)"
    env["git"] = {"code_root": str(code_root), "head": C.head_commit(code_root), "tags_at_head": C.tags_at_head(code_root),
                  "runner_tag": C.runner_tag_at_head(code_root), "submitted_v1": C.tag_commit(code_root, "submitted-v1")}
    return env


THREAD_ENV_VARS = ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "JISA_WORKER_THREADS")


def thread_env():
    """Thread settings requested through the environment (worker.py --threads) and the thread count of the process right now."""
    d = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
    d["cpu_count"] = os.cpu_count()
    try:
        d["process_threads_now"] = len(os.listdir(f"/proc/{os.getpid()}/task"))
    except OSError:
        d["process_threads_now"] = None
    try:
        d["sched_affinity_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        d["sched_affinity_cpus"] = None
    return d


def _nvidia_smi_gpu():
    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return p.stdout.strip().splitlines()
    except Exception as e:
        return str(e)


def code_hashes(code_root, runner_dir):
    code_root, runner_dir = Path(code_root), Path(runner_dir)
    orig = ["src/models/kitnet.py", "src/models/oopsie.py", "src/models/__special__.py", "src/models/cluster/dbscan.py",
            "src/models/networks/conv1d.py", "src/models/networks/conv2d.py", "src/models/networks/lstm.py",
            "src/models/networks/mha.py", "src/models/networks/multilayer.py", "src/models/networks/original_kitsune.py",
            "src/models/networks/stdev.py", "src/models/utils/sequential.py", "experiments/frames/launch_all.py",
            "src/database/test_database/database.py"]
    res = {"original": {}, "runner": {}}
    for f in orig:
        p = code_root / f
        res["original"][f] = C.sha256_file(p) if p.exists() else None
    for f in ("run_job.py", "worker.py", "pipelines.py", "instrument.py", "common.py", "smoke.py", "make_manifest.py",
              "networks_ctx.py"):
        p = runner_dir / f
        res["runner"][f] = C.sha256_file(p) if p.exists() else None
    return res


def json_safe_float(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else str(f)
    except Exception:
        return None
