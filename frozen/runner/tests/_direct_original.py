#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ruta DIRECTA del test (c): ejecuta el codigo original SIN el runner sobre la misma entrada (replica de
database.py:51-55 con nrows + truncado de launch_all.py:40-42), con los mismos argumentos de KitNET que
launch_all.py:54-65, el mismo bucle y el show_stats REAL (guarda <out>/<variant>_ar_<ar>.npy/.eps/.png).
No importa nada de revision/experiments.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--attack", default="OS_Scan")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--ar", default="True")
    ap.add_argument("--seed", default="None")
    ap.add_argument("--max-rows", type=int, required=True)
    ap.add_argument("--fm", type=int, required=True)
    ap.add_argument("--ad", type=int, required=True)
    ap.add_argument("--ew", type=int, required=True)
    ap.add_argument("--L", type=int, default=800)
    ap.add_argument("--device", default="/CPU:0")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code_root)
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    calls = []
    if a.seed != "None":
        s = int(a.seed)
        random.seed(s)
        np.random.seed(s)
        tf.keras.utils.set_random_seed(s)
        calls = ["random.seed", "numpy.random.seed", "tf.keras.utils.set_random_seed"]
    d = Path(a.data_root) / "KITSUNE" / a.attack.replace("_", " ")
    x_df = pd.read_csv(d / f"{a.attack}_dataset.csv", nrows=a.max_rows)
    y_df = pd.read_csv(d / f"{a.attack}_labels.csv", nrows=a.max_rows + 1)
    x = x_df.to_numpy()
    y = y_df.to_numpy()[:, -1]
    if len(x) != len(y):
        y = y[:len(x)]
    import experiments.frames.launch_all
    from src.models import KitNET
    is_ar = a.ar == "True"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tf.device(a.device):
        kn = KitNET(x.shape[-1],
                    feature_map=None,
                    ad_grace_period=a.ad,
                    execution_window=a.ew,
                    fm_grace_period=a.fm,
                    sequence_length=a.L,
                    hidden_ratio=0.22,
                    autoencoder_size=4,
                    clustering='dbscan',
                    ar=is_ar,
                    ae_type=a.variant,
                    output_ae_type='stat')
        for _, packet in enumerate(x):
            kn.process(packet, is_last=False if _ < len(x) - 1 else True)
        kn.show_stats(y, save=str(out / f"{a.variant}_ar_{is_ar}"))
    (out / "direct.json").write_text(json.dumps({"feature_map": [[int(v) for v in g] for g in kn.feature_map],
                                                 "n": len(kn.current_error), "seed_calls": calls,
                                                 "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}))


if __name__ == "__main__":
    main()
