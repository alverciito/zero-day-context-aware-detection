#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selection runs: trains one autoencoder per cell on training rows only and records its validation MSE ratio."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import traceback

T_ARRANQUE = time.time()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
R = "/home/acid137/Projects/review-zd"
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R)
sys.path.insert(0, AQUI)
import numpy as np
import tensorflow as tf
import modelos_rep as M
from src.models.utils import create_windowed_data

STREAMS = ["cic2018", "aci-iot-2023", "knad-active_wiretap"]
L = 800
FASES = {"criba": 1, "confirma": 2}
FILAS_CRIBA = 20_000
TOPE_S = 8 * 3600
REG = f"{AQUI}/registro_intentos.jsonl"
ENMIENDA = "analysis-plan-enmienda-1"


def ahora():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def lee(reg):
    if not os.path.exists(reg):
        return []
    return [json.loads(l) for l in open(reg) if l.strip()]


def anota(reg, rec):
    with open(reg, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _t(ts):
    return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))


def _celdas_ok(rs, fase, con_ref=True):
    return [float(r["segundos"]) for r in rs if r.get("tipo") == "celda" and not r.get("error")
            and r.get("fase") == fase and (con_ref or r.get("modelo") != "ref")]


def _esperado(rs, fase, modelo):
    """Duracion de una celda de (fase, modelo) segun las ya terminadas. Para un modelo con ventana no cuentan
    las de la referencia por registro, que son mucho mas cortas. 0.0 si no hay con que estimar."""
    con_ref = (modelo == "ref")
    mias, criba = _celdas_ok(rs, fase, con_ref), _celdas_ok(rs, "criba", con_ref)
    if fase == "confirma":
        return max(max(mias) if mias else 0.0, 7.5 * max(criba) if criba else 0.0)
    return max(mias) if mias else 0.0


def _techo(rs, fase, modelo):
    """Lo mas que se carga a una celda interrumpida: lo que dura una celda como ella (una hora si no se sabe)."""
    return _esperado(rs, fase, modelo) or 3600.0


def gastado(reg):
    """Wall-clock seconds charged to the GPU budget, with a capped share for every cell that started and did not end."""
    rs = lee(reg)
    cerradas = {r.get("uid") for r in rs if r.get("tipo") == "celda"}
    tot = 0.0
    for i, r in enumerate(rs):
        if r.get("tipo") in ("celda", "proceso", "identidad"):
            tot += float(r.get("segundos", 0.0))
        elif r.get("tipo") == "inicio" and r.get("uid") not in cerradas:
            fin = _t(rs[i + 1]["ts"]) if i + 1 < len(rs) and rs[i + 1].get("ts") else time.time()
            tot += min(max(0.0, fin - _t(r["ts"])), _techo(rs, r.get("fase"), r.get("modelo")))
    return tot


def prevision(reg, fase, modelo):
    """Duracion esperada de la siguiente celda cuando este proceso aun no ha ejecutado ninguna."""
    return _esperado(lee(reg), fase, modelo)


def _job_original(stream):
    c = sorted(glob.glob(f"{R}/results/revision/T1-A-*__{stream}__original__*__sA__r1/config.json"))
    if not c:
        raise FileNotFoundError(f"sin job T1-A de original (semilla A) para {stream}")
    return os.path.dirname(c[0])


def tramo_ae(stream):
    """Tramo de entrenamiento de los autoencoders de ESTE stream (config.json del job T1-A de original)."""
    a, b = json.load(open(f"{_job_original(stream)}/config.json"))["regions"]["ae_train"]
    return int(a), int(b)


def filas_fase(stream, fase):
    a, b = tramo_ae(stream)
    if fase == "criba":
        assert b - a >= FILAS_CRIBA, f"{stream}: tramo de entrenamiento menor que la criba"
        return a, a + FILAS_CRIBA
    return a, b


def grupos(stream):
    return json.load(open(f"{_job_original(stream)}/fm_stats.json"))["feature_map"]


def datos(stream, cols, a, b):
    ta, tb = tramo_ae(stream)
    assert ta <= a < b <= tb, f"{stream}: [{a}, {b}) se sale del tramo de entrenamiento [{ta}, {tb})"
    x = np.load(f"{R}/data/cache/streams/{stream}__corrected__v1/x.npy", mmap_mode="r")
    return np.asarray(x[a:b][:, cols], dtype=np.float64)


def _originales():
    from src.models.networks.conv1d import Conv1DAutoencoder
    from src.models.networks.conv2d import Conv2DAutoencoder
    from src.models.networks.mha import MHAAutoencoder
    return {"conv1d": Conv1DAutoencoder, "conv2d": Conv2DAutoencoder, "mha": MHAAutoencoder}


def corre(modelo, factores, stream, g, cols, fase, semilla, filas_smoke=None):
    a, b = filas_fase(stream, fase)
    if filas_smoke:
        b = a + int(filas_smoke)
    seg = datos(stream, cols, a, b)
    nval = len(seg) // 10
    tr, va = seg[:-nval], seg[-nval:]
    mn, mx = tr.min(0), tr.max(0)
    xt = (tr - mn) / (mx - mn + 1e-16)
    xv = (np.clip(va, mn, mx) - mn) / (mx - mn + 1e-16)
    n_v = len(cols)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(semilla)
    t0 = time.time()
    Pc = None
    if modelo == "ref":
        m, enc = M.construir_referencia(n_v)
        x32 = xt.astype("float32")
        h = m.fit(tf.data.Dataset.from_tensor_slices((x32, x32)).batch(1), epochs=1, verbose=0)
        t_train = time.time() - t0
        P = m.predict(xv.astype("float32"), batch_size=4096, verbose=0)
        H = enc.predict(xv.astype("float32"), batch_size=4096, verbose=0)
        Y = xv
    else:
        if factores == "ORIG":
            ae = _originales()[modelo](n_visible=n_v, hidden_ratio=M.HIDDEN_RATIO, lr=M.LR0, sequence_length=L,
                                       ar=False, haste=True)
            m, enc = ae.model, ae.encoder
        else:
            m, enc = M.construir(modelo, n_v, L, M.parse_factores(factores))
        ds, ult = create_windowed_data(xt, L, np.zeros((L, n_v)), batch_size=1)
        h = m.fit(ds, epochs=1, verbose=0)
        t_train = time.time() - t0
        lote = 64 if modelo == "mha" else 256
        dv, _ = create_windowed_data(xv, L, ult, batch_size=lote)
        V = np.lib.stride_tricks.sliding_window_view(np.concatenate([ult, xv], axis=0)[1:], L, axis=0)
        assert V.shape[0] == nval
        P, Y, H, Pc = [], [], [], []
        i0 = 0
        for xb, yb in dv:
            xb_np = xb.numpy()
            nb = xb_np.shape[0]
            if i0 == 0:
                assert np.allclose(xb_np[0], V[0].T) and np.allclose(xb_np[0, -1], xv[0]), "desfase de ventanas"
            P.append(m(xb, training=False).numpy())
            H.append(enc(xb, training=False).numpy())
            Y.append(yb.numpy())
            j = (np.arange(i0, i0 + nb) + nval // 2) % nval
            xc = np.transpose(V[j], (0, 2, 1)).copy()
            xc[:, -1, :] = xb_np[:, -1, :]
            Pc.append(m(xc, training=False).numpy())
            i0 += nb
        P, Y, H, Pc = np.concatenate(P), np.concatenate(Y), np.concatenate(H), np.concatenate(Pc)
    Y = np.asarray(Y, dtype=np.float64)
    suelo = float(np.mean((Y - Y.mean(axis=0)) ** 2))
    mse = float(np.mean((P - Y) ** 2))
    finito = bool(np.isfinite(mse))
    razon = (mse / suelo) if (finito and suelo > 0) else None
    msec = float(np.mean((Pc - Y) ** 2)) if Pc is not None else None
    return {
        "n_v": n_v, "n_h": M.n_hidden(n_v), "filas_train": int(len(xt)), "filas_val": int(len(xv)),
        "desde": a, "hasta": b, "tramo_ae": list(tramo_ae(stream)), "razon": razon,
        "val_mse": mse if finito else None, "suelo": suelo, "loss_epoca": float(h.history["loss"][-1]),
        "std_salida": float(np.std(P, axis=0).mean()), "muertas": float(np.mean(np.std(H, axis=0) < 1e-8)),
        "cociente_contexto_cambiado": (msec / mse) if (msec is not None and finito and mse > 0
                                                       and np.isfinite(msec)) else None,
        "lr_por_capa": ({str(k): {"fan_in": int(v["fan_in"]), "lr": float(v["lr"])}
                         for k, v in dict(m.lr_map).items()} if getattr(m, "lr_map", None) else None),
        "seg_train": round(t_train, 1), "segundos": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", choices=list(FASES), default="criba")
    ap.add_argument("--modelo", choices=["conv1d", "conv2d", "mha", "ref"])
    ap.add_argument("--factores", default="", help="letras separadas por comas, u ORIG para la clase de src/")
    ap.add_argument("--escalon", default="-")
    ap.add_argument("--celdas", default="", help="stream:grupo,... (por defecto, los 12 grupos)")
    ap.add_argument("--registro", default=REG)
    ap.add_argument("--smoke", type=int, default=0, help="filas totales; exige --registro distinto del oficial")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--identidad", action="store_true")
    ap.add_argument("--semilla", type=int, default=0, help="por defecto, la de la fase (criba 1, confirma 2)")
    a = ap.parse_args()
    t_proc = T_ARRANQUE
    if a.smoke and os.path.abspath(a.registro) == os.path.abspath(REG):
        print("ABORT: --smoke no puede escribir en el registro oficial"); return 2
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus and not a.cpu:
        print("ABORT: sin GPU (usa --cpu solo para pruebas)"); return 2
    commit = subprocess.run(["git", "-C", R, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    base = {"ts": None, "commit": commit, "enmienda": ENMIENDA, "dispositivo": "GPU" if gpus else "CPU",
            "smoke": bool(a.smoke)}

    if a.identidad:
        res = M.comprobar_identidad(L)
        ok = all(r["ok"] for r in res)
        anota(a.registro, dict(base, ts=ahora(), tipo="identidad", ok=ok, detalle=res,
                               segundos=round(time.time() - t_proc, 1)))
        for r in res:
            print(f"  identidad {r['familia']:7s} n_v={r['n_v']:3d} params={r['params']} "
                  f"orig={r['params_orig']} -> {'OK' if r['ok'] else 'DISTINTO'}", flush=True)
        return 0 if ok else 4

    if a.modelo != "ref" and a.factores != "ORIG":
        M.construir(a.modelo, 8, L, M.parse_factores(a.factores))
        tf.keras.backend.clear_session()
    if a.celdas:
        celdas = [(c.split(":")[0], int(c.split(":")[1])) for c in a.celdas.split(",")]
    else:
        celdas = [(s, g) for s in STREAMS for g in range(len(grupos(s)))]
    semilla = a.semilla or FASES[a.fase]
    hechas = {(r["stream"], r["grupo"]) for r in lee(a.registro)
              if r.get("tipo") == "celda" and not r.get("error") and r["fase"] == a.fase
              and r["modelo"] == a.modelo and r["factores"] == a.factores and r.get("semilla") == semilla
              and bool(r.get("smoke")) == bool(a.smoke)}
    ult, en_celdas, rc, n = 0.0, 0.0, 0, 0
    for stream, g in celdas:
        if (stream, g) in hechas:
            print(f"  ya hecha: {a.fase} {a.modelo} [{a.factores}] {stream}:{g}", flush=True)
            continue
        prev = ult if ult > 0 else prevision(a.registro, a.fase, a.modelo)
        if not a.smoke and gastado(a.registro) + (time.time() - t_proc - en_celdas) + prev > TOPE_S:
            anota(a.registro, dict(base, ts=ahora(), tipo="tope", fase=a.fase, modelo=a.modelo,
                                   factores=a.factores, pendiente=f"{stream}:{g}",
                                   gastado_s=round(gastado(a.registro), 1), tope_s=TOPE_S))
            print("TOPE de GPU alcanzado: paro.", flush=True)
            rc = 3
            break
        n += 1
        uid = f"{os.getpid()}-{n}"
        rec = dict(base, ts=ahora(), tipo="celda", uid=uid, fase=a.fase, modelo=a.modelo, factores=a.factores,
                   escalon=a.escalon, stream=stream, grupo=g, semilla=semilla)
        anota(a.registro, dict(rec, tipo="inicio"))
        t0 = time.time()
        try:
            rec.update(corre(a.modelo, a.factores, stream, g, grupos(stream)[g], a.fase, semilla,
                             a.smoke or None))
            rec["ts"] = ahora()
        except Exception as e:
            rec.update(error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-600:],
                       razon=None, segundos=round(time.time() - t0, 1))
        anota(a.registro, rec)
        ult = float(rec["segundos"])
        en_celdas += ult
        rz = rec.get("razon")
        print(f"  {a.fase:8s} {a.modelo:6s} [{a.factores or '-':9s}] {stream}:{g} n_v={rec.get('n_v', '?')} "
              f"razon={'%.4f' % rz if rz is not None else 'None'} suelo={rec.get('suelo', float('nan')):.2e} "
              f"muertas={rec.get('muertas', float('nan')):.2f} ctx={rec.get('cociente_contexto_cambiado')} s{semilla} "
              f"{rec['segundos']:.0f}s {rec.get('error', '')}", flush=True)
    anota(a.registro, dict(base, ts=ahora(), tipo="proceso", fase=a.fase, modelo=a.modelo,
                           factores=a.factores, segundos=round(time.time() - t_proc - en_celdas, 1),
                           nota="reloj del proceso fuera de las celdas"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
