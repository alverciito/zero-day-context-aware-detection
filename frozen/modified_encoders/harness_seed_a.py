#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""harness_seed_a.py - reruns the seed A Conv1D/Conv2D jobs with modified encoders and reports their convergence."""
import argparse
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

R = Path("/home/acid137/Projects/review-zd")
RESULTS = R / "results" / "revision"
LOGS = R / "logs"
AQUI = R / "revision" / "analysis" / "reparacion"
REGISTRO = AQUI / "registro_intentos.jsonl"
MODELOS_REP = AQUI / "modelos_rep.py"

TAG_ENMIENDA = "analysis-plan-enmienda-1"
TAG_BRAZO = "brazo-rep-v1"
FIJADO_BRAZO = ["revision/analysis/reparacion/brazo_rep.py", "revision/analysis/reparacion/cola_fase_b.sh",
                "revision/analysis/reparacion/puerta_fase_b.py"]
FIJADO_ENMIENDA = ["revision/analysis/reparacion/modelos_rep.py"]
BLOCK = "T10-REP"
BLOCK_CTRL = "T10-REP-CTRL"
FAMILIAS = {
    "conv1d": {"modulo": "src.models.networks.conv1d", "clase": "Conv1DAutoencoder",
               "fichero": "src/models/networks/conv1d.py"},
    "conv2d": {"modulo": "src.models.networks.conv2d", "clase": "Conv2DAutoencoder",
               "fichero": "src/models/networks/conv2d.py"},
}
DATASETS = ("ACI-IOT-2023", "KNAD:Active_Wiretap", "KNAD:Fuzzing", "KNAD:OS_Scan", "KNAD:SSDP_Flood",
            "KNAD:SSL_Renegotiation", "KNAD:Video_Injection", "CIC2017", "CIC2018")
SEED = "A"
SEEDS = {"A": 1234}
ARS = ("T", "F")
FIT_LINE = "self.model.fit(sequential_x, epochs=self.haste, verbose=1)"
EVAL_LINE = ("sequential_x, self.window = self.create_windowed(x, self.sequence_length, self.window, "
             "batch_size=EXECUTION_BS)")
VAL_DIV = 10
NORMALIZACION = "entrenamiento"
UMBRAL, SUELO_MIN, CTX_MIN = 0.90, 1e-4, 1.10
LOTE_VAL = 256
K_RUIDO = 10.0
TOPE_FAMILIA_S = 20 * 3600
FACTOR_INICIAL = 1.25
RAM_FACTOR, RAM_SISTEMA, RAM_DEFECTO, RAM_ESPERA_MAX_S = 1.10, 6 << 30, 45 << 30, 6 * 3600
MARK = "[T10-REP"
CHILD_ENV = {"TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_CPP_MIN_LOG_LEVEL": "1", "MPLBACKEND": "Agg",
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "LD_LIBRARY_PATH": ""}
THREAD_VARS = ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "JISA_WORKER_THREADS")
NEEDED_COMMON = ("JobPaths", "validate_row", "check_supported", "adopt_stale_tmps", "preserve_tmp", "commit",
                 "list_previous_attempts", "record_failure", "read_fail_count", "check_not_forbidden",
                 "is_registered_worktree", "worktree_dirty", "runner_tag_at_head", "head_commit", "tags_at_head",
                 "src_identical", "ckpt_paths_ignored", "write_json_atomic", "read_json", "sha256_file",
                 "gpu_compute_apps", "hostname", "now_iso", "is_under", "Interrupted", "HaltError",
                 "FrozenConfigError", "CACHE_ROOT", "DATA_ROOT", "MANIFEST", "MAX_FAILS")
RUN_REAL_PARAMS = ["spec", "a", "overrides", "code_root", "results_root", "tmp", "hb", "cfg", "timing", "log"]


def ahora():
    """Marca de tiempo local ISO, al segundo."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256_file(path, chunk=1 << 22):
    """sha256 de un fichero.

    :param path: ruta del fichero.
    :return: hexdigest.
    :rtype: str
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def leer_json(path, default=None):
    """Lee un JSON; devuelve `default` si no existe o no se puede leer."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def slug(dataset):
    """Mismo slug que common.ds_slug (se comprueba contra el runner en el hijo)."""
    if dataset.upper().startswith("KNAD:"):
        return "knad-" + dataset.split(":", 1)[1].lower()
    return dataset.lower()


def corpus(dataset):
    """Corpus del bloque T1 del dataset (KNAD, ACI, CIC2017 o CIC2018)."""
    return "KNAD" if dataset.upper().startswith("KNAD:") else {"ACI-IOT-2023": "ACI"}.get(dataset, dataset)


def resolver_dataset(texto):
    """Admite el nombre del manifiesto ("KNAD:Fuzzing") o su slug ("knad-fuzzing").

    :param str texto: lo que se paso en --dataset.
    :return: nombre del manifiesto.
    :rtype: str
    :raises SystemExit: si no es ninguno de los 9 streams.
    """
    for ds in DATASETS:
        if texto in (ds, slug(ds)) or texto.lower() == ds.lower():
            return ds
    raise SystemExit("dataset desconocido %r; validos: %s" % (texto, ", ".join(DATASETS)))


def ref_job_id(dataset, familia, ar):
    """job_id del job T1 de la rejilla que sirve de referencia (misma familia, AR/TSR y semilla A)."""
    return "T1-%s-%s__%s__%s__ar%s__corrected.v1__L800__k4__s%s__r1" % (SEED, corpus(dataset), slug(dataset), familia,
                                                                      ar, SEED)


def t10_job_id(dataset, familia, ar, control=False, replica=1):
    """job_id of a modified-encoder job, or of its control, which keeps the name of the original variant."""
    return "%s__%s__%s__ar%s__corrected.v1__L800__k4__s%s__r%d" % (
        BLOCK_CTRL if control else BLOCK, slug(dataset), familia if control else familia + "_rep", ar, SEED, replica)


def dir_control(results_root):
    """Donde viven el reloj, la reserva de RAM y los cerrojos de los padres (fuera de los directorios de job)."""
    root = Path(results_root).resolve()
    return (LOGS / "fase-b") if root == RESULTS.resolve() else (root / ".fase-b")


def lineas_marcadas(texto):
    """Lineas propias dentro de un trozo de log (la barra de progreso de Keras usa \\r)."""
    return [ln for ln in re.split(r"[\r\n]+", texto) if MARK in ln]


def hacer_log(job_id):
    """Devuelve la funcion de log del hijo, con el mismo formato que run_job."""
    def log(msg):
        print("%s %s %d %s] %s" % (ahora(), MARK, os.getpid(), job_id, msg), flush=True)
    return log


def norm_factores(txt):
    """'P, I,R' -> 'P,I,R' (mismas letras y orden que escribe escalera.py)."""
    return ",".join(f for f in (txt or "").replace(" ", "").split(",") if f)


def arreglo_elegido(familia, registro=REGISTRO):
    """Reads, read-only, what the selection runs chose for the family; RuntimeError if no modification was chosen and confirmed."""
    if familia not in FAMILIAS:
        raise RuntimeError("family %r has no modified version (only %s; none was found for MHA)" % (familia, sorted(FAMILIAS)))
    registro = Path(registro)
    if not registro.exists():
        raise RuntimeError("no existe el registro de la fase A: %s" % registro)
    rs = []
    with open(registro, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                try:
                    rs.append(json.loads(ln))
                except ValueError:
                    continue
    dec = [r for r in rs if r.get("tipo") == "decision" and r.get("clase") == "familia" and r.get("familia") == familia]
    if not dec:
        raise RuntimeError("%s: la fase A no ha anotado ninguna decision de familia" % familia)
    res = str(dec[-1].get("resultado") or "")
    m = re.match(r"^ELEGIDO escalon (\d+): ([A-Z](?:,[A-Z])*)$", res)
    if not m:
        raise RuntimeError("%s: the last decision of the selection runs does not choose a modification: %r" % (familia, res))
    cn = [r for r in rs if r.get("tipo") == "decision" and r.get("clase") == "control_negativo_confirma"
          and r.get("familia") == familia]
    if not cn:
        raise RuntimeError("%s: falta el control negativo con el presupuesto completo (enmienda 3.3)" % familia)
    if cn[-1].get("veredicto") != "no_pasa":
        raise RuntimeError("%s: control negativo completo con veredicto %r, se exige 'no_pasa'"
                           % (familia, cn[-1].get("veredicto")))
    return m.group(2), {"registro": str(registro), "registro_sha256": sha256_file(registro), "lineas": len(rs),
                        "escalon": int(m.group(1)), "decision_familia": dec[-1], "control_negativo_confirma": cn[-1]}


def puerta_git(con_brazo=True):
    """El codigo que se va a ejecutar es el fijado por los tags (como la puerta de escalera.py).

    :return: None si todo esta en orden, o el motivo.
    """
    def git(*a):
        return subprocess.run(["git", "-C", str(R)] + list(a), capture_output=True, text=True)
    grupos = [(TAG_ENMIENDA, FIJADO_ENMIENDA)] + ([(TAG_BRAZO, FIJADO_BRAZO)] if con_brazo else [])
    for tag, ficheros in grupos:
        if not git("tag", "-l", tag).stdout.strip():
            return "falta el tag %s" % tag
        if git("status", "--porcelain", "--", *ficheros).stdout.strip():
            return "hay cambios sin commit en %s" % ", ".join(ficheros)
        if git("diff", "--quiet", tag, "HEAD", "--", *ficheros).returncode != 0:
            return "el codigo de HEAD no es el fijado por el tag %s (%s)" % (tag, ", ".join(ficheros))
    return None


def cargar_modelos_rep():
    """Importa modelos_rep.py por su ruta (el directorio de este guion no esta en sys.path del hijo)."""
    spec = importlib.util.spec_from_file_location("modelos_rep", str(MODELOS_REP))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modelos_rep"] = mod
    spec.loader.exec_module(mod)
    return mod


def firma_reparado(M, m, enc, dec, familia, f, n_v, L, iniciales=True):
    """Comprueba capa a capa que (m, enc, dec) es el modelo de `modelos_rep.construir(familia, n_v, L, f)`.

    :param bool iniciales: True si los pesos son los de la inicializacion (comprueba el nucleo del factor I).
    :return: (dict descriptivo, lista de fallos).
    """
    import numpy as np

    fallos = []
    n_h = int(M.n_hidden(n_v))
    capas = [c for mod in (enc, dec) for c in mod.layers if type(c).__name__ != "InputLayer"]
    nombres = [type(c).__name__ for c in capas]
    es_p = isinstance(m, M.PasoPorCapa)
    if es_p != ("P" in f):
        fallos.append("factor P %s pero el modelo es %s" % ("pedido" if "P" in f else "no pedido", type(m).__name__))
    n_leaky = nombres.count("LeakyReLU")
    n_relu = sum(1 for c in capas if c.get_config().get("activation") == "relu")
    if (n_leaky, n_relu) != ((3, 0) if "A" in f else (0, 3)):
        fallos.append("factor A: %d LeakyReLU y %d ReLU" % (n_leaky, n_relu))
    if any(abs(float(c.get_config().get("alpha", 0.01)) - 0.01) > 1e-9 for c in capas if type(c).__name__ == "LeakyReLU"):
        fallos.append("factor A: negative slope other than 0.01")
    n_lambda, n_concat = nombres.count("Lambda"), nombres.count("Concatenate")
    if (n_lambda, n_concat) != ((1, 1) if "R" in f else (0, 0)):
        fallos.append("factor R: %d Lambda y %d Concatenate" % (n_lambda, n_concat))
    densas_enc = [c for c in enc.layers if type(c).__name__ == "Dense"]
    extra = (n_h if familia == "conv1d" else int(np.ceil(n_v / 2.0)) * n_h) if "R" in f else 0
    entra = int(densas_enc[-1].kernel.shape[0]) if densas_enc else None
    if len(densas_enc) != 1 or entra != n_h + extra or int(densas_enc[-1].kernel.shape[1]) != n_h:
        fallos.append("densa del cuello con entrada %s (se esperaba %d) o salida != %d" % (entra, n_h + extra, n_h))
    convs = [c for c in enc.layers if type(c).__name__ in ("Conv1D", "Conv2D")]
    esperado = ([("Conv1D", n_v, (L // 4,)), ("Conv1D", n_h, (L // 4,))] if familia == "conv1d"
                else [("Conv2D", 32, (3, 3)), ("Conv2D", n_h, (3, 3))])
    visto = [(type(c).__name__, int(c.filters), tuple(int(k) for k in c.kernel_size)) for c in convs]
    if visto != esperado:
        fallos.append("convoluciones %s != %s" % (visto, esperado))
    densas_dec = [c for c in dec.layers if type(c).__name__ == "Dense"]
    if [int(c.units) for c in densas_dec] != [n_h, n_v] or densas_dec[-1].get_config().get("activation") != "sigmoid":
        fallos.append("decodificador %s != [%d, %d] con sigmoide" % ([int(c.units) for c in densas_dec], n_h, n_v))
    if tuple(int(v) for v in enc.input_shape[1:]) != (L, n_v):
        fallos.append("entrada del codificador %s != (%d, %d)" % (enc.input_shape, L, n_v))
    toma = []
    for c in convs:
        ini = type(c.kernel_initializer).__name__
        if (ini == "TapOrtogonal") != ("I" in f):
            fallos.append("factor I: %s inicializada con %s" % (c.name, ini))
        if iniciales:
            k = c.kernel.numpy()
            pos = tuple((int(s) - 1) // 2 for s in k.shape[:-2])
            q = k[pos].astype(np.float64)
            fuera = k.copy()
            fuera[pos] = 0.0
            nz = int(np.count_nonzero(fuera))
            g = (q.T @ q) if q.shape[0] >= q.shape[1] else (q @ q.T)
            orto = bool(np.allclose(g, np.eye(g.shape[0]), atol=1e-4))
            toma.append({"capa": c.name, "posicion": list(pos), "no_nulos_fuera": nz, "semiortogonal": orto})
            if "I" in f and (nz != 0 or not orto):
                fallos.append("factor I: %s con %d pesos fuera de la toma o toma no ortogonal" % (c.name, nz))
            if "I" not in f and nz == 0:
                fallos.append("factor I no pedido y %s tiene el nucleo a cero fuera de la toma" % c.name)
    return {"clase_modelo": type(m).__name__, "capas": nombres, "n_h": n_h, "entrada_cuello": entra,
            "convoluciones": [list(v[:2]) + [list(v[2])] for v in visto], "nucleos_iniciales": toma or None,
            "params_entrenables": int(sum(int(np.prod(v.shape)) for v in m.trainable_variables))}, fallos


def regla_paso(m, enc, dec, n_v, lr0, pasos):
    """Factor P tras entrenar: lr por variable = lr0 * min(1, n_v / fan_in) y un paso de Adam por fila.

    :return: (lr por variable, pasos por grupo, lista de fallos).
    """
    import numpy as np

    fallos, ult = [], None
    mapa = {str(k): {"fan_in": int(v["fan_in"]), "lr": float(v["lr"])} for k, v in dict(m.lr_map).items()}
    variables = [v for mod in (enc, dec) for v in mod.trainable_variables]
    for mod in (enc, dec):
        ult = None
        for v in mod.trainable_variables:
            sh = [int(s) for s in v.shape]
            fi = int(np.prod(sh[:-1])) if len(sh) >= 2 else (ult if ult else n_v)
            ult = fi if len(sh) >= 2 else ult
            esperado = lr0 * min(1.0, n_v / float(max(fi, 1)))
            d = mapa.get(v.name)
            if d is None or d["fan_in"] != fi or abs(d["lr"] - esperado) > 1e-15:
                fallos.append("lr de %s = %s, se esperaba fan_in %d y lr %.6g" % (v.name, d, fi, esperado))
    grupos = []
    for opt, vs in (m._grupos or []):
        it = int(opt.iterations.numpy())
        grupos.append({"lr": float(opt.learning_rate.numpy()), "variables": len(vs), "pasos": it})
        if it not in (pasos, pasos + 1):
            fallos.append("grupo con lr %.6g: %d pasos de Adam para %d filas" % (grupos[-1]["lr"], it, pasos))
    if sum(g["variables"] for g in grupos) != len(variables) or len(mapa) != len(variables):
        fallos.append("variables en los grupos %d / en lr_map %d != %d entrenables"
                      % (sum(g["variables"] for g in grupos), len(mapa), len(variables)))
    return mapa, grupos, fallos


def selftest(familia, factores):
    """CPU check before any GPU use: identity test of modelos_rep, and signature and steps of the modified model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(R))
    import numpy as np
    import tensorflow as tf
    M = cargar_modelos_rep()

    ident = M.comprobar_identidad(800)
    res = {"tensorflow": tf.__version__, "modelos_rep_sha256": sha256_file(MODELOS_REP),
           "identidad": [{k: r[k] for k in ("familia", "n_v", "ok", "params", "params_orig")} for r in ident],
           "identidad_ok": all(r["ok"] for r in ident)}
    ok = res["identidad_ok"]
    if familia and factores:
        f = M.parse_factores(factores)
        n_v, L, pasos = 8, 800, 6
        tf.keras.utils.set_random_seed(1234)
        m, enc = M.construir(familia, n_v, L, f)
        dec = m.decoder if "P" in f else m.layers[1]
        desc, fallos = firma_reparado(M, m, enc, dec, familia, f, n_v, L)
        x = np.random.RandomState(0).rand(pasos, L, n_v).astype("float32")
        m.fit(tf.data.Dataset.from_tensor_slices((x, x[:, -1, :])).batch(1), epochs=1, verbose=0)
        if "P" in f:
            _, grupos, f2 = regla_paso(m, enc, dec, n_v, M.LR0, pasos)
            fallos += f2
            res["grupos"] = grupos
        if int(m._train_counter.numpy()) != pasos:
            fallos.append("train_counter %d != %d" % (int(m._train_counter.numpy()), pasos))
        res.update(familia=familia, factores=norm_factores(factores), firma=desc, fallos=fallos)
        ok = ok and not fallos
    res["ok"] = bool(ok)
    print("%s selftest] %s" % (MARK, json.dumps(res, ensure_ascii=False)), flush=True)
    return 0 if ok else 1


def diagnostico_cola(model, encoder, crear_ventanas, es_ar, xv, ult, L, lote=LOTE_VAL):
    """Razon de 3.1, persistencia (solo AR) y contexto cambiado de 3.8, con los pesos finales.

    Mismas cuentas que prevuelo.corre() (prevuelo.py:183-216). `xv` es la cola ya recortada y normalizada con
    el rango de entrenamiento; `ult` son las L ultimas filas normalizadas del 90 % que entreno (historia real).

    :return: dict con floats de Python (o None), listo para JSON y para pickle.
    """
    import numpy as np

    n_val = int(len(xv))
    dv, _ = crear_ventanas(xv, L, ult, batch_size=lote)
    base = np.concatenate([ult, xv], axis=0)
    V = np.lib.stride_tricks.sliding_window_view(base[:-1] if es_ar else base[1:], L, axis=0)
    if V.shape[0] != n_val:
        raise ValueError("ventanas de la cola: %d != %d filas" % (V.shape[0], n_val))
    P, Y, H, Pc, U = [], [], [], [], []
    i0 = 0
    for xb, yb in dv:
        xb_np, yb_np = xb.numpy(), yb.numpy()
        nb = xb_np.shape[0]
        if i0 == 0 and not (np.allclose(xb_np[0], V[0].T) and np.allclose(yb_np[0], xv[0])):
            raise ValueError("desfase entre las ventanas de la cola y sus indices")
        P.append(model(xb, training=False).numpy())
        H.append(encoder(xb, training=False).numpy())
        Y.append(yb_np)
        U.append(xb_np[:, -1, :])
        j = (np.arange(i0, i0 + nb) + n_val // 2) % n_val
        xc = np.transpose(V[j], (0, 2, 1)).copy()
        xc[:, -1, :] = xb_np[:, -1, :]
        Pc.append(model(xc, training=False).numpy())
        i0 += nb
    if i0 != n_val:
        raise ValueError("la cola dio %d ventanas, se esperaban %d" % (i0, n_val))
    P, H, Pc, U = np.concatenate(P), np.concatenate(H), np.concatenate(Pc), np.concatenate(U)
    Y = np.asarray(np.concatenate(Y), dtype=np.float64)
    if not np.array_equal(Y, np.asarray(xv, dtype=np.float64)):
        raise ValueError("el objetivo de las ventanas de la cola no son las filas de la cola")
    suelo = float(np.mean((Y - Y.mean(axis=0)) ** 2))
    mse = float(np.mean((P - Y) ** 2))
    finito = bool(np.isfinite(mse))
    razon = (mse / suelo) if (finito and suelo > 0) else None
    msec = float(np.mean((Pc - Y) ** 2))
    ctx = (msec / mse) if (finito and mse > 0 and np.isfinite(msec)) else None
    out = {"paradigma": "AR" if es_ar else "TSR", "filas_val": n_val, "suelo": suelo,
           "val_mse": mse if finito else None, "razon": razon,
           "evaluable": bool(suelo >= SUELO_MIN), "entrenado": bool(razon is not None and razon < UMBRAL),
           "std_salida": float(np.std(P, axis=0).mean()), "muertas": float(np.mean(np.std(H, axis=0) < 1e-8)),
           "val_mse_contexto_cambiado": msec if np.isfinite(msec) else None,
           "cociente_contexto_cambiado": ctx, "depende_contexto": bool(ctx is not None and ctx >= CTX_MIN),
           "contexto_cambiado_nota": ("AR: se conserva la ultima fila de la entrada (la anterior al objetivo) y se "
                                      "cambian las L-1 previas; la enmienda 3.8 solo lo define en TSR: descriptivo"
                                      if es_ar else "TSR: enmienda 3.8 (la fila actual se conserva)"),
           "mse_persistencia": None, "razon_persistencia": None}
    if es_ar:
        mp = float(np.mean((U - Y) ** 2))
        out["mse_persistencia"] = mp if np.isfinite(mp) else None
        out["razon_persistencia"] = (mse / mp) if (finito and np.isfinite(mp) and mp > 0) else None
    return out


def instalar_envoltorio(NET, M, familia, factores, control, log, fits, sha_esperado, error_cls):
    """Wraps in memory __init__, train and execute of the original class: modified model, or observer for the control."""
    import numpy as np
    import tensorflow as tf

    cls = getattr(NET, FAMILIAS[familia]["clase"])
    fuente_fichero = Path(inspect.getsourcefile(cls))
    sha = sha256_file(fuente_fichero)
    if sha_esperado and sha != sha_esperado:
        raise error_cls("%s %s != %s del job T1 de referencia" % (fuente_fichero.name, sha[:12], sha_esperado[:12]))
    fuente = inspect.getsource(cls.train)
    if fuente.count(FIT_LINE) != 1 or fuente.count(EVAL_LINE) != 1:
        raise error_cls("%s.train no contiene las lineas congeladas que el envoltorio reproduce" % cls.__name__)
    if NET.BATCH_SIZE != 1 or NET.EXECUTION_BS != 512 or NET.HASTE_VALUE != 1:
        raise error_cls("constantes de __special__ distintas de las congeladas (lote 1, 512, 1 epoca)")
    f = frozenset() if control else M.parse_factores(factores)
    if not control and (not f or not f <= M.FACTORES[familia] or norm_factores(factores) not in M.ESCALERA[familia]):
        raise error_cls("factores %r fuera de la escalera de %s" % (factores, familia))
    init_original, train_original = cls.__dict__["__init__"], cls.__dict__["train"]
    execute_original = cls.__dict__["execute"]
    vivos = {}

    def huella_pesos(model):
        h = hashlib.sha256()
        for w in model.weights:
            h.update(np.ascontiguousarray(w.numpy()).tobytes())
        return h.hexdigest()

    def huella(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    class Contador(tf.keras.callbacks.Callback):
        """Cuenta los lotes que `fit` entrega al modelo (uno por ventana: lote 1)."""

        def __init__(self):
            super().__init__()
            self.lotes = 0

        def on_train_batch_end(self, batch, logs=None):
            self.lotes += 1

    def __init__(self, *args, **kwargs):
        init_original(self, *args, **kwargs)
        if control:
            return
        n_v, L = int(self.input_dim), int(self.sequence_length)
        if float(self.lr) != float(M.LR0) or int(self.n_hidden) != int(M.n_hidden(n_v)) or int(self.haste) != 1:
            raise error_cls("T10-REP: lr %r, n_hidden %r o epocas %r distintos de los congelados"
                            % (self.lr, self.n_hidden, self.haste))
        m0, _ = M.construir(familia, n_v, L, frozenset())
        fo, co = M._firma(self.model)
        f0, c0 = M._firma(m0)
        params_orig = int(self.model.count_params())
        if fo != f0 or co != c0 or params_orig != int(m0.count_params()):
            raise error_cls("T10-REP: con los factores apagados el constructor no coincide con %s (n_v=%d)"
                            % (cls.__name__, n_v))
        del m0
        m, enc = M.construir(familia, n_v, L, f)
        dec = m.decoder if "P" in f else m.layers[1]
        if "P" in f:
            m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        desc, fallos = firma_reparado(M, m, enc, dec, familia, f, n_v, L)
        if fallos:
            raise error_cls("T10-REP: the model built is not the modification %s of %s: %s"
                            % (sorted(f), familia, "; ".join(fallos)))
        self.encoder, self.decoder, self.model = enc, dec, m
        vivos[id(self)] = {"model": m, "encoder": enc, "decoder": dec, "firma": desc, "params_original": params_orig,
                           "identidad_factores_apagados": True, "pesos_iniciales_sha256": huella_pesos(m),
                           "execute": {"llamadas": 0, "filas": 0}, "train": 0}

    def train(self, input_x):
        rec = {"ae_index": len(fits), "ae_name": getattr(self, "name", None), "clase": type(self).__name__,
               "n_visible": int(self.input_dim), "n_hidden": int(self.n_hidden),
               "x_train_shape": [int(v) for v in np.shape(input_x)], "x_train_dtype": str(np.asarray(input_x).dtype),
               "x_train_sha256": huella(input_x), "ventana": self.create_windowed.__name__}
        N = int(len(input_x))
        fallos = []
        t0 = time.perf_counter()
        if control:
            rec.update(modo="observador", filas_train=N, filas_val=0, pesos_iniciales_sha256=huella_pesos(self.model))
            salida = train_original(self, input_x)
            rec["train_counter"] = int(self.model._train_counter.numpy())
            rec["optimizer_iterations"] = int(self.model.optimizer.iterations.numpy())
            if rec["train_counter"] != N:
                fallos.append("pasos %d != %d filas" % (rec["train_counter"], N))
        else:
            info = vivos.get(id(self))
            if (info is None or self.model is not info["model"] or self.encoder is not info["encoder"]
                    or self.decoder is not info["decoder"]):
                raise error_cls("T10-REP: the model about to train is not the modified one")
            info["train"] += 1
            L, es_ar = int(self.sequence_length), self.create_windowed.__name__ == "create_windowed_data_ar"
            n_val = N // VAL_DIV
            n_tr = N - n_val
            if n_val < 1 or n_tr < 1 or info["train"] != 1 or np.count_nonzero(self.window) != 0:
                raise error_cls("T10-REP: tramo de %d filas sin cola, train repetido o relleno inicial no nulo" % N)
            NET.logging.info(f"Training {self.name}...")
            tr, va = input_x[:n_tr], input_x[n_tr:]
            base = tr if NORMALIZACION == "entrenamiento" else input_x
            self.norm_min = np.min(base, axis=0)
            self.norm_max = np.max(base, axis=0)
            ancho = self.norm_max - self.norm_min + 1e-16
            xt = (tr - self.norm_min) / ancho
            xv = (np.clip(va, self.norm_min, self.norm_max) - self.norm_min) / ancho
            sequential_x, self.window = self.create_windowed(xt, L, self.window, batch_size=NET.BATCH_SIZE)
            ult = self.window
            cuenta = Contador()
            hist = self.model.fit(sequential_x, epochs=self.haste, verbose=1, callbacks=[cuenta])
            loss_epoca = float(hist.history["loss"][-1])
            seg_fit = time.perf_counter() - t0
            NET.logging.info(f"Training {self.name}... Done!")
            conv = diagnostico_cola(self.model, self.encoder, self.create_windowed, es_ar, xv, ult, L)
            x_all = np.concatenate([xt, xv], axis=0)
            seq_all, self.window = self.create_windowed(x_all, L, np.zeros_like(self.window),
                                                        batch_size=NET.EXECUTION_BS)
            salida = []
            for (x_batch, y_batch) in seq_all:
                predicted = self.model.predict(x_batch, verbose=0)
                mse_batched = tf.sqrt(tf.reduce_mean(tf.square(y_batch - predicted), axis=1))
                salida.extend(mse_batched.numpy())
            rec.update(modo="modificado", familia=familia, factores=sorted(f), filas_train=n_tr, filas_val=n_val,
                       normalizacion_ajustada_con=("las %d filas que entrenan (90 %%)" % n_tr
                                                   if NORMALIZACION == "entrenamiento" else "el tramo entero"),
                       norm_min_sha256=huella(np.asarray(self.norm_min, dtype=np.float64)),
                       norm_max_sha256=huella(np.asarray(self.norm_max, dtype=np.float64)),
                       cola_sha256=huella(va), pesos_iniciales_sha256=info["pesos_iniciales_sha256"],
                       firma=info["firma"], params_original=info["params_original"],
                       identidad_factores_apagados=info["identidad_factores_apagados"],
                       lotes_en_fit=int(cuenta.lotes), train_counter=int(self.model._train_counter.numpy()),
                       loss_epoca=loss_epoca, segundos_fit=seg_fit, execute=info["execute"])
            if rec["lotes_en_fit"] != n_tr or rec["train_counter"] != n_tr:
                fallos.append("fit dio %d lotes y %d pasos; se esperaban %d (las %d filas de la cola no entrenan)"
                              % (rec["lotes_en_fit"], rec["train_counter"], n_tr, n_val))
            ceros = np.zeros((L, rec["n_visible"]))
            if (not np.array_equal(self.window, np.concatenate([ceros, x_all], axis=0)[-L:])
                    or not np.array_equal(ult, np.concatenate([ceros, xt], axis=0)[-L:])):
                fallos.append("la ventana de estado no son las ultimas L filas del tramo")
            if self.model is not info["model"]:
                fallos.append("el modelo cambio durante train")
            _, f_firma = firma_reparado(M, self.model, self.encoder, self.decoder, familia, f, rec["n_visible"], L,
                                        iniciales=False)
            fallos += f_firma
            if "P" in f:
                rec["lr_por_capa"], rec["grupos_adam"], f_p = regla_paso(self.model, self.encoder, self.decoder,
                                                                          rec["n_visible"], float(self.lr), n_tr)
                fallos += f_p
                rec["optimizer_de_adorno_iteraciones"] = int(self.model.optimizer.iterations.numpy())
                if rec["optimizer_de_adorno_iteraciones"] != 0:
                    fallos.append("el Adam de adorno ha dado %d pasos" % rec["optimizer_de_adorno_iteraciones"])
            else:
                rec["lr_por_capa"], rec["grupos_adam"] = None, None
                rec["optimizer_iterations"] = int(self.model.optimizer.iterations.numpy())
            rec["pesos_finales_sha256"] = huella_pesos(self.model)
            if rec["pesos_finales_sha256"] == rec["pesos_iniciales_sha256"]:
                fallos.append("los pesos no han cambiado tras fit")
            conv.update(filas_train=n_tr, loss_epoca=loss_epoca, lr_por_capa=rec["lr_por_capa"])
            rec["convergencia"] = conv
            self.rep_convergencia = conv
        if len(salida) != N:
            fallos.append("train devuelve %d errores para %d filas" % (len(salida), N))
        rec["segundos"] = time.perf_counter() - t0
        rec["assert_ok"], rec["fallos"] = not fallos, fallos
        fits.append(rec)
        estado = "ASSERT OK" if not fallos else "ASSERT FALLA: " + "; ".join(fallos)
        if control:
            log("ae%d %r n_v=%d filas=%d [observador]: pasos=%d | %.1f s | x_train %s | pesos iniciales %s | %s"
                % (rec["ae_index"], rec["ae_name"], rec["n_visible"], N, rec["train_counter"], rec["segundos"],
                   rec["x_train_sha256"][:12], rec["pesos_iniciales_sha256"][:12], estado))
        else:
            c = rec["convergencia"]

            def fmt(v, p="%.4f"):
                return "None" if v is None else p % v
            log("ae%d %r n_v=%d %s [%s] train=%d cola=%d: razon=%s suelo=%.3e evaluable=%s entrenado=%s | "
                "persistencia=%s | contexto=%s | muertas=%.2f std=%.3e | loss=%.5g pasos=%d | %.1f s | x_train %s | %s"
                % (rec["ae_index"], rec["ae_name"], rec["n_visible"], c["paradigma"], ",".join(sorted(f)),
                   rec["filas_train"], rec["filas_val"], fmt(c["razon"]), c["suelo"], c["evaluable"], c["entrenado"],
                   fmt(c["razon_persistencia"]), fmt(c["cociente_contexto_cambiado"]), c["muertas"], c["std_salida"],
                   rec["loss_epoca"], rec["train_counter"], rec["segundos"], rec["x_train_sha256"][:12], estado))
        if fallos:
            raise error_cls("T10-REP: el autoencoder %d no ha entrenado como se declara: %s"
                            % (rec["ae_index"], "; ".join(fallos)))
        return salida

    def execute(self, x):
        if not control:
            info = vivos.get(id(self))
            if info is None or self.model is not info["model"] or info["train"] != 1:
                raise error_cls("T10-REP: execute() with a model that is not the modified and trained one")
            info["execute"]["llamadas"] += 1
            info["execute"]["filas"] += int(len(x))
        return execute_original(self, x)

    __init__.__wrapped__, train.__wrapped__, execute.__wrapped__ = init_original, train_original, execute_original
    cls.__init__, cls.train, cls.execute = __init__, train, execute
    return {"objetivo": "%s.%s (__init__, train, execute)" % (FAMILIAS[familia]["modulo"], cls.__name__),
            "fichero": str(fuente_fichero), "sha256": sha, "lineas_congeladas": [FIT_LINE, EVAL_LINE],
            "modo": "control (solo observa: ni sustituye el modelo ni reserva filas)" if control
                    else "modified: factors %s" % ",".join(sorted(f)),
            "factores": None if control else sorted(f), "val_div": VAL_DIV, "normalizacion": NORMALIZACION,
            "modelos_rep": {"path": str(MODELOS_REP), "sha256": sha256_file(MODELOS_REP), "tag": TAG_ENMIENDA},
            "mecanismo": "in-memory wrapper around the frozen class; src/ and the runner unchanged"}


class Latido:
    """Sustituto minimo de common.Heartbeat: run_job solo le pide fase y fila. Anota las fases."""

    def __init__(self, log):
        self.phase, self.row, self.n = "inicio", 0, 0
        self._log, self._t = log, time.time()

    def set_phase(self, phase, n=None):
        t = time.time()
        self._log("fase %s -> %s (+%.1f s)" % (self.phase, phase, t - self._t))
        self.phase, self._t = phase, t
        if n is not None:
            self.n = n

    def beat(self):
        return None

    def stop(self, *_a, **_k):
        return None


def cargar_referencia(dataset, familia, ar):
    """Lee el job T1 de referencia y devuelve lo que el hijo necesita de el.

    :return: (ref_id, ref_dir, config, arrays, env).
    :raises SystemExit: si el job de referencia no esta DONE.
    """
    rid = ref_job_id(dataset, familia, ar)
    d = RESULTS / rid
    if not (d / "DONE").exists():
        raise SystemExit("el job T1 de referencia %s no tiene DONE" % rid)
    cfg = leer_json(d / "config.json")
    if not cfg:
        raise SystemExit("config.json ilegible en %s" % d)
    return rid, d, cfg, leer_json(d / "arrays.json", {}) or {}, leer_json(d / "env.json", {}) or {}


def comprobar_runner(C, RJ, code_root, runner_dir, ref_cfg, log):
    """El arbol del que se importa es, byte a byte, el que ejecuto el job T1 de referencia.

    :return: dict de comprobaciones para config.json.
    :raises RuntimeError: si algo no coincide.
    """
    ref_code = ref_cfg.get("code") or {}
    probs = []
    if Path(C.__file__).resolve().parent != runner_dir or Path(RJ.__file__).resolve().parent != runner_dir:
        probs.append("common/run_job importados de %s y %s, no de %s" % (C.__file__, RJ.__file__, runner_dir))
    if not C.is_registered_worktree(code_root):
        probs.append("%s no es un worktree registrado" % code_root)
    sucio = C.worktree_dirty(code_root)
    if sucio:
        probs.append("worktree sucio: %s" % " ".join(sucio.split())[:200])
    tag, head = C.runner_tag_at_head(code_root), C.head_commit(code_root)
    if tag != ref_code.get("runner_tag") or head != ref_code.get("head"):
        probs.append("HEAD %s (%s) != %s (%s) del job T1" % (str(head)[:10], tag, str(ref_code.get("head"))[:10],
                                                            ref_code.get("runner_tag")))
    hashes = {"runner": {}, "original": {}}
    for grupo, base in (("runner", runner_dir), ("original", code_root)):
        for f, h_ref in sorted(((ref_cfg.get("code_hashes") or {}).get(grupo) or {}).items()):
            if h_ref is None:
                continue
            p = base / f
            h = sha256_file(p) if p.exists() else None
            hashes[grupo][f] = {"sha256": h, "igual_que_T1": h == h_ref}
            if h != h_ref:
                probs.append("%s: sha256 %s != %s del job T1" % (f, str(h)[:12], h_ref[:12]))
    ok, detalle = C.src_identical(code_root)
    if not ok:
        probs.append("original code not identical to the submitted version: %s" % detalle)
    if probs:
        raise RuntimeError("el runner no es el del job T1 de referencia: " + "; ".join(probs))
    n = sum(len(v) for v in hashes.values())
    log("runner %s (%s) = el del job T1: %d ficheros con el mismo sha256; src %s" % (tag, str(head)[:10], n, detalle))
    return {"code_root": str(code_root), "runner_dir": str(runner_dir), "main_repo": str(C.MAIN_REPO), "head": head,
            "tags_at_head": C.tags_at_head(code_root), "worktree_clean": True, "registered_worktree": True,
            "runner_tag": tag, "src_identical_to_submitted_v1": detalle,
            "same_files_as_reference_job": hashes, "n_files_same_sha256": n}


def comprobar_cache(C, pipelines, spec, code_root):
    """La cache del stream existe (nunca se construye desde aqui: data/ no se toca) y es la misma.

    :return: dict con la clave, la huella del constructor y los sha256 de x e y.
    :raises RuntimeError: si falta o si la huella no coincide.
    """
    key = pipelines.cache_key(spec)
    final = Path(C.CACHE_ROOT) / key
    if not (final / "DONE").exists():
        raise RuntimeError("the cache %s does not exist; this script does not build it (data/ is never written)" % final)
    meta = C.read_json(final / "meta.json") or {}
    fp = pipelines.builder_fingerprint(pipelines.builder_for(spec, None), code_root)
    if (meta.get("builder") or {}).get("fingerprint_sha256") != fp:
        raise RuntimeError("cache %s: huella del constructor distinta; load_stream intentaria reconstruirla" % key)
    return {"key": key, "path": str(final), "builder_fingerprint": fp,
            "x_sha256": (meta.get("x") or {}).get("sha256"), "y_sha256": (meta.get("y") or {}).get("sha256")}


def envolver_regiones(RJ, ref_regions, C, log):
    """Hace que run_job.regions_for falle ANTES del bucle si las regiones no son las del job T1."""
    original = RJ.regions_for

    def regions_for(*args, **kwargs):
        reg = original(*args, **kwargs)
        plano = json.loads(json.dumps(reg, default=str))
        if plano != ref_regions:
            dif = sorted(k for k in set(plano) | set(ref_regions) if plano.get(k) != ref_regions.get(k))
            raise C.FrozenConfigError("T10-REP: regiones distintas de las del job T1 en %s" % dif)
        log("regiones identicas a las del job T1 de referencia (fm %s, entrenamiento de los AE %s, train %s)"
            % (reg.get("fm"), reg.get("ae_train"), reg.get("train")))
        return reg

    RJ.regions_for = regions_for


def dif_scores(a, b, fm, T):
    """Diferencia entre dos scores.npy de la misma forma. NaN en la misma posicion cuenta como igual.

    :return: dict con la igualdad, las filas distintas por tramo y las diferencias maxima y media.
    """
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    dif = ~((a == b) | (na & nb))
    ad = np.abs(a - b)
    fin = np.isfinite(ad)
    post = fin.copy()
    post[:fm] = False
    return {"identical": bool(not dif.any()), "rows_different": int(dif.sum()),
            "rows_different_in_fm": int(dif[:fm].sum()), "rows_different_in_ae_train": int(dif[fm:T].sum()),
            "rows_different_post_train": int(dif[T:].sum()),
            "max_abs_diff": float(ad[fin].max()) if fin.any() else 0.0,
            "mean_abs_diff_desde_fm": float(ad[post].mean()) if post.any() else 0.0,
            "nan_mismas_posiciones": bool(np.array_equal(na, nb)), "n_nan": int(na.sum()),
            "n_inf": int(np.isinf(a).sum())}


def tramo_desde_cache(x, fmaps, a, b, n_val, dtype):
    """Recalcula, desde la cache del stream y sin pasar por el modelo, lo que cada AE tuvo que recibir.

    :param x: matriz del stream (memmap).
    :param fmaps: grupos de features (fm_stats.json del propio job).
    :param int a, b: tramo de entrenamiento de los AE [a, b).
    :param int n_val: filas reservadas al final (0 en el control).
    :return: lista de dicts con el sha256 del tramo y el min/max por columna de las filas que entrenan.
    """
    import numpy as np

    seg_all = np.asarray(x[a:b])
    out = []
    for cols in fmaps:
        seg = np.ascontiguousarray(seg_all[:, [int(c) for c in cols]]).astype(dtype, copy=False)
        tr = seg[:len(seg) - n_val] if (n_val and NORMALIZACION == "entrenamiento") else seg
        out.append({"x_train_sha256": hashlib.sha256(np.ascontiguousarray(seg).tobytes()).hexdigest(),
                    "min": np.min(tr, axis=0), "max": np.max(tr, axis=0), "filas": int(len(seg))})
    return out


def comparar_con_referencia(tmp, ref_dir, ref_cfg, ref_arrays, cfg, fits, control, x_cache=None):
    """Compares the outputs with the T1 reference job; whatever does not depend on the model must be identical."""
    import csv
    import numpy as np

    out, malos = {}, []
    arrays = leer_json(tmp / "arrays.json", {}) or {}
    s, sr = np.load(tmp / "scores.npy", mmap_mode="r"), np.load(ref_dir / "scores.npy", mmap_mode="r")
    fm = int(ref_cfg["regions"]["fm"][1])
    a_tr, T = int(ref_cfg["regions"]["ae_train"][0]), int(ref_cfg["regions"]["ae_train"][1])
    k = int(ref_cfg["hyperparameters_frozen"]["k"])
    sc = {"shape": list(s.shape), "shape_ref": list(sr.shape), "dtype": str(s.dtype), "dtype_ref": str(sr.dtype),
          "sha256": (arrays.get("scores.npy") or {}).get("sha256"),
          "sha256_ref": (ref_arrays.get("scores.npy") or {}).get("sha256")}
    sc["same_shape_dtype"] = bool(s.shape == sr.shape and s.dtype == sr.dtype)
    if sc["same_shape_dtype"]:
        sc.update(dif_scores(s, sr, fm, T))
        if sc["rows_different_in_fm"]:
            malos.append("scores distintos en el tramo del Feature Map (%d filas)" % sc["rows_different_in_fm"])
    else:
        malos.append("scores.npy %s %s != %s %s" % (s.shape, s.dtype, sr.shape, sr.dtype))
    out["scores"] = sc
    for f in ("labels.npy", "ae_errors.npy"):
        mine, ref = arrays.get(f) or {}, ref_arrays.get(f) or {}
        out[f] = {"shape": mine.get("shape"), "shape_ref": ref.get("shape"), "sha256": mine.get("sha256"),
                  "sha256_ref": ref.get("sha256"), "same_shape": mine.get("shape") == ref.get("shape"),
                  "identical": bool(mine.get("sha256")) and mine.get("sha256") == ref.get("sha256")}
        if not out[f]["same_shape"]:
            malos.append("%s con forma %s != %s" % (f, mine.get("shape"), ref.get("shape")))
    if not out["labels.npy"]["identical"]:
        malos.append("labels.npy con sha256 distinto del job T1")
    if control and not out["ae_errors.npy"]["identical"]:
        try:
            e, er = np.load(tmp / "ae_errors.npy", mmap_mode="r"), np.load(ref_dir / "ae_errors.npy", mmap_mode="r")
            d = np.abs(np.asarray(e, dtype=np.float64) - np.asarray(er, dtype=np.float64))
            fin = np.isfinite(d)
            out["ae_errors.npy"].update(max_abs_diff=float(d[fin].max()) if fin.any() else 0.0,
                                        mean_abs_diff=float(d[fin].mean()) if fin.any() else 0.0)
            del e, er, d, fin
        except Exception as ex:
            out["ae_errors.npy"]["error"] = "%s: %s" % (type(ex).__name__, ex)
    fs, fsr = leer_json(tmp / "fm_stats.json", {}) or {}, leer_json(ref_dir / "fm_stats.json", {}) or {}
    out["feature_map"] = {"identical": fs.get("feature_map") is not None and fs.get("feature_map") == fsr.get("feature_map"),
                          "sizes": fs.get("sizes_final"), "sizes_ref": fsr.get("sizes_final"),
                          "dbscan_trajectory_identical": fs.get("trajectory") == fsr.get("trajectory"),
                          "labels_plus1_identical": fs.get("labels_plus1") == fsr.get("labels_plus1")}
    if not out["feature_map"]["identical"]:
        malos.append("mapa de features distinto del job T1")
    if not out["feature_map"]["dbscan_trajectory_identical"]:
        malos.append("trayectoria de DBSCAN distinta de la del job T1")
    stats = None
    try:
        m, mr = np.load(tmp / "model_stats.npz"), np.load(ref_dir / "model_stats.npz")
        stats = {kk: np.asarray(m[kk]) for kk in m.files}
        norm = {kk: bool(kk in mr.files and np.array_equal(m[kk], mr[kk], equal_nan=True))
                for kk in m.files if kk.endswith(("_norm_min", "_norm_max")) and kk.startswith("ae")}
        out["normalization"] = {"keys": sorted(norm), "all_identical": bool(norm) and all(norm.values()),
                                "same_keys": sorted(m.files) == sorted(mr.files),
                                "nota": None if control else "the modified model fits the min-max on the training "
                                                             "part only (first 90 percent): it need not match that "
                                                             "of the T1 job"}
        if control and not out["normalization"]["all_identical"]:
            malos.append("min/max de normalizacion de algun AE distinto del job T1: %s"
                         % sorted(kk for kk, v in norm.items() if not v))
        if len(norm) != 2 * k:
            malos.append("model_stats.npz con %d claves de normalizacion, se esperaban %d" % (len(norm), 2 * k))
    except Exception as e:
        out["normalization"] = {"error": "%s: %s" % (type(e).__name__, e)}
        malos.append("model_stats.npz no comparable: %s" % e)

    def leer_train_log(p):
        with open(p, newline="") as fh:
            return list(csv.DictReader(fh))
    try:
        tl, tlr = leer_train_log(tmp / "train_log.csv"), leer_train_log(ref_dir / "train_log.csv")
        cols = ("ae_index", "class", "n_visible", "n_hidden", "epochs") + (("train_steps",) if control else ())
        igual = len(tl) == len(tlr) and all(all(x1.get(c) == x2.get(c) for c in cols) for x1, x2 in zip(tl, tlr))
        pasos = [r.get("train_steps") for r in tl]
        out["train_log"] = {"same_structure": bool(igual), "columns_compared": list(cols),
                            "loss_epoch_1": [r.get("loss_epoch_1") for r in tl],
                            "loss_epoch_1_ref": [r.get("loss_epoch_1") for r in tlr],
                            "train_steps": pasos, "train_steps_ref": [r.get("train_steps") for r in tlr]}
        if not igual:
            malos.append("train_log.csv: clase, tamanos, epocas o pasos distintos del job T1")
        if pasos != [str(f_.get("filas_train")) for f_ in fits]:
            malos.append("train_log.csv: pasos %s != filas que entrenan %s" % (pasos, [f_.get("filas_train") for f_ in fits]))
    except Exception as e:
        out["train_log"] = {"error": "%s: %s" % (type(e).__name__, e)}
        malos.append("train_log.csv no comparable: %s" % e)
    for kk in ("hyperparameters_frozen", "kitnet_config_check", "feature_map_sizes", "regions"):
        igual = json.loads(json.dumps(cfg.get(kk), default=str)) == ref_cfg.get(kk)
        out[kk + "_identical"] = bool(igual)
        if not igual:
            malos.append("config.%s distinto del job T1" % kk)
    he, her = cfg.get("hyperparameters_effective") or {}, ref_cfg.get("hyperparameters_effective") or {}
    claves = ("kitnet", "__special__", "autoencoders", "output_layer")
    igual = json.loads(json.dumps({kk: he.get(kk) for kk in claves}, default=str)) == {kk: her.get(kk) for kk in claves}
    out["hyperparameters_effective_identical"] = bool(igual)
    out["hyperparameters_effective_note"] = ("soft comparison. In the control a mismatch goes to anomalies; with the "
                                             "modified model it cannot match (the model is different) and is not "
                                             "checked")
    if len(fits) != k or not all(f_.get("assert_ok") for f_ in fits):
        malos.append("se esperaban %d autoencoders con el assert en verde y hay %d" % (k, len(fits)))
    n_val = [int(f_.get("filas_val") or 0) for f_ in fits]
    if [int(f_.get("filas_train") or 0) + v for f_, v in zip(fits, n_val)] != [T - a_tr] * len(fits):
        malos.append("filas de entrenamiento mas cola != tramo de los AE del job T1 [%d, %d)" % (a_tr, T))
    if (not control) and n_val != [(T - a_tr) // VAL_DIV] * len(fits):
        malos.append("cola reservada %s != %d filas (10 %% del tramo)" % (n_val, (T - a_tr) // VAL_DIV))
    if x_cache is not None and fs.get("feature_map") and len(fits) == len(fs["feature_map"]):
        try:
            x = np.load(x_cache, mmap_mode="r")
            indep = tramo_desde_cache(x, fs["feature_map"], a_tr, T, n_val[0] if n_val else 0,
                                      fits[0].get("x_train_dtype") or "float64")
            filas = []
            for i, (f_, d) in enumerate(zip(fits, indep)):
                fila = {"ae_index": i, "x_train_igual_que_cache": f_.get("x_train_sha256") == d["x_train_sha256"]}
                if stats is not None:
                    fila["norm_min_igual_que_cache"] = bool(np.array_equal(stats.get("ae%d_norm_min" % i), d["min"]))
                    fila["norm_max_igual_que_cache"] = bool(np.array_equal(stats.get("ae%d_norm_max" % i), d["max"]))
                filas.append(fila)
                if not all(v for kk, v in fila.items() if kk != "ae_index"):
                    malos.append("ae%d: el tramo o el min-max no son los recalculados desde la cache: %s" % (i, fila))
            out["desde_cache"] = {"tramo": [a_tr, T], "filas_reservadas": n_val[0] if n_val else 0,
                                  "normalizacion": NORMALIZACION, "autoencoders": filas,
                                  "todo_igual": all(all(v for kk, v in fl.items() if kk != "ae_index") for fl in filas)}
        except Exception as e:
            out["desde_cache"] = {"error": "%s: %s" % (type(e).__name__, e)}
            malos.append("no se pudo recalcular el tramo desde la cache: %s" % e)
    else:
        malos.append("sin cache o sin mapa de features para recalcular el tramo de cada AE")
    if control:
        out["control_identico"] = bool(sc.get("identical") and out["ae_errors.npy"]["identical"])
    return out, malos


def informe_convergencia(jid, rid, row, familia, factores, fits, fmaps):
    """El convergence_report.json del job: una entrada por autoencoder, SIN etiquetas (enmienda 4).

    :return: dict listo para escribir.
    """
    aes = []
    for i, f_ in enumerate(fits):
        c = dict(f_.get("convergencia") or {})
        aes.append(dict({"ae_index": i, "class": f_.get("clase"), "n_visible": f_.get("n_visible"),
                         "n_hidden": f_.get("n_hidden"), "features": (fmaps[i] if fmaps and i < len(fmaps) else None),
                         "familia": familia, "factores": factores, "filas_train": f_.get("filas_train"),
                         "train_counter": f_.get("train_counter"),
                         "normalizacion_ajustada_con": f_.get("normalizacion_ajustada_con"),
                         "grupos_adam": f_.get("grupos_adam")}, **c))
    ev = [a_ for a_ in aes if a_.get("evaluable")]
    return {"schema": "jisa-t10-rep-convergencia/1", "job_id": jid, "reference_job": rid, "block": BLOCK,
            "dataset": row.get("dataset"), "stream": slug(row.get("dataset") or ""), "seed": row.get("seed"),
            "rep": row.get("rep"), "familia": familia, "variant": familia + "_rep", "factores": factores,
            "ar": str(row.get("ar")) == "True", "paradigma": "AR" if str(row.get("ar")) == "True" else "TSR",
            "decision": "amendment to the analysis plan, modified encoders, seed A", "tag": "[ENMIENDA-1] [EXPLORATORIO]",
            "criterios": {"razon": "MSE de validacion con los pesos finales / MSE del mejor predictor constante "
                                   "sobre la misma cola (enmienda 3.1)", "umbral_entrenado": UMBRAL,
                          "suelo_minimo_evaluable": SUELO_MIN, "cociente_contexto_minimo": CTX_MIN,
                          "cola": "last 10 %% of the training segment of the AE (rows // %d)" % VAL_DIV,
                          "normalizacion": NORMALIZACION, "usa_etiquetas": False},
            "summary": {"n_aes": len(aes), "n_evaluables": len(ev),
                        "n_entrenados": sum(1 for a_ in ev if a_.get("entrenado")),
                        "n_dependen_contexto": sum(1 for a_ in ev if a_.get("entrenado") and a_.get("depende_contexto"))},
            "autoencoders": aes}


def hijo(a):
    """Ejecuta un job (o su control, o solo su preflight). Devuelve el codigo de salida del runner."""
    t_start = time.time()
    dataset, familia, ar, control = resolver_dataset(a.dataset), a.familia, a.ar, bool(a.control)
    replica = int(a.replica)
    jid = t10_job_id(dataset, familia, ar, control, replica)
    log = hacer_log(jid)
    rid, ref_dir, ref_cfg, ref_arrays, ref_env = cargar_referencia(dataset, familia, ar)
    code_root = Path(ref_cfg["code"]["code_root"]).resolve()
    runner_dir = code_root / "revision" / "experiments"
    if not (runner_dir / "run_job.py").exists():
        log("ENV: no existe el worktree del runner del job T1: %s" % runner_dir)
        return 8
    aqui = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in (aqui, str(R / "revision" / "experiments"))]
    sys.path.insert(0, str(runner_dir))
    import common as C
    import run_job as RJ

    ensayo = a.gpu == "cpu"
    results_root = Path(a.results_root).resolve()
    real = results_root == RESULTS.resolve()
    factores, fase_a = norm_factores(a.factores), None
    try:
        faltan = [n for n in NEEDED_COMMON if not hasattr(C, n)]
        params = list(inspect.signature(RJ.run_real).parameters)
        if faltan or params != RUN_REAL_PARAMS or not hasattr(RJ, "regions_for"):
            raise RuntimeError("API del runner inesperada: faltan %s; run_real%s" % (faltan, params))
        if slug(dataset) != C.ds_slug(dataset) or C.SEEDS.get(SEED) != SEEDS[SEED]:
            raise RuntimeError("slug o semilla distintos de los del runner")
        if os.environ.get("LD_LIBRARY_PATH"):
            raise RuntimeError("LD_LIBRARY_PATH no esta vacio; lanza con LD_LIBRARY_PATH=")
        if replica != 1 and not control:
            raise RuntimeError("--replica solo existe para el control del arnes")
        if Path(a.registro).resolve() != REGISTRO.resolve() and real and not a.preflight:
            raise RuntimeError("--registro distinto del oficial solo en ensayos fuera de %s" % RESULTS)
        if a.sin_registro:
            if not a.preflight:
                raise RuntimeError("--sin-registro solo con --preflight")
        else:
            fac_reg, fase_a = arreglo_elegido(familia, a.registro)
            if factores != fac_reg:
                raise RuntimeError("--factores %r != %r, que es lo que eligio la fase A para %s"
                                   % (factores, fac_reg, familia))
        if not control and not factores and not a.sin_registro:
            raise RuntimeError("the factors of the modification are missing")
        motivo = puerta_git(con_brazo=False)
        if motivo:
            raise RuntimeError("modelos_rep.py no es el fijado por la enmienda: %s" % motivo)
        puerta_brazo = puerta_git() or "OK"
        if puerta_brazo != "OK" and not ensayo and not a.preflight:
            raise RuntimeError("el arnes no es el fijado por %s: %s" % (TAG_BRAZO, puerta_brazo))
        code_checks = comprobar_runner(C, RJ, code_root, runner_dir, ref_cfg, log)
        row = dict(ref_cfg["manifest_row"])
        spec = C.validate_row(row, test_mode=False)
        C.check_supported(spec)
        if not (spec.job_id == rid and spec.variant == familia and spec.ar is (ar == "T")
                and spec.pipeline == "corrected" and spec.seed == SEED and spec.dataset == dataset
                and spec.window == 800 and spec.k == 4 and spec.grouping == "dbscan"
                and str(row.get("contamination", "0")) == "0" and not getattr(spec, "contamination_pct", 0)
                and spec.mode == "full"):
            raise RuntimeError("la fila de referencia no es la celda %s ar%s esperada: %s" % (familia, ar, spec.public()))
        if os.environ.get("PYTHONHASHSEED") != str(spec.seed_int):
            raise RuntimeError("PYTHONHASHSEED=%r, se esperaba %r (lo fija el proceso padre)"
                               % (os.environ.get("PYTHONHASHSEED"), str(spec.seed_int)))
        import pipelines
        cache_check = comprobar_cache(C, pipelines, spec, code_root)
        if (ref_cfg.get("cache") or {}).get("x", {}).get("sha256") != cache_check["x_sha256"]:
            raise RuntimeError("la cache del stream no es la que uso el job T1 (sha256 de x distinto)")
        env_ref = ref_env.get("env_vars") or {}
        env_dif = {k: (os.environ.get(k), v) for k, v in env_ref.items() if os.environ.get(k) != v}
        if ensayo or a.preflight:
            env_dif.pop("CUDA_VISIBLE_DEVICES", None)
        if env_dif:
            raise RuntimeError("variables de entorno distintas de las del job T1 (actual, T1): %s" % env_dif)
        if ensayo and not a.preflight and C.is_under(results_root, C.RESULTS_ROOT):
            raise RuntimeError("--gpu cpu es solo para ensayos: no puede escribir bajo %s" % C.RESULTS_ROOT)
        if not ensayo and results_root != Path(C.RESULTS_ROOT).resolve():
            raise RuntimeError("en GPU el destino es %s" % C.RESULTS_ROOT)
    except SystemExit:
        raise
    except Exception as e:
        log("ENV/INVALID antes de empezar: %s: %s" % (type(e).__name__, e))
        return C.ENV

    P = C.JobPaths(results_root, jid)
    if P.done.exists() and not a.preflight:
        log("DONE ya existe: nada que hacer (idempotente)")
        return C.OK
    if P.final.exists() and not P.done.exists():
        log("ANOMALIA: %s existe sin DONE; requiere persona (no se toca)" % P.final)
        return C.ANOMALY

    fits = []
    con_modelo = control or bool(factores)
    try:
        RJ.setup_tf_env(a.gpu)
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        t = time.time()
        import tensorflow as tf
        tf_import_s = time.time() - t
        NET = importlib.import_module(FAMILIAS[familia]["modulo"])
        if not C.is_under(NET.__file__, code_root):
            raise C.EnvError("src.models importado desde %s, no desde %s" % (NET.__file__, code_root))
        M = cargar_modelos_rep()
        sha_net = ((ref_cfg.get("code_hashes") or {}).get("original") or {}).get(FAMILIAS[familia]["fichero"])
        if not sha_net:
            raise C.EnvError("el job T1 no anoto el sha256 de %s" % FAMILIAS[familia]["fichero"])
        parche = None
        if con_modelo:
            parche = instalar_envoltorio(NET, M, familia, factores, control, log, fits, sha_net, C.HaltError)
        versiones = {"tensorflow": tf.__version__, "keras": __import__("keras").__version__}
        ver_ref = {k: (ref_env.get("versions") or {}).get(k) for k in versiones}
        if versiones != ver_ref:
            raise C.EnvError("versiones %s != %s del job T1" % (versiones, ver_ref))
    except Exception as e:
        log("ENV al preparar TF y el envoltorio: %s: %s" % (type(e).__name__, e))
        return C.ENV
    if parche:
        log("envoltorio instalado en %s: %s" % (parche["objetivo"], parche["modo"]))

    if a.preflight:
        try:
            import instrument as I
            cfgm = I.frozen_config(spec, {}, False)
            ok_ign, det_ign = C.ckpt_paths_ignored(C.RESULTS_ROOT, jid)
            if ok_ign is False:
                raise RuntimeError(det_ign)
            mini = "no modification to build"
            if parche and not control:
                cls = getattr(NET, FAMILIAS[familia]["clase"])
                ae = cls(n_visible=8, hidden_ratio=cfgm["hidden_ratio"], lr=cfgm["lr"], sequence_length=cfgm["L"],
                         ar=spec.ar, haste=False, name="preflight")
                olr = float(ae.model.optimizer.learning_rate.numpy())
                if type(ae).__name__ != I.AE_CLASSES[spec.variant] or abs(olr - cfgm["lr"]) > 1e-9:
                    raise RuntimeError("the modified AE would not pass check_aes: class %s, lr %r" % (type(ae).__name__, olr))
                mini = "minimal modified AE (n_v=8) built: %s, %d parameters" % (type(ae.model).__name__,
                                                                                     ae.model.count_params())
        except Exception as e:
            log("PREFLIGHT FALLA: %s: %s" % (type(e).__name__, e))
            return C.ENV
        log("PREFLIGHT OK: referencia %s; runner %s; cache %s (x %s); fm %d + ad %d, L %d, k %d, lr %g; fase A: %s; "
            "%s; %s" % (rid, code_checks["runner_tag"], cache_check["key"], str(cache_check["x_sha256"])[:12],
                        cfgm["fm"], cfgm["ad"], cfgm["L"], cfgm["k"], cfgm["lr"],
                        "SIN COMPROBAR (--sin-registro)" if fase_a is None
                        else "escalon %d [%s], control negativo completo no_pasa" % (fase_a["escalon"], factores),
                        mini, "DONE ya existe" if P.done.exists() else "por ejecutar"))
        return C.OK

    if C.read_fail_count(results_root, jid) >= C.MAX_FAILS:
        log("BLOCKED: fail_count >= %d (ver %s)" % (C.MAX_FAILS, P.state))
        return C.BLOCKED
    C.check_not_forbidden(results_root, "results_root")
    ok_ign, det_ign = C.ckpt_paths_ignored(results_root, jid)
    if ok_ign is False:
        log("ENV: ckpt_gitignore: %s" % det_ign)
        return C.ENV
    code_checks["ckpt_gitignore"] = det_ign

    P.tmp_root.mkdir(parents=True, exist_ok=True)
    lock_path = P.tmp_root / (jid + ".t10lock")
    lock_fh = open(lock_path, "a+")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("LOCKED: otro proceso esta ejecutando este job (%s)" % lock_path)
        lock_fh.close()
        return C.LOCKED
    host, pid = C.hostname(), os.getpid()
    tmp, hb = None, None

    def al_recibir_senal(signum, _frame):
        raise C.Interrupted("senal %s" % signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, al_recibir_senal)
    signal.signal(signal.SIGINT, al_recibir_senal)
    try:
        if P.done.exists():
            log("DONE aparecio mientras se esperaba: nada que hacer")
            return C.OK
        adopted = C.adopt_stale_tmps(P, host, pid, log)
        tmp, reuse = P.new_tmp(host, pid)
        if reuse:
            adopted.append(reuse)
        hb = Latido(log)
        que = ("control del arnes en modo observador (replica %d): ni sustituye el modelo ni reserva filas" % replica
               if control else "modification %s chosen by the selection runs, with the last 10 %% of the segment held out" % factores)
        row_t10 = dict(row, job_id=jid, block=BLOCK_CTRL if control else BLOCK,
                       variant=familia if control else familia + "_rep", rep=str(replica), priority="9100",
                       timing_ref="0",
                       notes=("[ENMIENDA-1] [EXPLORATORIO]: %s; reference job %s; outside "
                              "manifest.csv, S*_d, H1-H4 and the FINAL aggregation" % (que, rid)))
        harness = Path(__file__).resolve()
        cfg = {"job_id": jid, "manifest_row": row_t10, "manifest_path": None,
               "manifest_note": "synthetic row: block %s is deliberately not in manifest.csv; the "
                                "row that the runner executed is reference.manifest_row" % row_t10["block"],
               "spec": spec.public(),
               "spec_note": "es la fila T1 de referencia tal cual: run_job.run_real ejecuto exactamente esa celda",
               "mode": spec.mode, "test_mode": False, "test_overrides": {}, "max_rows": None,
               "seed_label": spec.seed, "seed_int": spec.seed_int, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
               "gpu_arg": a.gpu, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "code": code_checks, "results_root": str(results_root), "nan_is_result": True,
               "started": C.now_iso(), "host": host, "pid": pid,
               "reference": {"job_id": rid, "dir": str(ref_dir), "manifest_row": row,
                             "config_sha256": sha256_file(ref_dir / "config.json"),
                             "scores_sha256": (ref_arrays.get("scores.npy") or {}).get("sha256"),
                             "runner_tag": (ref_cfg.get("code") or {}).get("runner_tag"),
                             "date": ref_env.get("date"), "nvidia_smi": ref_env.get("nvidia_smi")},
               "t10_rep": {"decision": "D-59", "enmienda": TAG_ENMIENDA, "tag": "[ENMIENDA-1] [EXPLORATORIO]",
                           "control": control, "replica": replica, "ensayo_cpu": ensayo, "familia": familia,
                           "factores": None if control else factores, "fase_a": fase_a,
                           "factor": ("ninguno: control del arnes en modo observador" if control else
                                      "encoder, decoder y model de modelos_rep.construir(%r, n_v, 800, {%s}); ultimo "
                                      "10 %% del tramo de cada AE reservado; todo lo demas es el job T1 de referencia"
                                      % (familia, factores)),
                           "declaraciones": {
                               "normalizacion": "min-max de cada AE ajustado con %s (enmienda 3.1)"
                                                % ("el 90 % que entrena" if NORMALIZACION == "entrenamiento"
                                                   else "el tramo entero"),
                               "optimizer_de_adorno": "con el factor P, model.optimizer es un Adam(0,002) que no da "
                                                      "ningun paso (lo exige check_aes); hyperparameters_effective."
                                                      "autoencoders[*].optimizer y count_params, y "
                                                      "train_log.csv:optimizer_iterations, NO describen el "
                                                      "entrenamiento real: ver t10_rep.fits[*]",
                               "orden_rng": "per AE: original model, model with the factors switched off, modified model",
                               "salida_de_train": "RMSE por fila del tramo entero con los pesos finales y relleno de "
                                                  "ceros (como T8); las filas de la cola puntuan fuera de muestra",
                               "contexto_cambiado_en_AR": "descriptivo: la enmienda 3.8 solo lo define en TSR"},
                           "scope": "fuera del plan congelado, de S*_d, de H1-H4 y del agregado FINAL",
                           "harness": {"path": str(harness), "sha256": sha256_file(harness),
                                       "main_repo_head": C.head_commit(C.MAIN_REPO), "tag": TAG_BRAZO,
                                       "puerta_git": puerta_brazo},
                           "envoltorio": parche, "cache_check": cache_check}}
        timing = {"wall_seconds": None, "timing_ref": 0, "host": host, "gpu": a.gpu, "test_mode": False,
                  "phases": {"lock_to_start_s": time.time() - t_start, "tf_import_before_run_real_s": tf_import_s},
                  "concurrent_jobs_same_host": {"start": len(C.live_locks_on_host(results_root, host, jid))},
                  "gpu_compute_apps": {"start": C.gpu_compute_apps() if a.gpu != "cpu" else None},
                  "nice": os.nice(0)}
        envolver_regiones(RJ, ref_cfg["regions"], C, log)
        ns = argparse.Namespace(gpu=a.gpu, test_mode=False, max_rows=None, cache_root=str(C.CACHE_ROOT),
                                data_root=str(C.DATA_ROOT), cache_wait_s=3600.0, manifest=str(C.MANIFEST))
        log("empieza run_job.run_real con la fila de %s (%s)" % (rid, "CONTROL r%d" % replica if control
                                                                 else "modification " + factores))
        res = RJ.run_real(spec, ns, {}, code_root, results_root, tmp, hb, cfg, timing, log)

        anomalies = list(res.get("anomalies", []))
        cfg.update(res.get("config", {}))
        cfg["previous_attempts"] = {"adopted_now": adopted, "in_state": C.list_previous_attempts(P)}
        timing["phases"].update(res.get("phases", {}))
        for k, v in res.get("timing", {}).items():
            timing[k] = v
        timing["concurrent_jobs_same_host"]["end"] = len(C.live_locks_on_host(results_root, host, jid))
        timing["gpu_compute_apps"]["end"] = C.gpu_compute_apps() if a.gpu != "cpu" else None
        timing["rss_peak_bytes"] = RJ._rss_peak()
        present = {p.name for p in tmp.iterdir()}
        for f in RJ.NON_ESSENTIAL_FULL:
            if f not in present:
                anomalies.append("falta la salida no esencial %s" % f)
        if spec.save_ckpt and "ckpt" not in present:
            anomalies.append("save_ckpt=1 pero no hay ckpt/")
        notas = (cfg.get("hyperparameters_effective") or {}).get("notes")
        if isinstance(notas, dict) and not control:
            notas["t10_rep"] = ("T10-REP: en ESTE job los autoencoders son los de modelos_rep.construir con los factores "
                                "%s; con el factor P el bloque optimizer de cada AE es un Adam de adorno. El "
                                "entrenamiento real esta en t10_rep.fits y en convergence_report.json" % factores)
        vs_ref, malos = comparar_con_referencia(tmp, ref_dir, ref_cfg, ref_arrays, cfg, fits, control,
                                                Path(cache_check["path"]) / "x.npy")
        t10 = cfg["t10_rep"]
        t10["fits"] = fits
        t10["all_fits_assert_ok"] = bool(fits) and all(f.get("assert_ok") for f in fits)
        t10["vs_reference"] = vs_ref
        t10["requirements_failed"] = malos
        extra_out = {"rep_check.json"}
        C.write_json_atomic(tmp / "rep_check.json", {
            "job_id": jid, "reference_job": rid, "control": control, "replica": replica, "envoltorio": parche,
            "fits": fits, "all_fits_assert_ok": t10["all_fits_assert_ok"], "desde_cache": vs_ref.get("desde_cache")})
        if not control:
            fmaps = (leer_json(tmp / "fm_stats.json", {}) or {}).get("feature_map")
            C.write_json_atomic(tmp / "convergence_report.json",
                                informe_convergencia(jid, rid, row_t10, familia, factores, fits, fmaps))
            extra_out.add("convergence_report.json")
        if malos:
            raise C.HaltError("T10-REP: el job no es comparable con el T1 de referencia: " + "; ".join(malos))
        sc = vs_ref["scores"]
        log("frente al job T1: formas %s = %s; mapa de features identico %s; labels identicas %s; tramo y min-max "
            "de cada AE iguales a los recalculados desde la cache %s; scores identicos %s (%d filas distintas, "
            "|dif| max %.6g, media desde fm %.6g)"
            % (sc["shape"], sc["shape_ref"], vs_ref["feature_map"]["identical"], vs_ref["labels.npy"]["identical"],
               (vs_ref.get("desde_cache") or {}).get("todo_igual"), sc["identical"], sc["rows_different"],
               sc["max_abs_diff"], sc["mean_abs_diff_desde_fm"]))
        if control:
            t10["control_identico"] = vs_ref["control_identico"]
            t10["control_deterministas_ok"] = True
            otra = P.root / t10_job_id(dataset, familia, ar, True, 1)
            if replica != 1 and (otra / "DONE").exists():
                import numpy as np
                s1 = np.load(otra / "scores.npy", mmap_mode="r")
                s2 = np.load(tmp / "scores.npy", mmap_mode="r")
                t10["vs_replica_1"] = dict(dif_scores(s2, s1, int(ref_cfg["regions"]["fm"][1]),
                                                      int(ref_cfg["regions"]["ae_train"][1])), job_id=otra.name)
            log("CONTROL r%d: todo lo determinista es IDENTICO al job T1; scores.npy %s (sha256 %s frente a %s)"
                % (replica, "IDENTICO" if vs_ref["control_identico"] else
                   "NO identico: decide la regla de las dos replicas (K = %g)" % K_RUIDO,
                   str(sc["sha256"])[:16], str(sc["sha256_ref"])[:16]))
            if not vs_ref["hyperparameters_effective_identical"]:
                anomalies.append("T10-REP-CTRL: hyperparameters_effective no coincide literalmente con el del job T1 "
                                 "(comparacion blanda; revisar nombres de capa)")
        elif sc["identical"]:
            anomalies.append("T10-REP: the scores are IDENTICAL to those of the unmodified job; the modification had no effect")
        cfg["anomalies"] = anomalies
        cfg["outputs"] = sorted(present | {"config.json", "timing.json"} | extra_out |
                                ({"anomalies.json"} if anomalies else set()))
        timing["wall_seconds"] = time.time() - t_start
        C.write_json_atomic(tmp / "config.json", cfg)
        C.write_json_atomic(tmp / "timing.json", timing)
        if anomalies:
            C.write_json_atomic(tmp / "anomalies.json", {"job_id": jid, "anomalies": anomalies})
        present = {p.name for p in tmp.iterdir()}
        missing = [f for f in RJ.ESSENTIAL["full"] if f not in present]
        if missing:
            raise RuntimeError("faltan salidas esenciales: %s" % missing)
        hb.set_phase("commit")
        rc = C.commit(tmp, P)
        ph = timing["phases"]
        log("DONE en %.1f s (fm %.1f, train %.1f, exec %.1f, ckpt %.1f)%s -> %s"
            % (time.time() - t_start, ph.get("fm_s") or 0, ph.get("train_s") or 0, ph.get("exec_s") or 0,
               ph.get("ckpt_s") or 0, " con %d anomalias" % len(anomalies) if anomalies else "", P.final))
        return rc
    except C.Interrupted as e:
        _conservar(C, P, tmp, "interrumpido", log)
        log("INTERRUPTED: %s" % e)
        return C.INTERRUPTED
    except BaseException as e:
        tb = traceback.format_exc()
        try:
            n = C.record_failure(results_root, jid, "fase=%s fila=%s\n%s" % (getattr(hb, "phase", "?"),
                                                                            getattr(hb, "row", "?"), tb[-8000:]))
        except Exception:
            n = "?"
        _conservar(C, P, tmp, "fallo %s" % n, log)
        halt = isinstance(e, C.HaltError)
        log("%s (error_%s.txt): %s: %s" % ("HALT" if halt else "FAILED", n, type(e).__name__, str(e)[:800]))
        print(tb[-4000:], flush=True)
        return C.HALT if halt else C.FAILED
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
            lock_fh.close()
            lock_path.unlink()
        except OSError:
            pass


def _conservar(C, P, tmp, motivo, log):
    """Moves an attempt that has scores.npy to .state/<job>/attempt_k instead of deleting it."""
    if tmp is None:
        return
    try:
        dest = C.preserve_tmp(P, tmp, motivo)
        if dest:
            log("intento con scores.npy conservado en %s" % dest)
    except Exception as e:
        log("aviso: no se pudo conservar %s: %s" % (tmp, e))


ESTADO = {"hijo": None, "parar": False}


def _parar(signum, _frame):
    ESTADO["parar"] = True
    h = ESTADO["hijo"]
    if h is not None and h.poll() is None:
        try:
            h.send_signal(signal.SIGTERM)
        except OSError:
            pass


def entorno_hijo(gpu):
    """El entorno que worker.child_env daba a cada job de la rejilla (mas la semilla del hash)."""
    env = dict(os.environ)
    env.update(CHILD_ENV)
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else str(gpu)
    env["PYTHONHASHSEED"] = str(SEEDS[SEED])
    for k in THREAD_VARS + ("JISA_SEED_REEXEC",):
        env.pop(k, None)
    return env


def ref_de(dataset, familia, ar):
    """Lo que el padre necesita del job T1 de referencia: tag del runner, reloj y pico de RAM."""
    d = RESULTS / ref_job_id(dataset, familia, ar)
    cfg, tim = leer_json(d / "config.json", {}) or {}, leer_json(d / "timing.json", {}) or {}
    return {"done": (d / "DONE").exists(), "runner_tag": (cfg.get("code") or {}).get("runner_tag"),
            "runner_dir": (cfg.get("code") or {}).get("runner_dir"), "wall": float(tim.get("wall_seconds") or 0.0),
            "rss": int(tim.get("rss_peak_bytes") or 0)}


def lanzar_hijo(dataset, ar, a, control=False, replica=1, preflight=False):
    """Lanza un hijo, guarda su salida completa en logs/jobs/ y retransmite sus lineas propias.

    :return: (codigo de salida, ruta del log del hijo, segundos).
    """
    jid = t10_job_id(dataset, a.familia, ar, control, replica)
    cwd = Path(ref_de(dataset, a.familia, ar)["runner_dir"] or R)
    gpu = "cpu" if preflight else a.gpu
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--familia", a.familia, "--dataset", dataset,
           "--ar", ar, "--factores", norm_factores(a.factores), "--gpu", gpu, "--results-root", str(a.results_root),
           "--registro", str(a.registro), "--replica", str(replica)]
    cmd += ["--control"] if control else []
    cmd += ["--preflight"] if preflight else []
    cmd += ["--sin-registro"] if a.sin_registro else []
    (LOGS / "jobs").mkdir(parents=True, exist_ok=True)
    jl = LOGS / "jobs" / ("%s%s.%s.%d.log" % (jid, ".preflight" if preflight else "",
                                              time.strftime("%Y%m%dT%H%M%S"), os.getpid()))
    uid = None
    if not preflight:
        uid = "%d-%s" % (os.getpid(), time.strftime("%Y%m%dT%H%M%S"))
        reloj_anota(a.results_root, a.familia, {"tipo": "inicio", "uid": uid, "job_id": jid, "t0": time.time(),
                                                "log": str(jl), "control": control})
    t0 = time.time()
    pos = 0
    with open(jl, "ab") as fh:
        p = subprocess.Popen(cmd, env=entorno_hijo(gpu), stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(cwd if cwd.exists() else R))
        ESTADO["hijo"] = p
        while True:
            rc = p.poll()
            with open(jl, "rb") as rd:
                rd.seek(pos)
                trozo = rd.read()
            corte = max(trozo.rfind(b"\n"), trozo.rfind(b"\r")) + 1 if rc is None else len(trozo)
            for ln in lineas_marcadas(trozo[:corte].decode("utf-8", "replace")):
                print(ln, flush=True)
            pos += corte
            if rc is not None:
                break
            time.sleep(2.0)
        ESTADO["hijo"] = None
    seg = time.time() - t0
    if uid:
        reloj_anota(a.results_root, a.familia, {"tipo": "fin", "uid": uid, "job_id": jid, "rc": rc, "segundos": seg,
                                                "control": control})
    if rc != 0:
        cola = re.split(r"[\r\n]+", open(jl, "rb").read()[-6000:].decode("utf-8", "replace"))
        print("%s padre] %s salio con codigo %d; ultimas lineas de %s:" % (MARK, jid, rc, jl), flush=True)
        ruido = ("INFO:tensorflow", "WARNING:tensorflow", "INFO:src.models")
        for ln in [c for c in cola if c.strip() and MARK not in c and not c.startswith(ruido)][-15:]:
            print("    | " + ln[:300], flush=True)
    return rc, jl, seg


def reloj_path(results_root, familia):
    return dir_control(results_root) / ("reloj.%s.jsonl" % familia)


def reloj_lee(results_root, familia):
    p = reloj_path(results_root, familia)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def reloj_anota(results_root, familia, rec):
    p = reloj_path(results_root, familia)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(rec, ts=ahora()), ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def reloj_gastado(results_root, familia):
    """Segundos de reloj cargados a la familia: cada hijo terminado (jobs y controles, acaben como acaben) y,
    de un hijo que empezo y no dejo cierre (padre muerto), el tiempo hasta la ultima escritura de su log."""
    rs = reloj_lee(results_root, familia)
    cerrados = {r.get("uid") for r in rs if r.get("tipo") == "fin"}
    tot = 0.0
    for r in rs:
        if r.get("tipo") == "fin":
            tot += float(r.get("segundos") or 0.0)
        elif r.get("tipo") == "inicio" and r.get("uid") not in cerrados:
            try:
                tot += max(0.0, os.path.getmtime(r["log"]) - float(r["t0"]))
            except (OSError, KeyError, TypeError, ValueError):
                pass
    return tot


def factor_duracion(results_root, familia):
    """Cuanto mas tarda un job T10 que su job T1: el mayor cociente visto en la familia (prevision prudente)."""
    cocientes = []
    for r in reloj_lee(results_root, familia):
        if r.get("tipo") == "cociente" and r.get("rc") == 0 and not r.get("control") and r.get("ref_wall"):
            cocientes.append(float(r["segundos"]) / float(r["ref_wall"]))
    return max(cocientes) if cocientes else FACTOR_INICIAL


def _meminfo():
    out = {}
    with open("/proc/meminfo") as fh:
        for ln in fh:
            k, v = ln.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    return out


def _vivo(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def ram_reservar(results_root, jid, bytes_):
    """Reserva `bytes_` para el job si caben junto a las reservas vivas de los otros padres.

    :return: (True, texto) si se reservo; (False, motivo) si ahora no cabe.
    """
    d = dir_control(results_root)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "ram.lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            vivas = [r for r in (leer_json(d / "ram.json", []) or []) if _vivo(r.get("pid")) and r.get("pid") != os.getpid()]
            mem = _meminfo()
            otras = sum(int(r.get("bytes") or 0) for r in vivas)
            tope = mem["MemTotal"] - RAM_SISTEMA
            if otras + bytes_ > tope:
                return False, "reservado por otros %.1f GB + %.1f GB > %.1f GB" % (otras / 2**30, bytes_ / 2**30,
                                                                                tope / 2**30)
            if not vivas and bytes_ > mem["MemAvailable"]:
                return False, "disponible %.1f GB < %.1f GB" % (mem["MemAvailable"] / 2**30, bytes_ / 2**30)
            vivas.append({"pid": os.getpid(), "job": jid, "bytes": int(bytes_), "ts": ahora()})
            tmp = d / "ram.json.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(vivas, fh)
            os.replace(tmp, d / "ram.json")
            return True, "%.1f GB reservados (otros: %.1f GB)" % (bytes_ / 2**30, otras / 2**30)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def ram_liberar(results_root):
    d = dir_control(results_root)
    try:
        with open(d / "ram.lock", "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            vivas = [r for r in (leer_json(d / "ram.json", []) or []) if _vivo(r.get("pid")) and r.get("pid") != os.getpid()]
            tmp = d / "ram.json.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(vivas, fh)
            os.replace(tmp, d / "ram.json")
            fcntl.flock(lk, fcntl.LOCK_UN)
    except OSError:
        pass


def lanzar_con_ram(dataset, ar, a, quien, control=False, replica=1):
    """Lanza un hijo cuando su RAM cabe. La espera no cuenta en el reloj de la familia.

    :return: (codigo, log, segundos) o (None, motivo, 0.0) si se agoto la espera o se pidio parar.
    """
    ref = ref_de(dataset, a.familia, ar)
    need = int((ref["rss"] or RAM_DEFECTO) * RAM_FACTOR)
    jid = t10_job_id(dataset, a.familia, ar, control, replica)
    t0, aviso = time.time(), 0.0
    while True:
        if ESTADO["parar"]:
            return None, "interrumpido", 0.0
        ok, txt = ram_reservar(a.results_root, jid, need)
        if ok:
            break
        if time.time() - t0 > RAM_ESPERA_MAX_S:
            return None, "sin RAM tras %.0f s: %s" % (time.time() - t0, txt), 0.0
        if time.time() - aviso > 600:
            print("%s %s espero RAM para %s: %s" % (ahora(), quien, jid, txt), flush=True)
            aviso = time.time()
        time.sleep(30)
    try:
        return lanzar_hijo(dataset, ar, a, control=control, replica=replica)
    finally:
        ram_liberar(a.results_root)


def veredicto_control(results_root, familia, tag):
    """Lo que dicen los controles DONE de (familia, version del runner). Solo lee sus config.json.

    :return: (estado, detalle) con estado en {'identico', 'equivalente', 'falta_replica', 'falla', 'sin_control'}.
    """
    r1 = r2 = None
    for d in sorted(Path(results_root).glob("%s__*__%s__ar*__r*" % (BLOCK_CTRL, familia))):
        c = leer_json(d / "config.json", {}) or {}
        t10 = c.get("t10_rep") or {}
        if not (d / "DONE").exists() or (c.get("code") or {}).get("runner_tag") != tag:
            continue
        if t10.get("replica") == 1 and r1 is None:
            r1 = (d.name, t10)
        elif t10.get("replica") == 2 and r2 is None:
            r2 = (d.name, t10)
    if r1 is None:
        return "sin_control", {}
    s1 = (r1[1].get("vs_reference") or {}).get("scores") or {}
    det = {"r1": r1[0], "harness_sha256": (r1[1].get("harness") or {}).get("sha256"),
           "deterministas_ok": r1[1].get("control_deterministas_ok"),
           "r1_vs_T1": {k: s1.get(k) for k in ("identical", "rows_different", "max_abs_diff", "mean_abs_diff_desde_fm",
                                                "nan_mismas_posiciones")}}
    if not r1[1].get("control_deterministas_ok"):
        return "falla", det
    if r1[1].get("control_identico") is True:
        return "identico", det
    if r2 is None or r2[0].replace("__r2", "__r1") != r1[0]:
        return "falta_replica", det
    s2 = (r2[1].get("vs_reference") or {}).get("scores") or {}
    s12 = r2[1].get("vs_replica_1") or {}
    det.update(r2=r2[0], r2_vs_T1={k: s2.get(k) for k in det["r1_vs_T1"]},
               r2_vs_r1={k: s12.get(k) for k in det["r1_vs_T1"]}, K=K_RUIDO)
    try:
        m1, m2, m12 = (float(s1["mean_abs_diff_desde_fm"]), float(s2["mean_abs_diff_desde_fm"]),
                       float(s12["mean_abs_diff_desde_fm"]))
    except (KeyError, TypeError, ValueError):
        return "falla", dict(det, motivo="faltan las diferencias de alguna replica")
    nan_ok = all(s.get("nan_mismas_posiciones") for s in (s1, s2, s12))
    det["regla"] = "max(%.6g, %.6g) <= %g x %.6g y NaN en las mismas posiciones" % (m1, m2, K_RUIDO, m12)
    if r2[1].get("control_deterministas_ok") and nan_ok and m12 > 0 and max(m1, m2) <= K_RUIDO * m12:
        return "equivalente", det
    return "falla", dict(det, motivo="el job T1 no es indistinguible de una replica mas del arnes")


def resumen_job(d):
    """Lo que el JSON de resumen guarda de un job terminado (todo sale de sus propios ficheros; sin etiquetas)."""
    cfg = leer_json(d / "config.json", {}) or {}
    tim = leer_json(d / "timing.json", {}) or {}
    conv = leer_json(d / "convergence_report.json", {}) or {}
    t10 = cfg.get("t10_rep") or {}
    vs = t10.get("vs_reference") or {}
    ph = tim.get("phases") or {}
    return {"job_id": d.name, "dataset": (cfg.get("manifest_row") or {}).get("dataset"),
            "ar": (cfg.get("manifest_row") or {}).get("ar"), "control": t10.get("control"),
            "replica": t10.get("replica"), "factores": t10.get("factores"),
            "done": (d / "DONE").read_text().strip() if (d / "DONE").exists() else None,
            "reference_job": (cfg.get("reference") or {}).get("job_id"),
            "runner_tag": (cfg.get("code") or {}).get("runner_tag"),
            "harness_sha256": (t10.get("harness") or {}).get("sha256"),
            "all_fits_assert_ok": t10.get("all_fits_assert_ok"),
            "filas_train": [f.get("filas_train") for f in t10.get("fits") or []],
            "filas_val": [f.get("filas_val") for f in t10.get("fits") or []],
            "convergencia": conv.get("summary"),
            "razon": [x.get("razon") for x in conv.get("autoencoders") or []],
            "control_identico": t10.get("control_identico"),
            "scores_identical_to_reference": (vs.get("scores") or {}).get("identical"),
            "feature_map_identical": (vs.get("feature_map") or {}).get("identical"),
            "labels_identical": (vs.get("labels.npy") or {}).get("identical"),
            "desde_cache_todo_igual": (vs.get("desde_cache") or {}).get("todo_igual"),
            "regions_identical": vs.get("regions_identical"),
            "wall_seconds": tim.get("wall_seconds"), "train_s": ph.get("train_s"), "fm_s": ph.get("fm_s"),
            "exec_s": ph.get("exec_s"), "rss_peak_bytes": tim.get("rss_peak_bytes"), "anomalies": cfg.get("anomalies")}


def escribir_resumen(familia, extra=None):
    """Rewrites the summary JSON of a family from its job directories (idempotent)."""
    jobs, ctrls = [], []
    for ds in DATASETS:
        for ar in ARS:
            d = RESULTS / t10_job_id(ds, familia, ar)
            jobs.append(resumen_job(d) if (d / "DONE").exists() else
                        {"job_id": d.name, "dataset": ds, "ar": ar, "done": None})
    for c in sorted(RESULTS.glob("%s__*__%s__ar*__r*" % (BLOCK_CTRL, familia))):
        if (c / "DONE").exists():
            ctrls.append(resumen_job(c))
    hechos = [j for j in jobs if j["done"]]
    gast = reloj_gastado(RESULTS, familia)
    out = {"schema": "jisa-t10-rep/1", "created": ahora(), "tag": "[ENMIENDA-1] [EXPLORATORIO]", "decision": "D-59",
           "block": BLOCK, "familia": familia, "script": str(Path(__file__).resolve()),
           "script_sha256": sha256_file(__file__),
           "n_jobs": len(jobs), "n_done": len(hechos), "n_pending": len(jobs) - len(hechos),
           "pending": [j["job_id"] for j in jobs if not j["done"]],
           "all_done_assert_ok": bool(hechos) and all(j.get("all_fits_assert_ok") for j in hechos),
           "harness_sha256_distinct": sorted({j.get("harness_sha256") for j in hechos + ctrls if j.get("harness_sha256")}),
           "reloj_familia_s": gast, "reloj_familia_h": round(gast / 3600.0, 2), "tope_familia_h": TOPE_FAMILIA_S / 3600.0,
           "controls": ctrls, "jobs": jobs,
           "nota": "Post hoc experiment of the amendment to the analysis plan: outside S*_d, H1-H4 and the FINAL "
                   "aggregation. This file holds no detection metric. The entry rule of the comparison is computed "
                   "by puerta_fase_b.py from the convergence_report.json files."}
    if extra:
        out.update(extra)
    dest = AQUI / ("brazo_rep_%s.json" % familia)
    tmp = dest.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, dest)
    return out, dest


def padre(a):
    """Queue of one family: entry checks, CPU self-check, missing harness controls and jobs, one at a time."""
    signal.signal(signal.SIGTERM, _parar)
    signal.signal(signal.SIGINT, _parar)
    quien = "%s padre %s]" % (MARK, a.familia)
    if a.dataset:
        dss = [resolver_dataset(a.dataset)]
    elif a.all:
        dss = list(DATASETS)
    else:
        raise SystemExit("indica --dataset <stream> o --all")
    cola = [(ds, ar) for ds in dss for ar in ([a.ar] if a.ar else ARS)]
    refs = {j: ref_de(j[0], a.familia, j[1]) for j in cola}
    cola.sort(key=lambda j: (refs[j]["wall"], j))
    if os.path.realpath(sys.prefix) != os.path.realpath(R / "envs" / "tf215"):
        raise SystemExit("lanza este guion con %s (los hijos heredan el interprete)" % (R / "envs/tf215/bin/python"))
    if os.environ.get("LD_LIBRARY_PATH"):
        raise SystemExit("LD_LIBRARY_PATH no esta vacio; lanza con LD_LIBRARY_PATH=")
    root = Path(a.results_root).resolve()
    real = root == RESULTS.resolve()
    print("%s %s cola de %d job(s); factores %s; gpu=%s; destino %s; nice %d; script sha256 %s"
          % (ahora(), quien, len(cola), norm_factores(a.factores) or "-", a.gpu, a.results_root, os.nice(0),
             sha256_file(__file__)[:16]), flush=True)

    if not a.sin_registro:
        try:
            fac, det = arreglo_elegido(a.familia, a.registro)
        except RuntimeError as e:
            print("%s %s PARADA (fase A): %s" % (ahora(), quien, e), flush=True)
            return 3
        if norm_factores(a.factores) != fac:
            print("%s %s PARADA (fase A): --factores %r != %r del registro" % (ahora(), quien, a.factores, fac), flush=True)
            return 3
        print("%s %s fase A: escalon %d [%s]; control negativo completo: no_pasa (%s)"
              % (ahora(), quien, det["escalon"], fac, det["registro"]), flush=True)
    elif not a.preflight:
        raise SystemExit("--sin-registro solo con --preflight")
    motivo = puerta_git(con_brazo=(a.gpu != "cpu" and not a.preflight))
    if motivo:
        print("%s %s PARADA (puerta de git): %s" % (ahora(), quien, motivo), flush=True)
        return 2
    sin_ref = [ref_job_id(ds, a.familia, ar) for ds, ar in cola if not refs[(ds, ar)]["done"]]
    if sin_ref:
        print("%s %s PARADA: jobs T1 de referencia sin DONE: %s" % (ahora(), quien, sin_ref), flush=True)
        return 1

    if a.preflight:
        malos = 0
        for ds, ar in cola:
            rc, jl, sec = lanzar_hijo(ds, ar, a, preflight=True)
            malos += rc != 0
        print("%s %s PREFLIGHT: %d/%d en verde" % (ahora(), quien, len(cola) - malos, len(cola)), flush=True)
        return 1 if malos else 0

    ctl = dir_control(root)
    ctl.mkdir(parents=True, exist_ok=True)
    cerrojo = open(ctl / ("padre.%s.lock" % a.familia), "a+")
    try:
        fcntl.flock(cerrojo, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("%s %s ya hay otro padre de esta familia en marcha; salgo" % (ahora(), quien), flush=True)
        return 1

    st = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--selftest", "--familia", a.familia,
                         "--factores", norm_factores(a.factores)], env=entorno_hijo("cpu"), capture_output=True,
                        text=True, cwd=str(R))
    lin = lineas_marcadas(st.stdout)
    for ln in lin:
        print(ln[:1500], flush=True)
    if st.returncode != 0 or not lin:
        print("%s %s la autocomprobacion ha fallado; no se lanza nada\n%s" % (ahora(), quien, st.stderr[-3000:]),
              flush=True)
        return 4
    selftest_res = json.loads(lin[-1].split("selftest] ", 1)[1])

    def presupuesto(ds, ar):
        """(cabe, gastado, previsto) antes de lanzar un hijo del job (ds, ar)."""
        gast = reloj_gastado(root, a.familia)
        prev = refs[(ds, ar)]["wall"] * factor_duracion(root, a.familia)
        return gast + prev <= TOPE_FAMILIA_S, gast, prev

    def anota_ref(ds, ar, rc, sec, control):
        """Deja en el reloj el cociente frente al job T1 (para la prevision de los siguientes)."""
        reloj_anota(root, a.familia, {"tipo": "cociente", "job_id": t10_job_id(ds, a.familia, ar, control), "rc": rc,
                                      "segundos": sec, "ref_wall": refs[(ds, ar)]["wall"], "control": control})

    ejecutados, controles, sin_presupuesto = [], {}, []
    veto = None
    if a.con_control or a.solo_control:
        for tag in sorted({refs[j]["runner_tag"] for j in cola}, key=str):
            ds, ar = next(j for j in cola if refs[j]["runner_tag"] == tag)
            pend = [1 for j in cola if not (root / t10_job_id(j[0], a.familia, j[1]) / "DONE").exists()]
            estado, det = veredicto_control(root, a.familia, tag)
            for replica in (1, 2):
                if estado not in ("sin_control", "falta_replica") or ESTADO["parar"] or (not pend and not a.solo_control):
                    break
                if (estado == "sin_control") != (replica == 1):
                    continue
                cabe, gast, prev = presupuesto(ds, ar)
                if not cabe:
                    sin_presupuesto.append(t10_job_id(ds, a.familia, ar, True, replica))
                    break
                print("%s %s control del arnes para %s (replica %d) con %s ar%s" % (ahora(), quien, tag, replica, ds, ar),
                      flush=True)
                rc, jl, sec = lanzar_con_ram(ds, ar, a, quien, control=True, replica=replica)
                ejecutados.append({"job_id": t10_job_id(ds, a.familia, ar, True, replica), "rc": rc, "seconds": sec,
                                   "log": str(jl), "control": True})
                estado, det = veredicto_control(root, a.familia, tag)
                if rc != 0:
                    break
            controles[str(tag)] = {"estado": estado, "detalle": det}
            if estado in ("identico", "equivalente"):
                if det.get("harness_sha256") != sha256_file(__file__):
                    print("%s %s AVISO: el control de %s se hizo con otra version de este guion" % (ahora(), quien, tag),
                          flush=True)
                print("%s %s control de %s: %s %s" % (ahora(), quien, tag, estado.upper(),
                                                      det.get("regla") or "(scores.npy y ae_errors.npy byte a byte)"),
                      flush=True)
            elif pend or a.solo_control:
                veto = veto or "control de %s: %s" % (tag, estado)
                print("%s %s EL CONTROL DE %s NO VALE (%s): no se lanza ningun job de %s. %s"
                      % (ahora(), quien, tag, estado, a.familia, json.dumps(det, ensure_ascii=False)[:1200]), flush=True)

    parada = None
    if not a.solo_control and not veto:
        pend = [j for j in cola]
        n = 0
        while pend and not ESTADO["parar"] and not parada:
            ds, ar = pend.pop(0)
            n += 1
            jid = t10_job_id(ds, a.familia, ar)
            if (root / jid / "DONE").exists():
                print("%s %s [%d/%d] %s ya tiene DONE, lo salto" % (ahora(), quien, n, len(cola), jid), flush=True)
                continue
            cabe, gast, prev = presupuesto(ds, ar)
            if not cabe:
                sin_presupuesto.append(jid)
                reloj_anota(root, a.familia, {"tipo": "tope", "job_id": jid, "gastado_s": gast, "previsto_s": prev,
                                              "tope_s": TOPE_FAMILIA_S})
                print("%s %s [%d/%d] %s not launched: %.2f h spent + %.2f h expected > %.0f h (budget of the family)"
                      % (ahora(), quien, n, len(cola), jid, gast / 3600, prev / 3600, TOPE_FAMILIA_S / 3600), flush=True)
                ejecutados.append({"job_id": jid, "rc": None, "skipped": "presupuesto de la familia"})
                continue
            print("%s %s [%d/%d] lanzo %s (reloj de la familia %.2f h; previsto %.2f h)"
                  % (ahora(), quien, n, len(cola), jid, gast / 3600, prev / 3600), flush=True)
            rc, jl, sec = lanzar_con_ram(ds, ar, a, quien)
            if rc is None:
                parada = str(jl)
                ejecutados.append({"job_id": jid, "rc": None, "skipped": parada})
                break
            anota_ref(ds, ar, rc, sec, False)
            ejecutados.append({"job_id": jid, "rc": rc, "seconds": sec, "log": str(jl)})
            print("%s %s [%d/%d] %s -> codigo %d en %.0f s" % (ahora(), quien, n, len(cola), jid, rc, sec), flush=True)
            if rc in (8, 10):
                print("%s %s codigo %d = error sistematico: se detiene la cola" % (ahora(), quien, rc), flush=True)
                break
    elif veto:
        for ds, ar in cola:
            jid = t10_job_id(ds, a.familia, ar)
            if not (root / jid / "DONE").exists():
                ejecutados.append({"job_id": jid, "rc": None, "skipped": veto})

    gast = reloj_gastado(root, a.familia)
    if real:
        res, dest = escribir_resumen(a.familia, {"last_run": {
            "started_args": vars(a), "selftest": selftest_res, "ran": ejecutados, "controles": controles, "veto": veto,
            "sin_presupuesto": sin_presupuesto, "interrupted": ESTADO["parar"]}})
        print("%s %s resumen -> %s: %d/%d jobs DONE, faltan %d; controles %d"
              % (ahora(), quien, dest, res["n_done"], res["n_jobs"], res["n_pending"], len(res["controls"])), flush=True)
    fallos = [e for e in ejecutados if e.get("rc") not in (0, None)] + [e for e in ejecutados if e.get("skipped")]
    print("%s %s FIN: %d lanzados, %d con problemas%s; reloj de la familia %.2f h de %.0f"
          % (ahora(), quien, len([e for e in ejecutados if e.get("rc") is not None]), len(fallos),
             " (interrumpido)" if ESTADO["parar"] else "", gast / 3600, TOPE_FAMILIA_S / 3600), flush=True)
    if any(c["estado"] == "falla" for c in controles.values()):
        return 5
    if sin_presupuesto:
        return 12
    if veto:
        return 5
    if parada and not ESTADO["parar"]:
        return 13
    return 1 if (fallos or ESTADO["parar"]) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Modified-encoder jobs T10-REP of the amendment to the analysis plan (seed A).")
    ap.add_argument("--familia", default=None, choices=sorted(FAMILIAS), help="conv1d or conv2d (MHA has no modified version)")
    ap.add_argument("--factores", default="", help="los que eligio la fase A, p. ej. P,I,R (se comprueban con el registro)")
    ap.add_argument("--dataset", default=None, help="un stream (nombre del manifiesto o slug)")
    ap.add_argument("--ar", default=None, choices=ARS, help="T (AR) o F (TSR); por defecto los dos")
    ap.add_argument("--all", action="store_true", help="los 9 streams")
    ap.add_argument("--con-control", action="store_true",
                    help="antes de los jobs, el control del arnes de cada version del runner sin validar")
    ap.add_argument("--solo-control", action="store_true", help="solo los controles del arnes")
    ap.add_argument("--preflight", action="store_true", help="solo comprobaciones, en CPU y sin entrenar")
    ap.add_argument("--sin-registro", action="store_true",
                    help="solo con --preflight: no exige que la fase A haya elegido la familia")
    ap.add_argument("--resumen", action="store_true", help="rewrites the summary JSON of the family and exits")
    ap.add_argument("--puerta", action="store_true", help="solo comprueba los tags de git y sale (0 o 2)")
    ap.add_argument("--gpu", default="0", help="'0' (produccion) o 'cpu' (solo ensayos fuera de results/revision)")
    ap.add_argument("--results-root", default=str(RESULTS), help="solo se cambia en ensayos con --gpu cpu")
    ap.add_argument("--registro", default=str(REGISTRO), help="solo se cambia en ensayos fuera de results/revision")
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--control", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--replica", default="1", choices=("1", "2"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.puerta:
        motivo = puerta_git()
        print("%s puerta de git: %s" % (ahora(), motivo or "OK"), flush=True)
        return 2 if motivo else 0
    if not a.familia:
        raise SystemExit("indica --familia conv1d o --familia conv2d")
    if a.selftest:
        return selftest(a.familia, a.factores)
    if a.resumen:
        r, dest = escribir_resumen(a.familia)
        print("%s -> %d/%d jobs DONE, faltan %d" % (dest, r["n_done"], r["n_jobs"], r["n_pending"]))
        return 0
    if a.gpu not in ("0", "cpu"):
        raise SystemExit("--gpu solo admite 0 o cpu")
    if not os.path.isabs(a.results_root) or not os.path.isabs(a.registro):
        raise SystemExit("--results-root y --registro deben ser absolutas")
    if a.child:
        if not a.dataset or not a.ar:
            raise SystemExit("--child necesita --dataset y --ar")
        return hijo(a)
    return padre(a)


if __name__ == "__main__":
    sys.exit(main())
