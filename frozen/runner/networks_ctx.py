#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""networks_ctx.py - MLP autoencoders with a context window ('mlp_ctx', 'mlp_deep_ctx') and the re-trained pair, registered at run time."""
import numpy as np
import tensorflow as tf

from src.models.__special__ import logging, HASTE_VALUE, BATCH_SIZE, EXECUTION_BS
from src.models.utils import create_windowed_data, create_windowed_data_ar

__all__ = ["MLPContextAutoencoder", "DeepMLPContextAutoencoder", "CONTEXT_NETWORK_TYPES", "architecture_summary"]


class _WindowedMLPBase:
    """Parte comun de las dos variantes: mismo ciclo train/execute que las redes con ventana publicadas."""

    ERROR_FN = "rmse"
    VARIANT_KEY = None

    def __init__(self, n_visible=5, n_hidden=3, lr=0.001, corruption_level=0.0, hidden_ratio=None,
                 sequence_length=500, seed=1234, ar=False, haste: bool = False, name="Anonymous dA", **kwargs):
        """
        :param n_visible: numero de features del grupo (entrada y salida por fila).
        :param n_hidden: unidades del cuello de botella si hidden_ratio es None.
        :param lr: learning rate de Adam.
        :param corruption_level: probabilidad de dropout (solo la variante A, como multilayer.py:47).
        :param hidden_ratio: razon de compresion; n_hidden = ceil(hidden_ratio * n_visible).
        :param sequence_length: L, longitud de la ventana de contexto.
        :param seed: semilla declarada (no se usa aqui; las semillas las fija instrument.apply_seeds).
        :param ar: True -> create_windowed_data_ar (predice la fila siguiente); False -> create_windowed_data.
        :param haste: False -> epochs = HASTE_VALUE (kitnet.py:297-305).
        :param name: nombre del AE.
        """
        self.input_dim = n_visible
        self.lr = lr
        self.dropout = corruption_level
        self.n_hidden = int(np.ceil(hidden_ratio * n_visible) if hidden_ratio is not None else n_hidden)

        self.seed = seed
        self.name = name

        self.norm_min = 0
        self.norm_max = 1

        self.create_windowed = create_windowed_data_ar if ar else create_windowed_data
        self.haste = 1 if haste else HASTE_VALUE
        self.ar = bool(ar)

        self.window = np.zeros((sequence_length, n_visible))
        self.sequence_length = sequence_length

        self.encoder, self.decoder = self._build(n_visible, sequence_length)
        self.model = tf.keras.models.Sequential([self.encoder, self.decoder], name=type(self).__name__)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.build(input_shape=(None, sequence_length, n_visible))

    def _build(self, n_visible, sequence_length):
        raise NotImplementedError

    def _row_error(self, y_batch, predicted):
        """Error por fila del lote. 'rmse' = conv1d.py:90; 'mse' = multilayer.py:86-87 (sin raiz)."""
        sq = tf.reduce_mean(tf.square(y_batch - predicted), axis=1)
        return tf.sqrt(sq) if self.ERROR_FN == "rmse" else sq

    def train(self, input_x):
        """
        Entrena el AE y devuelve el error de reconstruccion fila a fila del propio tramo.
        :param input_x: tramo de entrenamiento (n, n_visible).
        :return: lista con el error por fila.
        """
        logging.info(f"Training {self.name}...")
        self.norm_min = np.min(input_x, axis=0)
        self.norm_max = np.max(input_x, axis=0)
        x = (input_x - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)
        sequential_x, self.window = self.create_windowed(x, self.sequence_length, self.window, batch_size=BATCH_SIZE)
        self.model.fit(sequential_x, epochs=self.haste, verbose=1)
        logging.info(f"Training {self.name}... Done!")
        sequential_x, self.window = self.create_windowed(x, self.sequence_length, self.window, batch_size=EXECUTION_BS)
        mse_per_sample = list()
        for (x_batch, y_batch) in sequential_x:
            predicted = self.model.predict(x_batch, verbose=0)
            mse_batched = self._row_error(y_batch, predicted)
            mse_per_sample.extend(mse_batched.numpy())
        return mse_per_sample

    def execute(self, x):
        """
        Puntua filas nuevas sin entrenar.
        :param x: bloque de ejecucion (n, n_visible).
        :return: error por fila (np.ndarray).
        """
        x = np.clip(x, self.norm_min, self.norm_max)
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)
        logging.info(f"Executing {self.name} with {len(x)} instances...")
        sequential_x, self.window = self.create_windowed(x, self.sequence_length, self.window, batch_size=EXECUTION_BS)

        mse_per_sample = list()
        for (x_batch, y_batch) in sequential_x:
            predicted = self.model.predict(x_batch, verbose=0)
            mse_batched = self._row_error(y_batch, predicted)
            mse_per_sample.extend(mse_batched.numpy())
        return np.array(mse_per_sample)


class MLPContextAutoencoder(_WindowedMLPBase):
    """MLP publicado (multilayer.py) con la ventana aplanada como entrada. Un unico factor cambiado."""

    ERROR_FN = "mse"
    VARIANT_KEY = "mlp_ctx"

    def _build(self, n_visible, sequence_length):
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(sequence_length, n_visible)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_hidden, activation='relu'),
            tf.keras.layers.Dropout(self.dropout)
        ], name='encoder')
        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(n_visible, activation='sigmoid')
        ], name='decoder')
        return encoder, decoder


class DeepMLPContextAutoencoder(_WindowedMLPBase):
    """Deep MLP de la Tabla 3 con ventana (arquitectura de deep_mlp_ae.py:52-83)."""

    ERROR_FN = "rmse"
    VARIANT_KEY = "mlp_deep_ctx"

    def _build(self, n_visible, sequence_length):
        input_dim = sequence_length * n_visible
        self.flat_dim = int(input_dim)
        self.h1 = max(1, input_dim // 4)
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(sequence_length, n_visible)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.h1, activation='relu'),
            tf.keras.layers.Dense(self.n_hidden, activation='relu')
        ], name='encoder')
        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.h1, activation='relu'),
            tf.keras.layers.Dense(n_visible, activation='sigmoid')
        ], name='decoder')
        return encoder, decoder


CONTEXT_NETWORK_TYPES = {"mlp_ctx": MLPContextAutoencoder, "mlp_deep_ctx": DeepMLPContextAutoencoder}


def architecture_summary(variant, feature_map_sizes, sequence_length=800, hidden_ratio=0.22):
    """
    Arquitectura y recuento de parametros de cada AE, calculados a mano (sin TF).
    :param variant: 'mlp_ctx' o 'mlp_deep_ctx'.
    :param feature_map_sizes: tamanos de los grupos del Feature Map (p. ej. [82, 15, 8, 10] en Active Wiretap).
    :param sequence_length: L.
    :param hidden_ratio: razon de compresion.
    :return: dict con una entrada por AE y el total.
    """
    if variant not in CONTEXT_NETWORK_TYPES:
        raise ValueError(f"variante {variant!r} no es una variante de contexto {sorted(CONTEXT_NETWORK_TYPES)}")
    aes, total = [], 0
    for i, nv in enumerate(feature_map_sizes):
        nh = int(np.ceil(hidden_ratio * nv))
        flat = sequence_length * nv
        if variant == "mlp_ctx":
            layers = [("encoder/flatten", 0), ("encoder/dense", flat * nh + nh), ("encoder/dropout", 0),
                      ("decoder/dense", nh * nv + nv)]
        else:
            h1 = max(1, flat // 4)
            layers = [("encoder/flatten", 0), ("encoder/dense_1", flat * h1 + h1), ("encoder/dense_2", h1 * nh + nh),
                      ("decoder/dense_1", nh * h1 + h1), ("decoder/dense_2", h1 * nv + nv)]
        n = sum(p for _, p in layers)
        total += n
        aes.append({"index": i, "n_visible": nv, "n_hidden": nh, "flat_dim": flat,
                    "h1": (max(1, flat // 4) if variant == "mlp_deep_ctx" else None),
                    "layers": layers, "params": n})
    return {"variant": variant, "sequence_length": sequence_length, "hidden_ratio": hidden_ratio,
            "feature_map_sizes": list(feature_map_sizes), "autoencoders": aes, "total_params": total,
            "error_fn": CONTEXT_NETWORK_TYPES[variant].ERROR_FN}


if __name__ == "__main__":
    import json
    for v in ("mlp_ctx", "mlp_deep_ctx"):
        print(json.dumps(architecture_summary(v, [82, 15, 8, 10]), indent=1))


VAL_FRAC = 0.10
LR_RULE = "lr_capa = lr0 * n_v / fan_in_capa (anclada en la aritmetica de Adam, no ajustada con metricas)"


class _TwoLROptimizer(tf.keras.Model):
    """Sequential(encoder, decoder) con un lr distinto por bloque. Adam da un paso +-lr por peso, asi que el lr
    por capa es la unica forma de igualar el movimiento de la preactivacion entre arquitecturas."""

    def __init__(self, encoder, decoder, lr_enc, lr_dec, **kw):
        super().__init__(**kw)
        self.encoder, self.decoder = encoder, decoder
        self.opt_enc = tf.keras.optimizers.Adam(learning_rate=lr_enc)
        self.opt_dec = tf.keras.optimizers.Adam(learning_rate=lr_dec)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, x, training=False):
        return self.decoder(self.encoder(x, training=training), training=training)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(y, self(x, training=True))
        self.opt_enc.apply_gradients(zip(tape.gradient(loss, self.encoder.trainable_variables),
                                         self.encoder.trainable_variables))
        self.opt_dec.apply_gradients(zip(tape.gradient(loss, self.decoder.trainable_variables),
                                         self.decoder.trainable_variables))
        del tape
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        self.loss_tracker.update_state(self.loss_fn(y, self(x, training=False)))
        return {"loss": self.loss_tracker.result()}


class _FixedBase(_WindowedMLPBase):
    """Base of the re-trained pair: per-layer learning-rate rule, held-out validation and convergence gates CG-0..CG-4."""

    ERROR_FN = "mse"
    LR0 = None

    USE_TWO_LR = True
    EPOCHS = None
    CONTAINMENT_INIT = False

    def __init__(self, **kw):
        self._lr0 = kw.get("lr", 0.001)
        super().__init__(**kw)
        if self.USE_TWO_LR:
            self._install_two_lr()
        else:
            self.lr_encoder = self.lr_decoder = self._lr0
            self.fan_in_encoder = int(self._fan_in_encoder(self.input_dim, self.sequence_length))
        if self.EPOCHS is not None:
            self.haste = int(self.EPOCHS)
        self.convergence = {"lr_rule": LR_RULE if self.USE_TWO_LR else "un solo Adam con lr0 (sin regla Pi)",
                            "epochs": self.haste,
                            "val_frac": VAL_FRAC, "gates": {}, "per_epoch": {}}

    def _fan_in_encoder(self, n_visible, sequence_length):
        raise NotImplementedError

    def _install_two_lr(self):
        """Instala el modelo de dos lr DESPUES de que la base haya construido el suyo (si no, lo pisa)."""
        n_visible, sequence_length = self.input_dim, self.sequence_length
        fan_in = self._fan_in_encoder(n_visible, sequence_length)
        lr_enc = self._lr0 * n_visible / float(fan_in)
        self.lr_encoder, self.lr_decoder, self.fan_in_encoder = lr_enc, self._lr0, int(fan_in)
        self.model = _TwoLROptimizer(self.encoder, self.decoder, lr_enc, self._lr0,
                                     name=type(self).__name__)
        self.model.compile()
        self.model.build(input_shape=(None, sequence_length, n_visible))

    @staticmethod
    def _constant_floor(y):
        """CG-1: MSE del mejor predictor constante (la media por feature del objetivo)."""
        return float(np.mean((np.asarray(y) - np.asarray(y).mean(axis=0)) ** 2))

    def _health(self, X, Y):
        """CG-4: unidades muertas, dispersion de la preactivacion y de la salida entre ventanas."""
        h = self.encoder.predict(X, verbose=0, batch_size=EXECUTION_BS)
        p = self.model.predict(X, verbose=0, batch_size=EXECUTION_BS)
        h2 = np.asarray(h).reshape(len(h), -1)
        return {"dead_units_frac": float(np.mean(np.all(h2 <= 0.0, axis=0))),
                "encoder_std_between_windows": float(np.mean(np.std(h2, axis=0))),
                "output_std_between_windows": float(np.mean(np.std(np.asarray(p), axis=0))),
                "mse": float(np.mean((np.asarray(p) - np.asarray(Y)) ** 2))}

    def _first_step_probe(self, X, Y):
        """CG-0: cuanto mueve UN paso de Adam la preactivacion del cuello de botella, frente a su dispersion
        inicial. Es analitica a proposito: Adam da un paso +-lr por peso, asi que la preactivacion se mueve
        ~lr * fan_in * E[|x|] sin depender del gradiente. No se entrena nada, asi que no ensucia el estado del
        optimizador ni consume presupuesto. Si el cociente es >> 1, el modelo muere en el primer paso."""
        Xs = np.asarray(X[:64], dtype=np.float64)
        h0 = np.asarray(self.encoder.predict(X[:64], verbose=0, batch_size=64)).reshape(min(64, len(X)), -1)
        step = float(self.lr_encoder * self.fan_in_encoder * np.mean(np.abs(Xs)))
        pre_std = float(np.mean(np.std(h0, axis=0))) or float("nan")
        return {"adam_step_on_preactivation": step,
                "encoder_out_std_at_init": pre_std,
                "ratio_step_over_std": step / pre_std if pre_std == pre_std and pre_std > 0 else None,
                "pass": bool(step < 1.0),
                "rule": "paso ~ lr_enc * fan_in * E[|x|]; con la regla Pi debe quedar del orden del MLP por "
                        "registro (0,12 con lr0 = 0,002, fan_in = 82 y E[|x|] ~ 0,73)"}

    def _copy_containment(self, ref):
        """F2: copia el por-registro entrenado 'ref' en este modelo con ventana, de forma que la funcion sea
        EXACTAMENTE la misma antes de entrenar. Flatten sobre (L, n_v) recorre en orden C (fila, feature), asi
        que la ultima fila de la ventana ocupa los ultimos n_v pesos de la capa densa del codificador."""
        W, b = [l for l in ref.encoder.layers if isinstance(l, tf.keras.layers.Dense)][0].get_weights()
        dl = [l for l in self.encoder.layers if isinstance(l, tf.keras.layers.Dense)][0]
        Wc, bc = dl.get_weights()
        if Wc.shape[0] != self.sequence_length * self.input_dim or W.shape != (self.input_dim, Wc.shape[1]):
            raise ValueError(f"init por contencion imposible: enc {Wc.shape} ref {W.shape} "
                             f"L={self.sequence_length} n_v={self.input_dim}")
        Wc[:] = 0.0
        Wc[-self.input_dim:, :] = W
        bc[:] = b
        dl.set_weights([Wc, bc])
        dd = [l for l in self.decoder.layers if isinstance(l, tf.keras.layers.Dense)][0]
        dd.set_weights([np.array(w, copy=True) for w in
                        [l for l in ref.decoder.layers if isinstance(l, tf.keras.layers.Dense)][0].get_weights()])

    def _containment_reference(self, input_x):
        """Entrena el por-registro de referencia sobre el MISMO tramo (misma normalizacion, mismo corte de
        validacion, mismos pasos). Devuelve (modelo, puertas). No hace el pase de errores por fila: aqui solo
        se necesitan sus PESOS y su MSE de validacion, no sus scores."""
        ref = MLPPerRecordFixedAutoencoder(
            n_visible=self.input_dim, n_hidden=self.n_hidden, lr=self._lr0, corruption_level=self.dropout,
            sequence_length=self.sequence_length, seed=self.seed, ar=self.ar,
            haste=(self.haste == 1), name=f"{self.name}/ref-contencion")
        ref.haste = self.haste
        ref.train(input_x, _reference_only=True)
        return ref, ref.convergence["gates"]

    def train(self, input_x, _reference_only=False):
        logging.info(f"Training {self.name} (par re-entrenado, lr_enc={self.lr_encoder:.3g})...")
        self.norm_min = np.min(input_x, axis=0)
        self.norm_max = np.max(input_x, axis=0)
        x = (input_x - self.norm_min) / (self.norm_max - self.norm_min + 1e-16)
        n_val = max(1, int(VAL_FRAC * len(x)))
        x_tr, x_va = x[:-n_val], x[-n_val:]

        seq_tr, self.window = self.create_windowed(x_tr, self.sequence_length, self.window, batch_size=BATCH_SIZE)
        win_va, _ = self.create_windowed(x_va, self.sequence_length, self.window.copy(), batch_size=EXECUTION_BS)
        Xv, Yv = [], []
        for xb, yb in win_va:
            Xv.append(xb.numpy()); Yv.append(yb.numpy())
        Xv = np.concatenate(Xv) if Xv else np.zeros((0,))
        Yv = np.concatenate(Yv) if Yv else np.zeros((0,))

        floor = self._constant_floor(Yv) if len(Yv) else float("nan")
        probe = None
        if len(Xv) >= 64:
            try: probe = self._first_step_probe(Xv, Yv)
            except Exception as e: probe = {"error": f"{type(e).__name__}: {e}"}

        val_data = (Xv, Yv) if len(Xv) else None
        ref_gates = None
        if self.CONTAINMENT_INIT and not _reference_only:
            ref, ref_gates = self._containment_reference(input_x)
            self._copy_containment(ref)
            h0 = self._health(Xv, Yv) if len(Xv) else {}
            self.convergence["containment_init"] = {
                "reference_val_mse": ref_gates.get("CG2_val_mse"),
                "val_mse_after_copy": h0.get("mse"),
                "copy_exact": bool(h0.get("mse") is not None and ref_gates.get("CG2_val_mse")
                                   and abs(h0["mse"] - ref_gates["CG2_val_mse"])
                                   <= 1e-6 + 1e-4 * abs(ref_gates["CG2_val_mse"])),
                "reference_gates": ref_gates}
            del ref
        hist = self.model.fit(seq_tr, epochs=self.haste, verbose=1, validation_data=val_data,
                              validation_batch_size=EXECUTION_BS if val_data is not None else None)
        health = self._health(Xv, Yv) if len(Xv) else {}
        ratio = (health.get("mse") / floor) if (health and floor and floor > 0) else None
        self.convergence["gates"] = {
            "CG0_first_step": probe,
            "CG1_constant_floor": floor,
            "CG1_ratio_val": ratio,
            "CG1_pass": bool(ratio is not None and ratio < 0.9),
            "CG1_verdict": (None if ratio is None else
                            ("entrenado" if ratio < 0.9 else ("degenerado" if ratio > 1.15 else "indeciso"))),
            "CG3_val_loss_per_epoch": [float(v) for v in hist.history.get("val_loss", [])],
            "CG3_pass": bool(hist.history.get("val_loss") and
                             all(v == v and abs(v) < float("inf") for v in hist.history["val_loss"])),
            "CG4_health": health,
            "CG4_pass": bool(health and health.get("output_std_between_windows", 0) > 1e-3),
            "lr_encoder": self.lr_encoder, "lr_decoder": self.lr_decoder, "fan_in_encoder": self.fan_in_encoder,
            "n_val_rows": int(n_val), "n_val_windows": int(len(Xv)),
            "CG2_pairing_key": {"n_visible": int(self.input_dim), "n_hidden": int(self.n_hidden),
                                "ar": bool(self.ar), "sequence_length": int(self.sequence_length),
                                "n_val_windows": int(len(Xv)), "variant": self.VARIANT_KEY},
            "CG2_val_mse": (float(health["mse"]) if health.get("mse") is not None else None),
            "CG2_reference_val_mse": (ref_gates or {}).get("CG2_val_mse"),
            "CG2_ratio_windowed_over_per_record": (
                (health["mse"] / ref_gates["CG2_val_mse"])
                if (ref_gates and ref_gates.get("CG2_val_mse") and health.get("mse") is not None
                    and ref_gates["CG2_val_mse"] > 0) else None),
            "CG2_pass": (None if not ref_gates or not ref_gates.get("CG2_val_mse") else
                         bool(health.get("mse") is not None
                              and health["mse"] / ref_gates["CG2_val_mse"] <= 1.02))}
        self.convergence["per_epoch"] = {"loss": [float(v) for v in hist.history.get("loss", [])],
                                         "val_loss": [float(v) for v in hist.history.get("val_loss", [])]}

        if _reference_only:
            logging.info(f"Training {self.name}... referencia de contencion lista "
                         f"(CG1={self.convergence['gates']['CG1_pass']})")
            return []

        seq_all, self.window = self.create_windowed(x, self.sequence_length, np.zeros_like(self.window),
                                                    batch_size=EXECUTION_BS)
        mse_per_sample = []
        for (xb, yb) in seq_all:
            mse_per_sample.extend(self._row_error(yb, self.model.predict(xb, verbose=0)).numpy())
        g = self.convergence["gates"]
        logging.info(f"Training {self.name}... Done! CG1={g['CG1_pass']} ({g['CG1_verdict']}, "
                     f"ratio={g['CG1_ratio_val']}) CG2={g['CG2_pass']} "
                     f"(ventana/registro={g['CG2_ratio_windowed_over_per_record']}) "
                     f"CG3={g['CG3_pass']} CG4={g['CG4_pass']} "
                     f"n_h={self.n_hidden} n_v={self.input_dim} lr_enc={self.lr_encoder:.3g}")
        return mse_per_sample


class MLPContextFixedAutoencoder(_FixedBase):
    """'mlp_ctx_fix': the windowed MLP with the per-layer learning-rate rule; the only factor changed."""

    VARIANT_KEY = "mlp_ctx_fix"
    CONTAINMENT_INIT = True

    def _fan_in_encoder(self, n_visible, sequence_length):
        return sequence_length * n_visible

    def _build(self, n_visible, sequence_length):
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(sequence_length, n_visible)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_hidden, activation='relu'),
            tf.keras.layers.Dropout(self.dropout)], name='encoder')
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(n_visible, activation='sigmoid')], name='decoder')
        return self.encoder, self.decoder


class MLPPerRecordFixedAutoencoder(_FixedBase):
    """'mlp_fix': control without window under the same protocol (held-out validation and gates)."""

    VARIANT_KEY = "mlp_fix"

    def _fan_in_encoder(self, n_visible, sequence_length):
        return n_visible

    def _build(self, n_visible, sequence_length):
        self.encoder = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(sequence_length, n_visible)),
            tf.keras.layers.Lambda(lambda w: w[:, -1, :]),
            tf.keras.layers.Dense(self.n_hidden, activation='relu'),
            tf.keras.layers.Dropout(self.dropout)], name='encoder')
        self.decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(n_visible, activation='sigmoid')], name='decoder')
        return self.encoder, self.decoder


class MLPContextFixedE5Autoencoder(MLPContextFixedAutoencoder):
    """'mlp_ctx_fix_e5': identical to mlp_ctx_fix with five epochs; the only factor changed."""

    VARIANT_KEY = "mlp_ctx_fix_e5"
    EPOCHS = 5


class MLPPerRecordFixedE5Autoencoder(MLPPerRecordFixedAutoencoder):
    """'mlp_fix_e5': el control sin ventana con el MISMO presupuesto de cinco epocas, para que el contraste del
    par siga aislando la ventana y no el numero de pasos."""

    VARIANT_KEY = "mlp_fix_e5"
    EPOCHS = 5


CONTEXT_NETWORK_TYPES = {"mlp_ctx": MLPContextAutoencoder, "mlp_deep_ctx": DeepMLPContextAutoencoder,
                         "mlp_ctx_fix": MLPContextFixedAutoencoder, "mlp_fix": MLPPerRecordFixedAutoencoder,
                         "mlp_ctx_fix_e5": MLPContextFixedE5Autoencoder,
                         "mlp_fix_e5": MLPPerRecordFixedE5Autoencoder}
__all__ += ["MLPContextFixedAutoencoder", "MLPPerRecordFixedAutoencoder",
            "MLPContextFixedE5Autoencoder", "MLPPerRecordFixedE5Autoencoder", "VAL_FRAC", "LR_RULE"]
