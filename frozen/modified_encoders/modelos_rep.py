#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windowed autoencoder families rebuilt layer by layer, with the switchable modifications P, N, I, R and A."""
import numpy as np
import tensorflow as tf

KL = tf.keras.layers
LR0 = 0.002
HIDDEN_RATIO = 0.22
FACTORES = {"conv1d": set("PIRA"), "conv2d": set("PIRA"), "mha": set("NRA")}
ESCALERA = {
    "mha": ["N", "N,R", "N,R,A"],
    "conv1d": ["P", "P,I", "P,I,R", "P,I,R,A"],
    "conv2d": ["P", "P,I", "P,I,R", "P,I,R,A"],
}


def n_hidden(n_v):
    return int(np.ceil(HIDDEN_RATIO * n_v))


def parse_factores(txt):
    return frozenset(f for f in (txt or "").replace(" ", "").split(",") if f)


class TapOrtogonal(tf.keras.initializers.Initializer):
    """Factor I. Nucleo a cero salvo la posicion alineada con la fila actual ((k-1)//2 en cada eje espacial con
    padding 'same'), que recibe una matriz (semi)ortogonal de ganancia 1."""

    def __call__(self, shape, dtype=None, **kw):
        shape = tuple(int(s) for s in shape)
        *esp, cin, cout = shape
        q = tf.keras.initializers.Orthogonal(gain=1.0)((cin, cout), dtype=tf.float32).numpy()
        k = np.zeros(shape, dtype=np.float32)
        k[tuple((s - 1) // 2 for s in esp)] = q
        return tf.constant(k, dtype=dtype or tf.float32)

    def get_config(self):
        return {}


def _init(f):
    return TapOrtogonal() if "I" in f else "glorot_uniform"


def _densa(x, n, f, nombre=None):
    """Densa con ReLU (original) o con LeakyReLU(0,01) (factor A)."""
    if "A" in f:
        return KL.LeakyReLU(alpha=0.01)(KL.Dense(n, name=nombre)(x))
    return KL.Dense(n, activation="relu", name=nombre)(x)


def _decoder(n_v, n_h, f):
    z = KL.Input(shape=(n_h,))
    return tf.keras.Model(z, KL.Dense(n_v, activation="sigmoid")(_densa(z, n_h, f)), name="decoder")


def _enc_conv1d(n_v, L, f):
    n_h, k = n_hidden(n_v), L // 4
    x = KL.Input(shape=(L, n_v))
    h = KL.Conv1D(n_v, k, padding="same", activation="tanh", kernel_initializer=_init(f))(x)
    h = KL.MaxPooling1D(pool_size=2, padding="same")(h)
    if "A" in f:
        h = KL.LeakyReLU(alpha=0.01)(KL.Conv1D(n_h, k, padding="same", kernel_initializer=_init(f))(h))
    else:
        h = KL.Conv1D(n_h, k, padding="same", activation="relu", kernel_initializer=_init(f))(h)
    g = KL.GlobalAveragePooling1D()(h)
    if "R" in f:
        g = KL.Concatenate()([g, KL.Lambda(lambda t: t[:, -1, :], name="ultima_posicion")(h)])
    return tf.keras.Model(x, _densa(g, n_h, f), name="encoder")


def _enc_conv2d(n_v, L, f):
    n_h = n_hidden(n_v)
    x = KL.Input(shape=(L, n_v))
    h = KL.Reshape((L, n_v, 1))(x)
    h = KL.Conv2D(32, (3, 3), padding="same", activation="tanh", kernel_initializer=_init(f))(h)
    h = KL.MaxPooling2D(pool_size=(2, 2), padding="same")(h)
    if "A" in f:
        h = KL.LeakyReLU(alpha=0.01)(KL.Conv2D(n_h, (3, 3), padding="same", kernel_initializer=_init(f))(h))
    else:
        h = KL.Conv2D(n_h, (3, 3), padding="same", activation="relu", kernel_initializer=_init(f))(h)
    g = KL.GlobalAveragePooling2D()(h)
    if "R" in f:
        ult = KL.Flatten()(KL.Lambda(lambda t: t[:, -1, :, :], name="ultima_fila_temporal")(h))
        g = KL.Concatenate()([g, ult])
    return tf.keras.Model(x, _densa(g, n_h, f), name="encoder")


def _enc_mha(n_v, L, f):
    n_h = n_hidden(n_v)
    modo = {"training": False} if "N" in f else {}
    x = KL.Input(shape=(L, n_v))
    att = KL.MultiHeadAttention(num_heads=n_h, key_dim=n_v)(x, x)
    n1 = KL.BatchNormalization()(att + x, **modo)
    d = KL.Dense(n_v, activation="tanh")(n1)
    n2 = KL.BatchNormalization()(d + n1, **modo)
    g = KL.GlobalAveragePooling1D()(n2)
    if "R" in f:
        g = KL.Concatenate()([g, KL.Lambda(lambda t: t[:, -1, :], name="ultima_posicion")(n2)])
    return tf.keras.Model(x, _densa(g, n_h, f), name="encoder")


_ENC = {"conv1d": _enc_conv1d, "conv2d": _enc_conv2d, "mha": _enc_mha}


class PasoPorCapa(tf.keras.Model):
    """Factor P. Un Adam por grupo de fan_in: lr = lr0 * min(1, n_v / fan_in). Los sesgos siguen a su nucleo."""

    def __init__(self, encoder, decoder, lr0, n_visible, **kw):
        super().__init__(**kw)
        self.encoder, self.decoder = encoder, decoder
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self._lr0, self._n_v = float(lr0), int(n_visible)
        self._grupos = None
        self.lr_map = {}

    @staticmethod
    def _fan_in(v):
        sh = v.shape.as_list()
        return int(np.prod(sh[:-1])) if len(sh) >= 2 else None

    def _construye(self):
        grupos = {}
        for mdl in (self.encoder, self.decoder):
            ult = None
            for v in mdl.trainable_variables:
                fi = self._fan_in(v)
                if fi is None:
                    fi = ult if ult else self._n_v
                else:
                    ult = fi
                lr = self._lr0 * min(1.0, self._n_v / float(max(fi, 1)))
                grupos.setdefault(round(lr, 12), []).append(v)
                self.lr_map[v.name] = {"fan_in": fi, "lr": lr}
        self._grupos = [(tf.keras.optimizers.Adam(learning_rate=lr), vs) for lr, vs in sorted(grupos.items())]

    def call(self, x, training=False):
        return self.decoder(self.encoder(x, training=training), training=training)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        x, y = data
        if self._grupos is None:
            self._construye()
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(y, self(x, training=True))
        for opt, vs in self._grupos:
            opt.apply_gradients(zip(tape.gradient(loss, vs), vs))
        del tape
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def construir(familia, n_v, L, factores, lr0=LR0):
    """Devuelve (modelo compilado, encoder). `factores` es un conjunto de letras de FACTORES[familia]."""
    f = frozenset(factores)
    if not f <= FACTORES[familia]:
        raise ValueError(f"factores {sorted(f - FACTORES[familia])} no definidos para {familia}")
    enc, dec = _ENC[familia](n_v, L, f), _decoder(n_v, n_hidden(n_v), f)
    if "P" in f:
        m = PasoPorCapa(enc, dec, lr0, n_v, name=f"{familia}_rep")
        m.compile()
    else:
        m = tf.keras.models.Sequential([enc, dec], name=f"{familia}_rep")
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr0), loss=tf.keras.losses.MeanSquaredError())
    m.build(input_shape=(None, L, n_v))
    return m, enc


def construir_referencia(n_v, lr0=LR0):
    """Referencia por registro (control positivo): n_v -> n_h ReLU -> n_v sigmoide, como multilayer.py."""
    n_h = n_hidden(n_v)
    enc = tf.keras.Sequential([KL.Input(shape=(n_v,)), KL.Dense(n_h, activation="relu")], name="encoder")
    m = tf.keras.Sequential([enc, KL.Dense(n_v, activation="sigmoid")], name="referencia")
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr0), loss=tf.keras.losses.MeanSquaredError())
    return m, enc


def _firma(modelo):
    """Formas de las variables (ordenadas) y clases de las capas con pesos: lo que define la arquitectura."""
    formas = sorted(tuple(v.shape.as_list()) for v in modelo.variables)

    def capas(m):
        out = []
        for c in m.layers:
            if isinstance(c, tf.keras.Model):
                out += capas(c)
            elif type(c).__name__ not in ("InputLayer",):
                out.append((type(c).__name__, c.get_config().get("activation"),
                            c.get_config().get("kernel_size"), c.get_config().get("padding")))
        return out
    return formas, capas(modelo)


def comprobar_identidad(L=800, tamanos=(8, 34, 82)):
    """Con los factores apagados, el constructor nuevo debe coincidir con la clase original de src/."""
    from src.models.networks.conv1d import Conv1DAutoencoder
    from src.models.networks.conv2d import Conv2DAutoencoder
    from src.models.networks.mha import MHAAutoencoder
    orig = {"conv1d": Conv1DAutoencoder, "conv2d": Conv2DAutoencoder, "mha": MHAAutoencoder}
    res = []
    for fam, cls in orig.items():
        for n_v in tamanos:
            o = cls(n_visible=n_v, hidden_ratio=HIDDEN_RATIO, lr=LR0, sequence_length=L, ar=False, haste=True)
            m, _ = construir(fam, n_v, L, frozenset())
            fo, co = _firma(o.model)
            fm, cm = _firma(m)
            ok = (fo == fm) and (co == cm) and (o.model.count_params() == m.count_params())
            res.append({"familia": fam, "n_v": n_v, "ok": bool(ok), "params": int(m.count_params()),
                        "params_orig": int(o.model.count_params()),
                        "capas": [c[0] for c in cm], "capas_orig": [c[0] for c in co]})
            tf.keras.backend.clear_session()
    return res
