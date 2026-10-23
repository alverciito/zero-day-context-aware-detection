"""Variant `original_eps`: the KitNET autoencoder with the same stabiliser (1e-16) in training and in execution."""
import sys
from pathlib import Path

import numpy as np

_REPO = str(Path(__file__).resolve().parents[2])
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.models.networks.original_kitsune import DenoisingAutoencoder

__all__ = ["DenoisingAutoencoderEps16", "EPS_NETWORK_TYPES", "register_eps_variant", "EPS_TRAIN_ORIGINAL"]

EPS_TRAIN_ORIGINAL = 1e-6
EPS_EXEC = 1e-16


class DenoisingAutoencoderEps16(DenoisingAutoencoder):
    """El dA del envio con `EPS_TRAIN = 1e-16`. **Un solo caracter de diferencia**, y nada mas.

    `train()` se sobreescribe ENTERO porque en el original la normalizacion esta incrustada en el bucle y no hay
    ningun metodo al que engancharse. El cuerpo es copia literal de `original_kitsune.py:86-117`, con `1e-6`
    sustituido por `self.EPS_TRAIN`. `execute()`, `reconstruct()`, `forward_encode()`, `forward_decode()`,
    `dropout()`, `inGrace()` y el constructor se heredan sin tocar: el estabilizador de ejecucion ya era 1e-16.

    CUIDADO al mantener esto: `self.W_prime = self.W.T` (`original_kitsune.py:51`) es una **vista** de `W`, y el
    atado entre codificador y decodificador solo sobrevive porque la actualizacion es `self.W += ...` **in
    situ**. Si alguien reescribe esa linea como `self.W = self.W + ...`, `W_prime` se queda apuntando al array
    viejo y el decodificador se congela **en silencio**. Por eso el cuerpo se copia literal.
    """

    EPS_TRAIN = EPS_EXEC

    def train(self, input_x):
        """Copia literal de `DenoisingAutoencoder.train` (original_kitsune.py:86-117) salvo el estabilizador.

        :param input_x: tramo de entrenamiento.
        :return: RMSE de reconstruccion por fila durante el entrenamiento.
        """
        l_h2_list = list()
        for x in input_x:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]

            x = (x - self.norm_min) / (self.norm_max - self.norm_min + self.EPS_TRAIN)

            if self.corruption_level > 0.0:
                tilde_x = self.dropout(x, self.corruption_level)
            else:
                tilde_x = x
            y = self.forward_encode(tilde_x)
            z = self.forward_decode(y)

            L_h2 = x - z
            L_h1 = np.dot(L_h2, self.W) * y * (1 - y)

            L_vbias = L_h2
            L_hbias = L_h1
            L_W = np.outer(tilde_x.T, L_h1) + np.outer(L_h2.T, y)

            self.W += self.lr * L_W
            self.h_bias += self.lr * L_hbias
            self.v_bias += self.lr * L_vbias
            l_h2_list.append(L_h2)

        self.n += len(input_x)
        l_h2_array = np.array(l_h2_list)
        return np.sqrt(np.mean(l_h2_array**2, axis=1))


EPS_NETWORK_TYPES = {"original_eps": DenoisingAutoencoderEps16}


def register_eps_variant():
    """Inyecta `original_eps` en `src.models.oopsie.NETWORK_TYPES` EN SITIO, con las mismas guardas que
    `instrument.register_context_variants`. Idempotente. Devuelve el registro para `config.json`.

    Muta el dict en sitio a proposito: `src.models.kitnet` lo importa por referencia
    (`kitnet.NETWORK_TYPES is oopsie.NETWORK_TYPES`), asi que reasignarlo no tendria efecto.
    """
    from src.models import kitnet as _kitnet
    from src.models import oopsie as _oopsie

    if _kitnet.NETWORK_TYPES is not _oopsie.NETWORK_TYPES:
        raise RuntimeError("kitnet.NETWORK_TYPES ya no es el mismo objeto que oopsie.NETWORK_TYPES: "
                           "el registro en caliente no llegaria a __build_daes__")
    before = dict(_oopsie.NETWORK_TYPES)
    for key, cls in EPS_NETWORK_TYPES.items():
        cur = _oopsie.NETWORK_TYPES.get(key)
        if cur is not None and cur is not cls:
            raise RuntimeError(f"la clave {key!r} ya esta ocupada por {cur!r}")
        _oopsie.NETWORK_TYPES[key] = cls
    for key, cls in before.items():
        if _oopsie.NETWORK_TYPES[key] is not cls:
            raise RuntimeError(f"el registro ha pisado la clave original {key!r}")
    return {"module": __name__, "file": __file__, "added": sorted(EPS_NETWORK_TYPES),
            "eps_train": float(DenoisingAutoencoderEps16.EPS_TRAIN),
            "eps_train_submitted": EPS_TRAIN_ORIGINAL, "eps_exec": EPS_EXEC,
            "decision": "mismo estabilizador en las dos fases"}
