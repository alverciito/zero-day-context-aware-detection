#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drives the selection runs of the amendment to the analysis plan and picks the first passing step per family."""
import argparse
import json
import os
import subprocess
import sys
import time

R = "/home/acid137/Projects/review-zd"
AQUI = f"{R}/revision/analysis/reparacion"
REG = f"{AQUI}/registro_intentos.jsonl"
RESUMEN = f"{AQUI}/resumen_prevuelo.json"
PY = [f"{R}/envs/tf215/bin/python", f"{AQUI}/prevuelo.py"]
ENV = dict(os.environ, LD_LIBRARY_PATH="", PYTHONPATH=R)
TAG = "analysis-plan-enmienda-1"
FIJADO = ["revision/analysis/reparacion/modelos_rep.py", "revision/analysis/reparacion/prevuelo.py",
          "revision/analysis/reparacion/escalera.py", "revision/analysis/reparacion/cola_prevuelo.sh",
          "revision/analysis/analysis_plan_enmienda_1.md"]
STREAMS = ["cic2018", "aci-iot-2023", "knad-active_wiretap"]
DOCE = [(s, g) for s in STREAMS for g in range(4)]
ORDEN = ["mha", "conv1d", "conv2d"]
ESCALERA = {
    "mha": ["N", "N,R", "N,R,A"],
    "conv1d": ["P", "P,I", "P,I,R", "P,I,R,A"],
    "conv2d": ["P", "P,I", "P,I,R", "P,I,R,A"],
}
SEMILLAS_REF = (1, 2, 3)
UMBRAL, SUELO_MIN, E_MIN = 0.90, 1e-4, 4


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), *a, flush=True)


def lee(reg):
    if not os.path.exists(reg):
        return []
    return [json.loads(l) for l in open(reg) if l.strip()]


def anota(reg, rec):
    rec = dict(rec, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(reg, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def celdas(reg, fase, modelo, factores, semilla=None):
    """Ultima anotacion de cada celda para (fase, modelo, factores[, semilla]). Incluye las que tienen error."""
    out = {}
    for r in lee(reg):
        if (r.get("tipo") == "celda" and not r.get("smoke") and r["fase"] == fase and r["modelo"] == modelo
                and r["factores"] == factores and (semilla is None or r.get("semilla") == semilla)):
            out[(r["stream"], r["grupo"])] = r
    return out


def completo(recs, cs):
    return all(c in recs and not recs[c].get("error") for c in cs)


def pasa(r):
    return r is not None and not r.get("error") and r.get("razon") is not None and r["razon"] < UMBRAL


def ref_evaluable(reg, fase, c):
    """Enmienda 3.2: suelo >= 1e-4 y la referencia por registro baja de 0,90 en alguna de las tres semillas."""
    rs = [celdas(reg, fase, "ref", "", s).get(c) for s in SEMILLAS_REF]
    rs = [r for r in rs if r is not None and not r.get("error")]
    return bool(rs) and rs[0].get("suelo", 0.0) >= SUELO_MIN and any(pasa(r) for r in rs)


def regla(recs, E):
    """Enmienda 3.4. Devuelve (veredicto, celdas que pasan). Exige que todas las celdas de E esten anotadas."""
    E = sorted(E)
    if len(E) < E_MIN:
        return "no_informativo", []
    ok = [c for c in E if pasa(recs.get(c))]
    falta = -(-2 * len(E) // 3)
    por_stream = all(any(c[0] == s for c in ok) for s in {c[0] for c in E})
    return ("pasa" if (len(ok) >= falta and por_stream) else "no_pasa"), ok


def txt(cs):
    return ",".join(f"{s}:{g}" for s, g in sorted(cs))


def lanza(reg, fase, modelo, factores, escalon, cs, semilla=None):
    """Ejecuta prevuelo.py sobre las celdas `cs` (None = las 12). Devuelve 'ok', 'tope' o 'error'.
    'error' = tras un reintento, el proceso no acabo bien o alguna celda falta o tiene error."""
    if cs is not None and len(cs) == 0:
        raise ValueError("lanza() sin celdas: nunca se ejecuta el conjunto vacio")
    objetivo = list(cs) if cs is not None else DOCE
    for intento in (1, 2):
        cmd = PY + ["--fase", fase, "--modelo", modelo, "--factores", factores, "--escalon", str(escalon),
                    "--registro", reg, "--celdas", txt(objetivo)] + (["--semilla", str(semilla)] if semilla else [])
        log("LANZA", " ".join(cmd[1:]))
        rc = subprocess.run(cmd, env=ENV, cwd=R).returncode
        if rc == 3:
            return "tope"
        if rc == 0 and completo(celdas(reg, fase, modelo, factores, semilla), objetivo):
            return "ok"
        mal = [c for c in objetivo if not completo(celdas(reg, fase, modelo, factores, semilla), [c])]
        log(f"  intento {intento}: rc={rc}, celdas con error o sin anotar: {txt(mal) or '-'}")
    return "error"


def lanza_ref(reg, fase, cs):
    for s in SEMILLAS_REF:
        st = lanza(reg, fase, "ref", "", "-", cs, semilla=s)
        if st != "ok":
            return st
    return "ok"


def elige_celdas(reg):
    """Enmienda 3.2: por stream, entre los grupos evaluables en la criba, el de mas features y el de menos."""
    nv = {c: r["n_v"] for c, r in celdas(reg, "criba", "ref", "").items() if "n_v" in r}
    elegidas = []
    for s in STREAMS:
        ev = sorted((nv[c], c[1]) for c in DOCE if c[0] == s and c in nv and ref_evaluable(reg, "criba", c))
        if not ev:
            continue
        mayor = min(g for n, g in ev if n == ev[-1][0])
        menor = min(g for n, g in ev if n == ev[0][0])
        elegidas.append((s, mayor))
        if menor != mayor:
            elegidas.append((s, menor))
    return elegidas


def puerta_git():
    def git(*a):
        return subprocess.run(["git", "-C", R] + list(a), capture_output=True, text=True)
    if not git("tag", "-l", TAG).stdout.strip():
        return f"falta el tag {TAG}"
    if git("status", "--porcelain", "--", *FIJADO).stdout.strip():
        return "hay cambios sin commit en la enmienda o en el arnes"
    if git("diff", "--quiet", TAG, "HEAD", "--", *FIJADO).returncode != 0:
        return f"el codigo de HEAD no es el fijado por el tag {TAG}"
    return None


def _t(ts):
    return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))


def gastado(reg):
    """La misma cuenta que prevuelo.gastado() (copiada: aqui no se importa TensorFlow)."""
    rs = lee(reg)

    def ok(fase, con_ref):
        return [float(r["segundos"]) for r in rs if r.get("tipo") == "celda" and not r.get("error")
                and r.get("fase") == fase and (con_ref or r.get("modelo") != "ref")]

    def techo(fase, modelo):
        con_ref = (modelo == "ref")
        mias, criba = ok(fase, con_ref), ok("criba", con_ref)
        if fase == "confirma":
            e = max(max(mias) if mias else 0.0, 7.5 * max(criba) if criba else 0.0)
        else:
            e = max(mias) if mias else 0.0
        return e or 3600.0
    cerradas = {r.get("uid") for r in rs if r.get("tipo") == "celda"}
    tot = 0.0
    for i, r in enumerate(rs):
        if r.get("tipo") in ("celda", "proceso", "identidad"):
            tot += float(r.get("segundos", 0.0))
        elif r.get("tipo") == "inicio" and r.get("uid") not in cerradas:
            fin = _t(rs[i + 1]["ts"]) if i + 1 < len(rs) and rs[i + 1].get("ts") else time.time()
            tot += min(max(0.0, fin - _t(r["ts"])), techo(r.get("fase"), r.get("modelo")))
    return tot


def resumen(reg, estado):
    rs = lee(reg)
    cerradas = {r.get("uid") for r in rs if r.get("tipo") == "celda"}
    gast = gastado(reg)
    tabla = {}
    for r in rs:
        if r.get("tipo") == "celda" and not r.get("smoke"):
            k = f"{r['fase']}|{r['modelo']}|{r['factores'] or '-'}|s{r.get('semilla')}"
            tabla.setdefault(k, {})[f"{r['stream']}:{r['grupo']}"] = {
                "razon": r.get("razon"), "ctx": r.get("cociente_contexto_cambiado"),
                "muertas": r.get("muertas"), "s": r.get("segundos"), "error": r.get("error")}
    out = {"escrito": time.strftime("%Y-%m-%dT%H:%M:%S"), "enmienda": TAG,
           "gpu_segundos": round(gast, 1), "gpu_horas": round(gast / 3600, 2), "tope_horas": 8,
           "celdas_sin_cerrar": sum(1 for r in rs if r.get("tipo") == "inicio" and r.get("uid") not in cerradas),
           "estado": estado, "decisiones": [r for r in rs if r.get("tipo") == "decision"], "tabla": tabla}
    json.dump(out, open(RESUMEN, "w"), indent=1, ensure_ascii=False)
    log(f"RESUMEN -> {RESUMEN}  (GPU {out['gpu_horas']} h de 8)")
    for fam, e in estado.get("familias", {}).items():
        log(f"  {fam:7s}: {e}")


def decision_familia(reg, fam):
    ya = [r for r in lee(reg) if r.get("tipo") == "decision" and r.get("clase") == "familia"
          and r.get("familia") == fam]
    return ya[-1]["resultado"] if ya else None


def estado_familia(reg, fam):
    """Lo que dice el registro de una familia: su ultima decision y, si la hay, la del control negativo completo."""
    res = decision_familia(reg, fam)
    if not res or not res.startswith("ELEGIDO"):
        return res
    cn = [r for r in lee(reg) if r.get("tipo") == "decision" and r.get("clase") == "control_negativo_confirma"
          and r.get("familia") == fam]
    if not cn:
        return res
    v = cn[-1]["veredicto"]
    return res + (" | control negativo completo: falla, como debe" if v == "no_pasa"
                  else f" | control negativo completo: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registro", default=REG)
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--sin-git", action="store_true", help="solo pruebas: no exige el tag ni el arbol limpio")
    ap.add_argument("--puerta", action="store_true", help="solo comprueba la puerta de git y sale (0 o 2)")
    a = ap.parse_args()
    reg = a.registro
    if a.puerta:
        motivo = puerta_git()
        log("puerta de git:", motivo or "OK")
        return 2 if motivo else 0
    estado = {"familias": {}}
    if a.resumen:
        for fam in ORDEN:
            if estado_familia(reg, fam):
                estado["familias"][fam] = estado_familia(reg, fam)
        resumen(reg, estado)
        return 0

    if not a.sin_git:
        motivo = puerta_git()
        if motivo:
            log("ABORT: la enmienda y el prevuelo tienen que estar fijados antes de gastar GPU:", motivo)
            return 2
    commit = subprocess.run(["git", "-C", R, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    anota(reg, {"tipo": "decision", "clase": "arranque", "commit": commit})

    if not any(r.get("tipo") == "identidad" and r.get("ok") and not r.get("smoke") for r in lee(reg)):
        rc = subprocess.run(PY + ["--identidad", "--registro", reg], env=ENV, cwd=R).returncode
        if rc != 0:
            anota(reg, {"tipo": "decision", "clase": "identidad", "resultado": "FALLA: el constructor no coincide"})
            log("ABORT: comprobacion de identidad")
            return 4

    st = lanza_ref(reg, "criba", None)
    if st != "ok":
        log("ABORT: referencia de criba:", st)
        resumen(reg, estado)
        return 3 if st == "tope" else 5
    E_c = elige_celdas(reg)
    anota(reg, {"tipo": "decision", "clase": "celdas", "elegidas": txt(E_c),
                "evaluables_criba": txt([c for c in DOCE if ref_evaluable(reg, "criba", c)]),
                "no_evaluables_criba": txt([c for c in DOCE if not ref_evaluable(reg, "criba", c)])})
    log("celdas del prevuelo:", txt(E_c))
    if len(E_c) < E_MIN:
        anota(reg, {"tipo": "decision", "clase": "prevuelo", "resultado": "no informativo: menos de 4 celdas en la criba"})
        resumen(reg, estado)
        return 6

    st = lanza_ref(reg, "confirma", E_c)
    if st != "ok":
        log("ABORT: referencia de confirmacion:", st)
        resumen(reg, estado)
        return 3 if st == "tope" else 5
    E_f = [c for c in E_c if ref_evaluable(reg, "confirma", c)]
    anota(reg, {"tipo": "decision", "clase": "evaluables", "criba": txt(E_c), "confirma": txt(E_f)})
    if len(E_f) < E_MIN:
        anota(reg, {"tipo": "decision", "clase": "prevuelo",
                    "resultado": "no informativo: menos de 4 celdas evaluables en la confirmacion"})
        log("PARADA: prevuelo no informativo en confirmacion; ninguna familia se rotula")
        resumen(reg, estado)
        return 6

    tope, errores = False, False
    for fam in ORDEN:
        if decision_familia(reg, fam):
            estado["familias"][fam] = estado_familia(reg, fam)
            continue
        if tope:
            estado["familias"][fam] = "sin ejecutar (tope de GPU)"
            continue
        st = lanza(reg, "criba", fam, "ORIG", 0, E_c)
        if st == "tope":
            tope = True
            estado["familias"][fam] = "sin ejecutar (tope de GPU)"
            continue
        if st == "error":
            errores = True
            estado["familias"][fam] = "no concluyente por error de ejecucion (control negativo)"
            anota(reg, {"tipo": "decision", "clase": "control_negativo", "familia": fam,
                        "veredicto": "no ejecutable", "E": txt(E_c)})
            continue
        v, ok = regla(celdas(reg, "criba", fam, "ORIG"), E_c)
        anota(reg, {"tipo": "decision", "clase": "control_negativo", "familia": fam, "veredicto": v,
                    "pasan": txt(ok), "E": txt(E_c)})
        if v == "pasa":
            res = "PARADA: la clase original pasa la criba; el prevuelo no reproduce el defecto"
            anota(reg, {"tipo": "decision", "clase": "familia", "familia": fam, "resultado": res})
            estado["familias"][fam] = res
            continue
        elegido, ult_confirmado, roto = None, False, False
        esc = ESCALERA[fam]

        def confirma(i, fac, nota="confirma"):
            """Devuelve 'pasa', 'no_pasa', 'tope' o 'error' y lo anota."""
            st = lanza(reg, "confirma", fam, fac, i, E_f)
            if st != "ok":
                return st
            v, ok = regla(celdas(reg, "confirma", fam, fac), E_f)
            anota(reg, {"tipo": "decision", "clase": "escalon", "familia": fam, "escalon": i, "factores": fac,
                        "fase": nota, "veredicto": v, "pasan": txt(ok), "E": txt(E_f)})
            log(f"{fam} escalon {i} [{fac}] {nota}: {v} ({len(ok)}/{len(E_f)})")
            return v

        for i, fac in enumerate(esc, start=1):
            st = lanza(reg, "criba", fam, fac, i, E_c)
            if st in ("tope", "error"):
                tope, roto = (st == "tope"), (st == "error")
                break
            v, ok = regla(celdas(reg, "criba", fam, fac), E_c)
            anota(reg, {"tipo": "decision", "clase": "escalon", "familia": fam, "escalon": i, "factores": fac,
                        "fase": "criba", "veredicto": v, "pasan": txt(ok), "E": txt(E_c)})
            log(f"{fam} escalon {i} [{fac}] criba: {v} ({len(ok)}/{len(E_c)})")
            if v != "pasa":
                continue
            v = confirma(i, fac)
            if v in ("tope", "error"):
                tope, roto = (v == "tope"), (v == "error")
                break
            ult_confirmado = ult_confirmado or (i == len(esc))
            if v == "pasa":
                elegido = (i, fac)
                break
        if elegido is None and not tope and not roto and not ult_confirmado:
            v = confirma(len(esc), esc[-1], "confirma (sin criba, 3.5 paso 3)")
            if v in ("tope", "error"):
                tope, roto = (v == "tope"), (v == "error")
            elif v == "pasa":
                elegido = (len(esc), esc[-1])
        if elegido:
            res = f"ELEGIDO escalon {elegido[0]}: {elegido[1]}"
            anota(reg, {"tipo": "decision", "clase": "familia", "familia": fam, "resultado": res})
        elif roto:
            errores = True
            res = "no concluyente por error de ejecucion (sin decision; se puede reanudar)"
        elif tope:
            res = "incompleta (tope de GPU)"
        else:
            res = "NINGUN ESCALON PASA con la escalera de la enmienda al plan de analisis"
            anota(reg, {"tipo": "decision", "clase": "familia", "familia": fam, "resultado": res})
        estado["familias"][fam] = res
        log(f"== {fam}: {res}")

    for fam in ORDEN:
        res = decision_familia(reg, fam) or ""
        if not res.startswith("ELEGIDO"):
            continue
        ya = [r for r in lee(reg) if r.get("tipo") == "decision" and r.get("clase") == "control_negativo_confirma"
              and r.get("familia") == fam and r.get("veredicto") in ("pasa", "no_pasa")]
        if ya:
            estado["familias"][fam] = estado_familia(reg, fam)
            continue
        if tope:
            anota(reg, {"tipo": "decision", "clase": "control_negativo_confirma", "familia": fam,
                        "veredicto": "no ejecutado (tope de GPU)"})
            estado["familias"][fam] = estado_familia(reg, fam)
            continue
        st = lanza(reg, "confirma", fam, "ORIG", 0, E_f)
        if st != "ok":
            tope, errores = tope or st == "tope", errores or st == "error"
            anota(reg, {"tipo": "decision", "clase": "control_negativo_confirma", "familia": fam,
                        "veredicto": "no ejecutado (tope de GPU)" if st == "tope" else "no ejecutable"})
            estado["familias"][fam] = estado_familia(reg, fam)
            continue
        v, ok = regla(celdas(reg, "confirma", fam, "ORIG"), E_f)
        anota(reg, {"tipo": "decision", "clase": "control_negativo_confirma", "familia": fam,
                    "veredicto": v, "pasan": txt(ok), "E": txt(E_f)})
        if v == "pasa":
            anota(reg, {"tipo": "decision", "clase": "familia", "familia": fam,
                        "resultado": "NO INTERPRETABLE: el original entrena con el presupuesto completo "
                                     f"(era {res})"})
        estado["familias"][fam] = estado_familia(reg, fam)
        log(f"== {fam}: {estado['familias'][fam]}")
    resumen(reg, estado)
    return 3 if tope else (7 if errores else 0)


if __name__ == "__main__":
    sys.exit(main())
