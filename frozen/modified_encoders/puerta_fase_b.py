#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry rule of the modified encoders: decides from the convergence reports of seed A whether each family enters."""
import argparse
import json
import os
import sys
import time

R = "/home/acid137/Projects/review-zd"
RESULTS = f"{R}/results/revision"
SALIDA = f"{R}/revision/analysis/reparacion/puerta_fase_b.json"
STREAMS = ["aci-iot-2023", "knad-active_wiretap", "knad-fuzzing", "knad-os_scan", "knad-ssdp_flood",
           "knad-ssl_renegotiation", "knad-video_injection", "cic2017", "cic2018"]
FAMILIAS = ["conv1d", "conv2d"]
UMBRAL, SUELO_MIN, CTX_MIN = 0.90, 1e-4, 1.10
K, STREAMS_MIN = 4, 7
PREVUELO = ("cic2018", "aci-iot-2023", "knad-active_wiretap")


def job_id(stream, familia, ar):
    return f"T10-REP__{stream}__{familia}_rep__ar{ar}__corrected.v1__L800__k4__sA__r1"


def finito(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def lee_job(root, stream, familia, ar):
    """Autoencoders of one job as its convergence report records them, or None and the reason when it cannot be used."""
    d = os.path.join(root, job_id(stream, familia, ar))
    if not os.path.exists(os.path.join(d, "DONE")):
        return None, "sin DONE"
    try:
        with open(os.path.join(d, "convergence_report.json"), encoding="utf-8") as fh:
            rep = json.load(fh)
    except (OSError, ValueError) as e:
        return None, f"convergence_report.json ilegible: {type(e).__name__}"
    if rep.get("familia") != familia or rep.get("paradigma") != ("AR" if ar == "T" else "TSR"):
        return None, "convergence_report.json no es de esta familia y paradigma"
    aes = []
    for a in rep.get("autoencoders") or []:
        suelo, razon = a.get("suelo"), a.get("razon")
        ev = finito(suelo) and suelo >= SUELO_MIN
        aes.append({"ae_index": a.get("ae_index"), "n_visible": a.get("n_visible"), "suelo": suelo, "razon": razon,
                    "evaluable": bool(ev), "entrenado": bool(ev and finito(razon) and razon < UMBRAL),
                    "cociente_contexto_cambiado": a.get("cociente_contexto_cambiado"),
                    "razon_persistencia": a.get("razon_persistencia"), "muertas": a.get("muertas"),
                    "filas_train": a.get("filas_train"), "filas_val": a.get("filas_val"),
                    "factores": rep.get("factores")})
    if len(aes) != K:
        return None, f"{len(aes)} autoencoders en convergence_report.json, se esperaban {K}"
    return aes, None


def regla(evaluables, entrenados, streams_con_entrenado):
    """Enmienda 4: 2/3 del denominador (aritmetica entera) y al menos 7 streams con algun AE entrenado."""
    return evaluables > 0 and 3 * entrenados >= 2 * evaluables and streams_con_entrenado >= STREAMS_MIN


def familia(root, fam):
    por_stream, faltan = {}, []
    ev = ent = n_str = 0
    ctx = 0
    factores = set()
    for s in STREAMS:
        aes, motivo = lee_job(root, s, fam, "F")
        if aes is None:
            faltan.append({"stream": s, "motivo": motivo})
            por_stream[s] = {"tsr": None, "motivo": motivo}
            continue
        e = [a for a in aes if a["evaluable"]]
        t = [a for a in e if a["entrenado"]]
        ev, ent, n_str = ev + len(e), ent + len(t), n_str + (1 if t else 0)
        ctx += sum(1 for a in t if finito(a["cociente_contexto_cambiado"]) and a["cociente_contexto_cambiado"] >= CTX_MIN)
        factores |= {str(a["factores"]) for a in aes}
        por_stream[s] = {"tsr": {"evaluables": len(e), "entrenados": len(t), "no_evaluables": K - len(e),
                                 "razon": [a["razon"] for a in aes], "suelo": [a["suelo"] for a in aes],
                                 "contexto": [a["cociente_contexto_cambiado"] for a in aes]}}
    ar_ev = ar_ent = ar_jobs = 0
    for s in STREAMS:
        aes, motivo = lee_job(root, s, fam, "T")
        if aes is None:
            por_stream[s]["ar"] = None
            continue
        ar_jobs += 1
        e = [a for a in aes if a["evaluable"]]
        ar_ev, ar_ent = ar_ev + len(e), ar_ent + sum(1 for a in e if a["entrenado"])
        por_stream[s]["ar"] = {"evaluables": len(e), "entrenados": sum(1 for a in e if a["entrenado"]),
                               "razon": [a["razon"] for a in aes],
                               "razon_persistencia": [a["razon_persistencia"] for a in aes],
                               "contexto": [a["cociente_contexto_cambiado"] for a in aes]}
    fuera = [por_stream[x]["tsr"] for x in STREAMS if x not in PREVUELO and por_stream[x].get("tsr")]
    fuera_prevuelo = {"streams_leidos": len(fuera), "streams_posibles": len(STREAMS) - len(PREVUELO),
                      "evaluables": sum(t["evaluables"] for t in fuera),
                      "entrenados": sum(t["entrenados"] for t in fuera),
                      "streams_con_algun_entrenado": sum(1 for t in fuera if t["entrenados"] > 0),
                      "nota": "descriptivo; no cambia la regla ni la decision."}
    nf = len(faltan)
    peor = regla(ev + K * nf, ent, n_str)
    mejor = regla(ev + K * nf, ent + K * nf, n_str + nf)
    if nf == 0:
        entra, estado = regla(ev, ent, n_str), "DECIDIDA"
    elif peor == mejor:
        entra, estado = peor, "DECIDIDA (determinada aunque faltan %d jobs TSR)" % nf
    else:
        entra, estado = None, "SIN DECIDIR: faltan %d jobs TSR y la decision depende de ellos" % nf
    if len(factores) > 1:
        entra, estado = None, "SIN DECIDIR: jobs con factores distintos %s" % sorted(factores)
    return {"familia": fam, "factores": sorted(factores)[0] if len(factores) == 1 else sorted(factores),
            "estado": estado, "entra": entra,
            "tsr": {"jobs_leidos": len(STREAMS) - nf, "jobs_que_faltan": faltan, "evaluables": ev, "entrenados": ent,
                    "fraccion_entrenados": (ent / ev) if ev else None, "minimo_exigido": "2/3",
                    "cumple_dos_tercios": bool(ev > 0 and 3 * ent >= 2 * ev) if nf == 0 else None,
                    "streams_con_algun_entrenado": n_str, "streams_minimos": STREAMS_MIN,
                    "cumple_streams": bool(n_str >= STREAMS_MIN) if nf == 0 else None,
                    "entrenados_que_dependen_del_contexto": ctx,
                    "nota_contexto": "enmienda 6: la frase sobre el contexto exige que al menos la mitad de los AE "
                                     "TSR entrenados tengan cociente >= 1,10; es redaccion, no puerta"},
            "fuera_de_los_streams_del_prevuelo": fuera_prevuelo,
            "ar_descriptivo": {"jobs_leidos": ar_jobs, "evaluables": ar_ev, "entrenados": ar_ent,
                               "nota": "AR no es puerta (enmienda 3.2)"},
            "texto_si_no_entra": "la modificacion no generaliza fuera de las celdas del prevuelo (se reporta sin test)",
            "por_stream": por_stream}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--familias", default=",".join(FAMILIAS))
    ap.add_argument("--results-root", default=RESULTS)
    ap.add_argument("--salida", default=SALIDA)
    a = ap.parse_args()
    fams = [f for f in a.familias.split(",") if f]
    if any(f not in FAMILIAS for f in fams):
        print("familias validas:", FAMILIAS)
        return 1
    if os.path.abspath(a.results_root) != os.path.abspath(RESULTS) and os.path.abspath(a.salida) == os.path.abspath(SALIDA):
        print("ABORT: con otro --results-root hay que dar otra --salida (no se pisa el fichero oficial)")
        return 1
    out = {"schema": "jisa-t10-rep-puerta/1", "escrito": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "enmienda": "analysis-plan-enmienda-1", "decision": "alcance: semilla A",
           "results_root": os.path.abspath(a.results_root),
           "regla": {"evaluable": f"suelo >= {SUELO_MIN}", "entrenado": f"razon < {UMBRAL}",
                     "denominador": "autoencoders TSR evaluables de la semilla A en los 9 streams",
                     "entra": f"entrenados >= 2/3 del denominador Y >= {STREAMS_MIN} de 9 streams con algun entrenado",
                     "usa_etiquetas": False, "metricas_de_deteccion": "ninguna"},
           "aviso": "anotar la decision en el registro de decisiones y confirmarla con commit ANTES de ejecutar "
                    "metrics_rev sobre ningun job de T10 (enmienda 4)",
           "familias": {f: familia(a.results_root, f) for f in fams}}
    leidos = sum(v["tsr"]["jobs_leidos"] for v in out["familias"].values())
    tmp = a.salida + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, a.salida)
    for f, v in out["familias"].items():
        t = v["tsr"]
        frac = "%.3f" % t["fraccion_entrenados"] if t["fraccion_entrenados"] is not None else "-"
        print(f"{f:7s} [{v['factores']}] {v['estado']}: entra = {v['entra']} | TSR: {t['entrenados']}/{t['evaluables']} "
              f"evaluables entrenados ({frac}; minimo 2/3) | streams con algun entrenado: "
              f"{t['streams_con_algun_entrenado']}/9 (minimo {STREAMS_MIN}) | jobs TSR leidos {t['jobs_leidos']}/9 | "
              f"AR (descriptivo): {v['ar_descriptivo']['entrenados']}/{v['ar_descriptivo']['evaluables']} | "
              f"fuera de los 3 streams del prevuelo (descriptivo): {v['fuera_de_los_streams_del_prevuelo']['entrenados']}/"
              f"{v['fuera_de_los_streams_del_prevuelo']['evaluables']} entrenados, "
              f"{v['fuera_de_los_streams_del_prevuelo']['streams_con_algun_entrenado']}/"
              f"{v['fuera_de_los_streams_del_prevuelo']['streams_posibles']} streams")
    print("->", a.salida)
    if leidos == 0:
        return 1
    return 3 if any(v["entra"] is None for v in out["familias"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
