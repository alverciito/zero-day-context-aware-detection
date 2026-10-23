"""Labels, banned words and the seed B guard shared by the scripts of the experiment with modified encoders."""
import re

ETIQUETA = "[ENMIENDA-1] [EXPLORATORIO]"
ROTULO_POST_HOC = ("post hoc experiment, specified in an amendment to the analysis plan dated 7 October 2026, "
                   "after the pre-specified results were known and before the selection runs were made")
PROHIBIDAS = re.compile(r"\b(un)?trained\b|did not train|significant|outperform|consistently|confirmed|confirmatory|"
                        r"pre-?registered|\bdefect|\brepaired\b", re.IGNORECASE)
ULTIMA_DECISION_ABIERTA_B = 76
MARCADOR_CORTE = re.compile(r"CORTE\s+SEMILLA\s+B\s*:", re.IGNORECASE)


class GuardaSemillaB(RuntimeError):
    """La decision sobre la semilla B no esta tomada, no esta comprometida o no dice lo que hace falta."""


def parrafo_decision(decision: str | None, texto: str) -> str:
    """Paragraph of a committed decision, checked to be later than the one that left the cut-off of seed B open."""
    if not decision:
        raise GuardaSemillaB("hace falta --decision-semilla-b D-NN (decision sobre el corte de la semilla B "
                             "confirmada con commit); hasta entonces solo datos sinteticos")
    m = re.fullmatch(r"D-(\d+)", decision.strip())
    if not m:
        raise GuardaSemillaB("--decision-semilla-b %r no tiene la forma D-NN" % decision)
    if int(m.group(1)) <= ULTIMA_DECISION_ABIERTA_B:
        raise GuardaSemillaB("%s no vale: la decision %d dejo abierto el corte de la semilla B; hace falta una "
                             "decision posterior" % (decision, ULTIMA_DECISION_ABIERTA_B))
    ini = re.search(r"^- \*\*%s \(" % re.escape(decision.strip()), texto, re.MULTILINE)
    if not ini:
        raise GuardaSemillaB("%s no esta en el registro de decisiones de HEAD (sin commit no cuenta)" % decision)
    resto = texto[ini.start():]
    fin = re.search(r"\n- \*\*", resto[1:])
    return resto[:fin.start() + 1] if fin else resto


def comprobar_corte_semilla_b(decision: str | None, texto: str) -> dict:
    """Check that a later committed decision states the seed B cut-off after the marker 'CORTE SEMILLA B:'."""
    parrafo = parrafo_decision(decision, texto)
    limpio = parrafo.replace("*", "").replace("`", "")
    mc = MARCADOR_CORTE.search(limpio)
    if not mc:
        raise GuardaSemillaB("%s no lleva el marcador 'CORTE SEMILLA B: ...' (formato en comun_fase_b.py); hablar "
                             "de la semilla B no basta" % decision)
    corte = re.split(r"\.(\s|$)", limpio[mc.end():].strip(), maxsplit=1)[0].strip()
    if not corte:
        raise GuardaSemillaB("%s: el marcador 'CORTE SEMILLA B:' no dice cuando se corta" % decision)
    return {"decision": decision.strip(), "parrafo": parrafo, "corte": corte}


def palabras_prohibidas(texto: str | None) -> list:
    """Palabras prohibidas que aparecen en un texto en ingles generado.

    :param texto: texto (None cuenta como vacio).
    :type texto: str | None
    :return: lista de coincidencias.
    :rtype: list
    """
    return [m.group(0) for m in PROHIBIDAS.finditer(texto or "")]
