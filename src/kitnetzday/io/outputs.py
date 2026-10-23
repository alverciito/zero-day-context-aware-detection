# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                           #
#   Project:      kitnetzday                                                #
#   File:         src/kitnetzday/io/outputs.py                              #
#   Version:      2.0.0                                                     #
#   Author:       Alberto Palomo Alonso                                     #
#   Affiliation:  Universidad de Alcalá - Escuela Politécnica Superior      #
#   Created:      2026-10-06                                                #
#   Modified:     2026-10-13                                                #
#   Description:  Output directory of the paper: writes every figure and    #
#                 table and keeps the record of what was produced.          #
#                                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import hashlib
import json
import logging
import math
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kitnetzday.naming import PLAN_TAGS
from kitnetzday.utils.formatting import ids_comment
from kitnetzday.utils.loggable import Loggable

_HASH_CHUNK = 1 << 22
# Script named in the provenance line of every generated .tex when the
# caller does not give another one.
DEFAULT_GENERATOR = "bin/make_figures.py"
# The PDF of a figure is written without its creation date, so that two
# runs over the same inputs give the same file.
PDF_METADATA: dict[str, Any] = {"CreationDate": None}


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                AUXILIARY                                  #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def _sha256_file(path: str | Path) -> str:
    """
    Hash a file, reading it in 4 MiB chunks.

    :param path: Path of an existing, readable file.
    :type path: str | Path
    :return: The sha256 digest as 64 lower-case hexadecimal characters.
    :rtype: str
    :raises OSError: If the file cannot be read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _write_text_atomic(path: str | Path, text: str) -> None:
    """
    Write a text file in one step, so a reader never sees it half done.

    The text goes to a temporary file in the same directory, is
    flushed and fsynced, and only then replaces the target, which is
    atomic. Its name carries the process and thread ids.

    :param path: Path of the file to write.
    :type path: str | Path
    :param text: Whole content of the file.
    :type text: str
    :raises OSError: If the file cannot be written or replaced.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}."
                         f"{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    """
    Last resort serializer for objects json does not know.

    :param o: Object json could not serialize.
    :type o: Any
    :return: A stand-in json accepts: the scalar of a numpy value,
        the list of an array or set, the path as a string, or str(o).
    :rtype: Any
    """
    try:
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _clean(o: Any) -> Any:
    """
    Make a nested structure strictly JSON-valid.

    NaN and the infinities become ``null``, because ``NaN`` is not
    JSON; numpy scalars and arrays, paths and keys become plain Python.

    :param o: Structure to clean, usually a manifest.
    :type o: Any
    :return: The same structure with JSON-valid leaves.
    :rtype: Any
    """
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Path):
        return str(o)
    return o


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                MAIN CLASS                                 #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Out(Loggable):
    """
    The output directory of the paper, and the record of what is in it.

    Every figure leaves three files behind (a PDF for the build, a PNG
    to look at on screen and a self-contained ``figure*``
    environment), and every table a ``.tex`` and an optional Markdown
    preview. All of them carry, as LaTeX comments, the script that
    produced them, the label of the analysis plan, the partial-inputs
    label when the run did not read the final aggregate, and the ids of
    the values they print.

    The class keeps the two registers the reports read afterwards:
    ``records``, one entry per output, and ``numbers``, the values
    listed in ``numbers_used.md``. The caller builds the figures and
    the script in ``bin/`` must select the non-interactive matplotlib
    backend ("Agg") first.

    :param out_dir: Directory the outputs go to; it is created.
    :type out_dir: str | Path
    :param dpi: Resolution of the PNG previews, 300 in the paper; the
        PDF is vector and ignores it.
    :type dpi: int
    :param cmd: Command that reproduces this run, for the README.
    :type cmd: str
    :param script_ref: Script and its hash, for the "Script" column of
        ``numbers_used.md``.
    :type script_ref: str
    :param cmd_rows: Command for the "Command to regenerate" column,
        when it differs from cmd; None reuses cmd.
    :type cmd_rows: str | None
    :param generator: Generating script of the provenance comment.
    :type generator: str
    :param partial_tag: Label of an output built from partial inputs.
    :type partial_tag: str
    :param root: Root of the package; paths inside it are shown
        relative to it in the reports. None shows every path as given.
    :type root: str | Path | None
    :param logger: Logger of the calling script, or None.
    :type logger: logging.Logger | None
    :raises TypeError: If logger is not a logging.Logger or None.
    :raises ValueError: If dpi is not a positive integer.
    :raises OSError: If the output directory cannot be created.
    """

    def __init__(self,
                 out_dir: str | Path,
                 dpi: int,
                 cmd: str,
                 script_ref: str,
                 cmd_rows: str | None = None,
                 generator: str = DEFAULT_GENERATOR,
                 partial_tag: str = PLAN_TAGS["partial"],
                 root: str | Path | None = None,
                 logger: logging.Logger | None = None) -> None:
        """
        Store the settings and create the output directory.

        :param out_dir: Directory the outputs go to.
        :type out_dir: str | Path
        :param dpi: Resolution of the PNG previews; positive.
        :type dpi: int
        :param cmd: Command that reproduces this run.
        :type cmd: str
        :param script_ref: Script and hash of this run.
        :type script_ref: str
        :param cmd_rows: Command for the rows of values, or None.
        :type cmd_rows: str | None
        :param generator: Generating script of the provenance comment.
        :type generator: str
        :param partial_tag: Label of an output built from partial
            inputs.
        :type partial_tag: str
        :param root: Root of the package, or None.
        :type root: str | Path | None
        :param logger: Logger created by the calling script, or None.
        :type logger: logging.Logger | None
        :raises TypeError: If logger is not a logging.Logger or None.
        :raises ValueError: If dpi is not a positive integer.
        :raises OSError: If the directory cannot be created.
        """
        super().__init__(logger=logger)
        if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
            self._error("[Out.__init__] Input error: the parameter dpi "
                        "must be a positive integer number of dots per "
                        f"inch, not {dpi!r}.")
        self.dir = Path(out_dir)
        self.dpi = dpi
        self.cmd = cmd
        self.cmd_rows = cmd_rows or cmd
        self.script_ref = script_ref
        self.generator = generator
        self.partial_tag = partial_tag
        self.root = None if root is None else Path(root).resolve()
        self.records: list[dict[str, Any]] = list()
        self.numbers: dict[str, tuple[Any, str]] = dict()
        self.dir.mkdir(parents=True, exist_ok=True)

    def shown(self, path: str | Path) -> str:
        """
        Write a path the way the reports and the manifest show it.

        A path inside the package root is written relative to it, with
        forward slashes, so that the generated files do not depend on
        where the package was unpacked; any other path is left as it
        was given.

        :param path: Path to show.
        :type path: str | Path
        :return: The relative path, or the path as given.
        :rtype: str
        """
        if self.root is None:
            return str(path)
        try:
            return Path(path).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    # ----- values used ----------------------------------------------- #
    def add_number(self,
                   idd: str,
                   value: Any,
                   pointer: str,
                   tag: str,
                   partial: bool) -> str:
        """
        Register one value as a row of ``numbers_used.md``.

        A value of None is dropped on purpose: a hole in the
        aggregate leaves no row. A value read from partial inputs is
        recorded with the partial-inputs label in front of its source.

        :param idd: Id of the row, unique inside one run.
        :type idd: str
        :param value: Value of the row, normally a float.
        :type value: Any
        :param pointer: Where the value comes from, as a pointer.
        :type pointer: str
        :param tag: Label of the analysis plan for this value.
        :type tag: str
        :param partial: Whether this run reads partial inputs.
        :type partial: bool
        :return: The id, so the caller can collect it in one expression.
        :rtype: str
        :raises ValueError: If idd is not a non-empty string.
        """
        pre = ""
        if not isinstance(idd, str) or not idd:
            self._error("[Out.add_number] Input error: the parameter "
                        f"idd must be a non-empty string, not {idd!r}.")
        if value is None:
            return idd
        pre = (self.partial_tag + " ") if partial else ""
        self.numbers[idd] = (value, f"{pre}{tag} {pointer}")
        return idd

    # ----- figures and tables ---------------------------------------- #
    def save_figure(self,
                    fig: Figure,
                    name: str,
                    caption: str,
                    label: str,
                    tag: str,
                    ids: Sequence[str],
                    partial: bool,
                    notes: Mapping[str, Any],
                    width: str = r"\textwidth",
                    env: str = "figure*") -> None:
        """
        Write one figure as PDF, PNG and a self-contained LaTeX block.

        The figure is closed afterwards, to free its memory.

        :param fig: Finished matplotlib figure.
        :type fig: Figure
        :param name: Base name of the files, without extension.
        :type name: str
        :param caption: Caption in academic English, already escaped.
        :type caption: str
        :param label: LaTeX label, for example "fig:main-heatmap".
        :type label: str
        :param tag: Label of the analysis plan for this output.
        :type tag: str
        :param ids: Ids of the values used by the figure.
        :type ids: Sequence[str]
        :param partial: Whether this run reads partial inputs.
        :type partial: bool
        :param notes: Extra information, with the optional keys
            "out_id" (id in the plan), "partial_inputs_reasons",
            "notes" and "missing".
        :type notes: Mapping[str, Any]
        :param width: Width of the included graphic in LaTeX units.
        :type width: str
        :param env: LaTeX float environment, "figure" or "figure*".
        :type env: str
        :raises ValueError: If name is not a non-empty string.
        :raises OSError: If a file cannot be written.
        """
        head = list()
        body = list()
        self._require_name("save_figure", name)
        pdf, png = self.dir / f"{name}.pdf", self.dir / f"{name}.png"
        fig.savefig(pdf, bbox_inches="tight", metadata=PDF_METADATA)
        fig.savefig(png, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        tex = self.dir / f"{name}.tex"
        head = [f"% {name} - generated by {self.generator} "
                f"({self.script_ref})",
                f"% Plan label: {tag}"]
        if partial:
            head.append(f"% {self.partial_tag}: " + "; ".join(
                notes.get("partial_inputs_reasons", list())))
        head.append(ids_comment(ids, name))
        for n in notes.get("notes", list()):
            head.append(f"% {n}")
        body = [f"\\begin{{{env}}}[t]", "  \\centering",
                f"  \\includegraphics[width={width}]"
                f"{{figures/{name}.pdf}}",
                f"  \\caption{{{caption}}}",
                f"  \\label{{{label}}}", f"\\end{{{env}}}"]
        _write_text_atomic(tex, "\n".join(head + body) + "\n")
        self.records.append(
            {"id": notes.get("out_id"), "name": name, "kind": "figure",
             "tag": tag, "partial_inputs": partial, "caption": caption,
             "label": label, "numbers_ids": ids,
             "files": [f"{name}.pdf", f"{name}.png", f"{name}.tex"],
             "notes": notes.get("notes", list()),
             "missing": notes.get("missing", list())})

    def save_table(self,
                   name: str,
                   tex_body: str,
                   md_preview: str | None,
                   caption: str,
                   label: str,
                   tag: str,
                   ids: Sequence[str],
                   partial: bool,
                   notes: Mapping[str, Any]) -> None:
        """
        Write one table as LaTeX, with an optional Markdown preview.

        :param name: Base name of the files, without extension.
        :type name: str
        :param tex_body: Whole LaTeX body, float environment
            included; the provenance comments are prepended.
        :type tex_body: str
        :param md_preview: Markdown preview, or None for none.
        :type md_preview: str | None
        :param caption: Caption in academic English, already escaped.
        :type caption: str
        :param label: LaTeX label, for example "tab:main-results".
        :type label: str
        :param tag: Label of the analysis plan for this output.
        :type tag: str
        :param ids: Ids of the values used by the table.
        :type ids: Sequence[str]
        :param partial: Whether this run reads partial inputs.
        :type partial: bool
        :param notes: Extra information, as in ``save_figure``.
        :type notes: Mapping[str, Any]
        :raises ValueError: If name is not a non-empty string.
        :raises OSError: If a file cannot be written.
        """
        head = list()
        files = list()
        self._require_name("save_table", name)
        tex = self.dir / f"{name}.tex"
        head = [f"% {name} - generated by {self.generator} "
                f"({self.script_ref})",
                f"% Plan label: {tag}"]
        if partial:
            head.append(f"% {self.partial_tag}: " + "; ".join(
                notes.get("partial_inputs_reasons", list())))
        head.append(ids_comment(ids, name))
        for n in notes.get("notes", list()):
            head.append(f"% {n}")
        _write_text_atomic(tex, "\n".join(head) + "\n" + tex_body + "\n")
        files = [f"{name}.tex"]
        if md_preview is not None:
            _write_text_atomic(self.dir / f"{name}.md",
                               md_preview + "\n")
            files.append(f"{name}.md")
        self.records.append(
            {"id": notes.get("out_id"), "name": name, "kind": "table",
             "tag": tag, "partial_inputs": partial, "caption": caption,
             "label": label, "numbers_ids": ids, "files": files,
             "notes": notes.get("notes", list()),
             "missing": notes.get("missing", list())})

    # ----- plain files ----------------------------------------------- #
    def write_text(self, name: str, text: str) -> Path:
        """
        Write a text file of the output directory, atomically.

        It is used for the README, the table of values and the
        checksums.

        :param name: File name inside the output directory.
        :type name: str
        :param text: Whole content of the file.
        :type text: str
        :return: The path written.
        :rtype: Path
        :raises ValueError: If name is not a non-empty string.
        :raises OSError: If the file cannot be written.
        """
        self._require_name("write_text", name)
        _write_text_atomic(self.dir / name, text)
        return self.dir / name

    def write_json(self, name: str, obj: Any) -> Path:
        """
        Write a JSON file of the output directory, atomically.

        NaN and the infinities become ``null`` first, so the file is
        strictly valid JSON. It is indented by two spaces, keeps
        non-ASCII characters as they are, and ends with a newline.

        :param name: File name inside the output directory.
        :type name: str
        :param obj: Structure to serialize, normally the manifest.
        :type obj: Any
        :return: The path written.
        :rtype: Path
        :raises ValueError: If name is not a non-empty string.
        :raises OSError: If the file cannot be written.
        """
        self._require_name("write_json", name)
        _write_text_atomic(self.dir / name,
                           json.dumps(_clean(obj), indent=2,
                                      ensure_ascii=False,
                                      default=_json_default) + "\n")
        return self.dir / name

    def sha256(self, name: str) -> str:
        """
        Hash one file of the output directory.

        :param name: File name inside the output directory.
        :type name: str
        :return: The sha256 digest as 64 hexadecimal characters.
        :rtype: str
        :raises ValueError: If name is not a non-empty string.
        :raises OSError: If the file cannot be read.
        """
        self._require_name("sha256", name)
        return _sha256_file(self.dir / name)

    def output_files(self, exclude: Sequence[str] = tuple()) -> list[str]:
        """
        Names of the files produced so far, sorted.

        :param exclude: File names left out, such as the checksums.
        :type exclude: Sequence[str]
        :return: File names, without directories, in sorted order.
        :rtype: list[str]
        :raises OSError: If the output directory cannot be listed.
        """
        return sorted(p.name for p in self.dir.iterdir()
                      if p.is_file() and p.name not in set(exclude))

    def _require_name(self, method: str, name: str) -> None:
        """
        Check that a file name is a usable non-empty string.

        :param method: Calling method, for the message.
        :type method: str
        :param name: Value received by the caller.
        :type name: str
        :raises ValueError: If name is not a non-empty string.
        """
        if not isinstance(name, str) or not name:
            self._error(f"[Out.{method}] Input error: the parameter "
                        "name must be a non-empty string, not "
                        f"{name!r}.")

    def __repr__(self) -> str:
        """
        Show the directory, the resolution and what was produced.

        :return: Text like "Out(dir='figures', dpi=300, records=7)".
        :rtype: str
        """
        return (f"Out(dir={str(self.dir)!r}, dpi={self.dpi}, "
                f"records={len(self.records)}, "
                f"numbers={len(self.numbers)})")


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                END OF FILE                                #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
