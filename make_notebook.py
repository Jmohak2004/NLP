"""Regenerate heatwave_nlp.ipynb from heatwave_nlp.py.

heatwave_nlp.py is the single source of truth. "# %%" starts a code cell and
"# %% [markdown]" starts a markdown cell (its lines lose the leading "# ").
Run:  python make_notebook.py
"""
import json

SOURCE, TARGET = "heatwave_nlp.py", "heatwave_nlp.ipynb"


def split_cells(text):
    cells, kind, lines = [], None, []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, lines))
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    cells.append((kind, lines))
    return cells


def to_cell(kind, lines):
    while lines and not lines[-1].strip():
        lines.pop()
    if kind == "markdown":
        lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
    source = [l + "\n" for l in lines]
    if source:
        source[-1] = source[-1].rstrip("\n")
    cell = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


with open(SOURCE, encoding="utf-8") as f:
    cells = [to_cell(k, l) for k, l in split_cells(f.read())]
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(TARGET, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f"Wrote {TARGET} with {len(cells)} cells.")
