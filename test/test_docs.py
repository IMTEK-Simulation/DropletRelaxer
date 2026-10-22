"""The documentation refers to code by dotted names (`DropletRelaxer.energy.area_and_grad`);
every such name must exist, so the docs cannot silently go stale when code moves."""
import glob
import importlib
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = re.compile(r"`((?:DropletRelaxer|DropletViewer)(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`")


def documented_names():
    names = set()
    for path in glob.glob(os.path.join(ROOT, "docs", "**", "*.qmd"), recursive=True):
        with open(path) as f:
            names.update(NAME.findall(f.read()))
    return sorted(names)


def resolve(name):
    parts = name.split(".")
    for k in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:k]))
        except ModuleNotFoundError as err:
            if err.name and not name.startswith(err.name) and err.name.split(".")[0] not in ("DropletRelaxer",
                                                                                            "DropletViewer"):
                raise                                       # a third-party dependency is missing
            continue
        for attr in parts[k:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(name)


def test_docs_mention_code():
    assert len(documented_names()) > 20


@pytest.mark.parametrize("name", documented_names())
def test_documented_name_exists(name):
    if name.startswith("DropletViewer"):
        pytest.importorskip("pyvista")
    resolve(name)
