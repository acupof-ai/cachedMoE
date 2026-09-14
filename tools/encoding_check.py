#!/usr/bin/env python3
"""Run the checkpoint's own prompt-encoding tests (encoding/test_encoding.py,
including the encoding/tests input -> output fixtures) without pytest.

deepMoE renders prompts with the official encoding.py, imported read-only by
tools/chat.py (docs/p3_chat.md §4), so "passing encoding/tests" means that
module passes its own suite in this environment. The .venv has no pytest and
there is no network, so this provides the three pieces of pytest the file uses
(mark.parametrize, raises, main) and runs every test function. Nothing is
written under the model directory (bytecode writing is disabled).

    .venv/Scripts/python.exe tools/encoding_check.py
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import re
import sys
import traceback
import types

sys.dont_write_bytecode = True
MODEL = os.environ.get("DEEPMOE_MODEL_DIR", r"D:\models\DeepSeek-V4.1-Flash")


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        if self.match and not re.search(self.match, str(ev)):
            raise AssertionError(f"exception {ev!r} does not match {self.match!r}")
        return True


def _parametrize(names, values):
    names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)

    def deco(fn):
        params = getattr(fn, "_params", [])
        rows = [v if len(names) > 1 else (v,) for v in values]
        fn._params = [(names, rows)] + params
        return fn
    return deco


def install_shim():
    pt = types.ModuleType("pytest")
    pt.raises = _Raises
    pt.mark = types.SimpleNamespace(parametrize=_parametrize)
    pt.main = lambda *a, **k: 0
    sys.modules["pytest"] = pt


def main():
    enc_dir = os.path.join(MODEL, "encoding")
    install_shim()
    sys.path.insert(0, enc_dir)
    spec = importlib.util.spec_from_file_location("test_encoding", os.path.join(enc_dir, "test_encoding.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    passed = failed = 0
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        fn = getattr(mod, name)
        groups = getattr(fn, "_params", [])
        combos = [dict()]
        for names, rows in groups:
            combos = [dict(c, **dict(zip(names, r))) for c in combos for r in rows]
        for kw in combos:
            label = f"{name}[{', '.join(f'{k}={v!r}' for k, v in kw.items())}]" if kw else name
            try:
                fn(**kw)
                passed += 1
            except Exception:
                failed += 1
                print(f"FAIL {label}")
                traceback.print_exc(limit=3)
    fixtures = len(getattr(mod, "FIXTURE_CASE_IDS", []))
    print(f"encoding/test_encoding.py: {passed} passed, {failed} failed "
          f"({fixtures} encoding/tests fixtures among them)")
    return 1 if failed else 0


if __name__ == "__main__":
    if not sys.flags.utf8_mode:
        import subprocess
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", "-B"] + sys.argv))
    sys.exit(main())
