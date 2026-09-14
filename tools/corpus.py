#!/usr/bin/env python3
"""Assemble the routing-trace corpus from text already on this machine.

Design section 9.2 asks for ">= 20K tokens of mixed Chinese, English and code over
several prompts". Two constraints shape this file:

  * **Nothing is downloaded.** Appendix B's no-proxy rule is about the weights, but
    the same discipline applies here: every source is a file already present, and
    every default source is either this repository (MIT), the model card and
    reference implementation shipped inside the checkpoint directory (MIT), or the
    CPython standard library (PSF). `--source` takes more.

  * **The length distribution has to be realistic**, because reuse distance (section
    9.1 Q2) is measured in tokens and a corpus of forty 500-token prompts has a very
    different tail from one of ten 2000-token prompts. Prompt lengths are drawn
    log-uniformly over [--min-len, --max-len], which puts most prompts in the few
    hundred token range with a real 1-2K tail, and prompts never straddle two
    sources: a prompt is contiguous text, the way a real one would be.

A "kind" (zh / en / code) is a share of the token budget, not a file list, so adding
one big Chinese file does not drown out the code.
"""

from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass, field

KINDS = ("en", "zh", "code")

# Splitting points, in preference order: a blank line, then a line break. Code is
# split on blank lines too, which keeps functions together more often than not.
_PARA_RE = re.compile(r"\n\s*\n")


@dataclass
class Source:
    path: str
    kind: str
    note: str = ""


@dataclass
class Prompt:
    text: str
    ids: list[int] = field(default_factory=list)
    kind: str = ""
    source: str = ""

    @property
    def n_tokens(self) -> int:
        return len(self.ids)


def default_sources(repo_dir: str, model_dir: str) -> list[Source]:
    """The files this machine has that are free to use.

    Chinese comes almost entirely from this repository's own design documents, which
    is a real but narrow register (systems engineering prose). That is a stated
    limitation of the trace, not an accident -- see docs/route_trace.md.
    """
    out: list[Source] = []

    def add(pattern: str, kind: str, note: str, limit: int | None = None):
        hits = sorted(glob.glob(pattern, recursive=True))
        for p in hits[:limit] if limit else hits:
            if os.path.isfile(p) and os.path.getsize(p) > 512:
                out.append(Source(p, kind, note))

    # Chinese technical prose (this repo, MIT).
    for name in ("docs/design.md", "docs/architecture.md", "docs/build.md", "README.md"):
        p = os.path.join(repo_dir, name)
        if os.path.isfile(p):
            out.append(Source(p, "zh", "deepMoE repo (MIT)"))

    # English technical prose (the checkpoint's own documentation, MIT).
    for name in ("README.md", "inference/README.md", "encoding/README.md",
                 "evaluation/README.md"):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            out.append(Source(p, "en", "DeepSeek-V4.1-Flash model card (MIT)"))
    p = os.path.join(model_dir, "DeepSeek_V41_Tech_Report.pdf")
    if os.path.isfile(p):
        out.append(Source(p, "en", "DeepSeek-V4.1-Flash tech report (MIT)"))

    # Code: C++ from this repo, Python from the reference implementation and stdlib.
    add(os.path.join(repo_dir, "core", "*.h"), "code", "deepMoE repo (MIT)")
    add(os.path.join(repo_dir, "core", "*.cpp"), "code", "deepMoE repo (MIT)")
    add(os.path.join(repo_dir, "storage", "*.cpp"), "code", "deepMoE repo (MIT)")
    add(os.path.join(repo_dir, "store", "*.cpp"), "code", "deepMoE repo (MIT)")
    add(os.path.join(repo_dir, "model", "*.cpp"), "code", "deepMoE repo (MIT)")
    add(os.path.join(repo_dir, "gpu", "vulkan", "*.cpp"), "code", "deepMoE repo (MIT)")
    add(os.path.join(model_dir, "inference", "*.py"), "code",
        "DeepSeek reference implementation (MIT)")
    add(os.path.join(model_dir, "encoding", "*.py"), "code",
        "DeepSeek reference implementation (MIT)")
    try:
        import sysconfig
        stdlib = sysconfig.get_paths()["stdlib"]
        for name in ("argparse.py", "json/encoder.py", "dataclasses.py", "typing.py",
                     "asyncio/tasks.py", "pathlib.py", "statistics.py", "textwrap.py",
                     "email/message.py", "http/client.py"):
            p = os.path.join(stdlib, name)
            if os.path.isfile(p):
                out.append(Source(p, "code", "CPython standard library (PSF)"))
    except Exception:
        pass
    return out


def read_source(src: Source) -> str:
    if src.path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""
        reader = PdfReader(src.path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    with open(src.path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _blocks(text: str) -> list[str]:
    out = []
    for para in _PARA_RE.split(text):
        para = para.strip("\n")
        if para.strip():
            out.append(para)
    return out


def build(sources: list[Source], tokenizer, target_tokens: int, min_prompts: int,
          min_len: int, max_len: int, seed: int) -> list[Prompt]:
    """Cut the sources into prompts of `min_len`..`max_len` tokens.

    The token budget and the prompt-count floor are both split equally across
    whichever kinds have sources, so one big Chinese document cannot crowd out the
    code. Within a kind, documents are visited round-robin, and a prompt is always a
    contiguous run of blocks from a single document.

    Blocks are tokenised once up front so a target length can be met accurately; a
    character-count proxy overshoots badly on Chinese (~1.5 chars per token) and
    undershoots on English (~4), which is enough to push the whole distribution off
    the requested range.
    """
    import random
    rng = random.Random(seed)

    by_kind: dict[str, list[tuple[Source, list[str], list[int]]]] = {}
    for src in sources:
        text = read_source(src)
        if not text.strip():
            continue
        blocks = _blocks(text)
        if not blocks:
            continue
        lens = [max(1, len(tokenizer.encode(b))) for b in blocks]
        by_kind.setdefault(src.kind, []).append((src, blocks, lens))

    active = [k for k in KINDS if by_kind.get(k)]
    if not active:
        raise SystemExit("no usable corpus sources")
    per_kind_tokens = target_tokens / len(active)
    per_kind_prompts = math.ceil(min_prompts / len(active))

    prompts: list[Prompt] = []
    for kind in active:
        docs = by_kind[kind]
        cursors = [0] * len(docs)          # resume where this doc left off
        got = 0
        made = 0
        doc_i = 0
        stalled = 0
        while (got < per_kind_tokens or made < per_kind_prompts) and stalled < len(docs):
            ci = doc_i % len(docs)
            src, blocks, lens = docs[ci]
            doc_i += 1
            if cursors[ci] >= len(blocks):
                stalled += 1
                continue
            stalled = 0
            # log-uniform over [min_len, max_len]: most prompts in the low hundreds,
            # with a real 1-2K tail, which is what a chat or agent workload looks like
            want = int(round(min_len * (max_len / min_len) ** rng.random()))
            chunk: list[str] = []
            n = 0
            while cursors[ci] < len(blocks) and n < want:
                bl = lens[cursors[ci]]
                # stop *before* a block that would overshoot badly, once the prompt is
                # already long enough to be worth keeping -- otherwise one 900-token
                # code block drags every prompt in that document up to 900
                if chunk and n + bl > want and n >= min_len // 2:
                    break
                chunk.append(blocks[cursors[ci]])
                n += bl
                cursors[ci] += 1
            text = "\n\n".join(chunk)
            ids = tokenizer.encode(text)[:max_len]
            if len(ids) < min_len // 4:
                continue
            prompts.append(Prompt(text=text, ids=ids, kind=kind, source=src.path))
            got += len(ids)
            made += 1

    rng.shuffle(prompts)
    total = sum(p.n_tokens for p in prompts)
    if total < target_tokens or len(prompts) < min_prompts:
        # everything the sources had; the caller reports the shortfall
        return prompts
    kept: list[Prompt] = []
    n = 0
    for p in prompts:
        kept.append(p)
        n += p.n_tokens
        if n >= target_tokens and len(kept) >= min_prompts:
            break
    return kept


def summarise(prompts: list[Prompt]) -> dict:
    lens = sorted(p.n_tokens for p in prompts)
    by_kind: dict[str, list[int]] = {}
    for p in prompts:
        by_kind.setdefault(p.kind, []).append(p.n_tokens)
    total = sum(lens) or 1

    def pct(q: float) -> int:
        return lens[min(len(lens) - 1, int(q * len(lens)))] if lens else 0

    return {
        "prompts": len(prompts),
        "tokens": sum(lens),
        "len_min": lens[0] if lens else 0,
        "len_p50": pct(0.5),
        "len_p90": pct(0.9),
        "len_max": lens[-1] if lens else 0,
        "by_kind": {k: {"prompts": len(v), "tokens": sum(v),
                        "share": round(sum(v) / total, 3)}
                    for k, v in sorted(by_kind.items())},
        "sources": sorted({p.source for p in prompts}),
    }
