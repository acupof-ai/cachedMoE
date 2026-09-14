#!/usr/bin/env python3
"""Validate the C++ tokenizer (text/tokenizer.cpp) against HF `tokenizers`, and
write the golden file tests/test_tokenizer.cpp checks.

    .venv/Scripts/python.exe tools/tokenizer_golden.py [--exe build/deepmoe.exe]
        [--write-golden] [--quick]

Case sets (docs/p3_chat.md §3.3):

  corpus      every file tools/corpus.py's default_sources() lists (Track B's
              routing-trace corpus: this repo's docs and C++, the model card and
              tech report, the reference Python, CPython stdlib files), whole
  longctx     tests/data/longctx/prompts.json's two prompts (and their stored ids)
  encoding    the model's encoding/tests expected outputs, and encoding.py's
              rendering of each input (thinking and chat mode)
  added       every added token alone, and glued between text
  adversarial hand-written: emoji (ZWJ, flags, skin tones), mixed scripts,
              whitespace runs of every Unicode space, CRLF mixes, digit runs,
              combining marks, byte-level lookalike characters, special-token
              fragments
  fuzz        seeded random strings over those pools
  codepoints  every Unicode scalar value, in chunks, in four small contexts

Every case is compared on ids (must be 100% equal), decode, decode with specials
skipped, and the streaming decode (must equal decode). Nothing is downloaded and
nothing is written under the model directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.dont_write_bytecode = True

MODEL = os.environ.get("DEEPMOE_MODEL_DIR", r"D:\models\DeepSeek-V4.1-Flash")


def fnv64(ids):
    h = 1469598103934665603
    for i in ids:
        for b in int(i).to_bytes(4, "little"):
            h ^= b
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def load_encoding_module():
    enc_dir = os.path.join(MODEL, "encoding")
    sys.path.insert(0, enc_dir)
    import encoding  # noqa: E402  (read-only import; bytecode writing disabled)
    return encoding, enc_dir


SPACES = ["\t", "\n", "\r", "\x0b", "\x0c", " ", "\x85", "\xa0", "\u1680", "\u2000", "\u2001",
          "\u2002", "\u2003", "\u2007", "\u200a", "\u2028", "\u2029", "\u202f", "\u205f",
          "\u3000", "\u200b", "\ufeff", "\r\n"]
POOLS = {
    "ascii": [chr(c) for c in range(0x20, 0x7F)],
    "digits": list("0123456789") + ["\u0663", "\u06f5", "\u0966", "\uff11", "\u2460", "\u00bd",
                                    "\u2167", "\U0001d7d9"],
    "cjk": [chr(c) for c in range(0x4E00, 0x4E80)] + ["\u9fa5", "\u9fa6", "\u9fff", "\u3400",
                                                      "\U00020000", "\u3007"],
    "kana": [chr(c) for c in range(0x3040, 0x3100, 3)] + ["\u30fc", "\uff76"],
    "hangul": ["\ud55c", "\uad6d", "\uc5b4", "\u1100", "\u1161"],
    "arabic": ["\u0627", "\u0644", "\u0639", "\u0631", "\u0628", "\u064a", "\u064e", "\u0651"],
    "indic": ["\u0915", "\u094d", "\u0937", "\u093f", "\u0928", "\u0926", "\u0940"],
    "latin": ["\u00e9", "e\u0301", "\u00df", "\u0130", "\u0131", "\u01c5", "\u1e9e", "\u00f8"],
    "greek_cyr": ["\u03b1", "\u03a9", "\u0416", "\u044f", "\u0451"],
    "emoji": ["\U0001f600", "\U0001f44d\U0001f3fd", "\U0001f468\u200d\U0001f469\u200d\U0001f467",
              "\U0001f1e8\U0001f1f3", "\u2764\ufe0f", "\U0001f9d1\u200d\U0001f4bb", "\u2603",
              "\U0001faf6", "#\ufe0f\u20e3", "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e"
              "\U000e0067\U000e007f"],
    "symbols": ["\u00a9", "\u2192", "\u2211", "\u221e", "\u20ac", "\u00a5", "\uff01", "\u3002",
                "\u300c", "\u300d", "\uff0c", "\u2014", "\u2026", "\u00b7", "\U0001d400", "\ufffd",
                "\ue000", "\U0010fffd", "\ufdd0", "\uffff"],
    "bytelevel": ["\u0120", "\u010a", "\u0109", "\u00a1", "\u00ad", "\u0100", "\u0143"],
    "space": SPACES,
    "special": ["<\uff5cUser\uff5c>", "<\uff5cAssistant\uff5c>", "<\uff5cend\u2581of\u2581sentence\uff5c>",
                "<\uff5cbegin\u2581of\u2581sentence\uff5c>", "<think>", "</think>", "\uff5cDSML\uff5c",
                "<\uff5c", "\uff5c>", "<|EOT|>", "<dsml:", "</dsml:", "\u2581", "<\uff5cUser",
                "<\uff5cSystem\uff5c>", "<|place_holder_mm_span_0442|>"],
}


def adversarial_cases():
    c = [
        "", " ", "  ", "\n", "\n\n", "\r\n", " \r\n ", "a", "Hello world", " Hello  world ",
        "Hello world!!! ... ?!", "12345678901", "a1b22c333d4444", "3.14159 1,000,000 -42 1e-9",
        "你好，世界！这是一个测试。", "日本語のテキストとカタカナ、ひらがな。", "한국어 텍스트",
        "中英mixed混合text文本123数字", "Привет, мир!", "مرحبا بالعالم", "नमस्ते दुनिया",
        "e\u0301\u0301\u0301 combining marks \u0301 alone", "\u0301", " \u0301", "a\u0301b",
        "\U0001f600\U0001f600 emoji \U0001f468\u200d\U0001f469\u200d\U0001f467 family \U0001f1e8\U0001f1f3",
        "tabs\t\tand\u3000ideographic\u3000spaces\u00a0nbsp", "trailing spaces   ",
        "   leading spaces", "line1\nline2\r\nline3\rline4", "x \n \n y", "x  \n\n  y",
        "\t\n\t\n", " \u2028 \u2029 ", "def f(x):\n    return x ** 2\n\n\nclass A:\n\tpass\n",
        "if (a && b) { return c->d[e]; } // comment", "<a href=\"x\">link</a>", "`code` ~tilde~",
        "'s 're n't I'm you'll", "!abc #tag @user $var %mod ^x &y *z (p) [q] {r} |s \\t /u",
        "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>hi<\uff5cAssistant\uff5c></think>",
        "<\uff5cUser\uff5c><\uff5cUser\uff5c>", "<think>reasoning</think>answer",
        "text<\uff5cend\u2581of\u2581sentence\uff5c>more", "<\uff5cUser", "\uff5c>", "<<\uff5cUser\uff5c>>",
        "<\uff5cbegin\u2581sys\uff5c><\uff5cbegin\u2581of\u2581sentence\uff5c>",
        "\u0120\u010a byte-level lookalikes \u0100\u00ad", "\ufffd replacement", "\ue000 private use",
        "\U0010ffff", "\ufeffBOM", "\u200bzero\u200bwidth", "a" * 1000, " " * 300, "\n" * 50,
        "你" * 500, "1" * 100, "!" * 64, "ab" * 300, "\U0001f600" * 40,
        "Ωmega ǅ ǈ ǋ titlecase", "ﬁ ligature ﬀ", "①②③ circled ¼ ½", "٣٤٥ arabic-indic digits",
        "𝟘𝟙𝟚 math digits 𝐀𝐁", "‼⁉ punct", "«quoted» „German“ ‚single‘", "—em–en-hyphen‐",
        "…ellipsis", "中文，标点。「引号」『书名』（括号）【方括号】", "x\u0000y", "\x01\x02\x7f",
        "C:\\Users\\path\\file.txt", "https://example.com/a?b=c&d=e#f", "user@example.com",
        "$ ls -la | grep '^d' > out.txt 2>&1", "∀x∈ℝ: x²≥0", "λ→μ", "½+¼=¾",
    ]
    return c


def fuzz_cases(n, seed):
    rng = random.Random(seed)
    names = list(POOLS)
    out = []
    for _ in range(n):
        k = rng.randint(1, 3)
        pools = [POOLS[rng.choice(names)] for _ in range(k)] + [POOLS["space"]]
        length = rng.randint(1, 48)
        out.append("".join(rng.choice(rng.choice(pools)) for _ in range(length)))
    return out


def codepoint_cases():
    cps = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]
    out = []
    for i in range(0, len(cps), 512):
        chunk = cps[i:i + 512]
        s = []
        for j, c in enumerate(chunk):
            ch = chr(c)
            s.append((ch, "a" + ch, " " + ch + ch, ch + "1 ")[j % 4])
        out.append("".join(s))
    return out


def corpus_cases():
    import corpus
    out = []
    for src in corpus.default_sources(REPO, MODEL):
        text = corpus.read_source(src)
        if text:
            out.append((os.path.relpath(src.path, REPO) if src.path.startswith(REPO) else src.path,
                        text))
    return out


def run_cpp(exe, texts):
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "cases.json")
        outp = os.path.join(d, "out.jsonl")
        with open(inp, "w", encoding="utf-8") as f:
            json.dump({"cases": [{"text": t} for t in texts]}, f, ensure_ascii=False)
        r = subprocess.run([exe, "tokenize", "--model", MODEL, "--in", inp, "--out", outp],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise SystemExit(f"deepmoe tokenize failed: {r.stderr}")
        sys.stderr.write("  c++: " + r.stderr.strip().splitlines()[-1] + "\n")
        with open(outp, encoding="utf-8") as f:
            return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "deepmoe.exe"))
    ap.add_argument("--write-golden", action="store_true")
    ap.add_argument("--quick", action="store_true", help="skip the code point sweep and the corpus")
    ap.add_argument("--fuzz", type=int, default=20000)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok_path = os.path.join(MODEL, "tokenizer.json")
    hf = Tokenizer.from_file(tok_path)
    with open(tok_path, "rb") as f:
        tok_sha = hashlib.sha256(f.read()).hexdigest()

    sets: dict[str, list[str]] = {}
    names: dict[str, list[str]] = {}
    sets["adversarial"] = adversarial_cases()
    added = [a["content"] for a in json.load(open(tok_path, encoding="utf-8"))["added_tokens"]]
    sets["added"] = added + [f"x{a} {a}y\n{a}{a}" for a in added]
    sets["fuzz"] = fuzz_cases(args.fuzz, 1234)

    encoding, enc_dir = load_encoding_module()
    enc_texts = []
    tdir = os.path.join(enc_dir, "tests")
    for fn in sorted(os.listdir(tdir)):
        if fn.startswith("test_output_") and fn.endswith(".txt"):
            with open(os.path.join(tdir, fn), encoding="utf-8") as f:
                enc_texts.append(f.read())
    for fn in sorted(os.listdir(tdir)):
        if fn.startswith("test_input_") and fn.endswith(".json"):
            for mode in ("chat", "thinking"):
                try:
                    for case in encoding.load_cases(os.path.join(tdir, fn)):
                        p, _ = encoding.encode_case(case, mode)
                        enc_texts.append(p)
                except Exception as e:  # a case that only makes sense in one mode
                    sys.stderr.write(f"  encoding {fn} {mode}: {e}\n")
    sets["encoding"] = enc_texts

    lc = json.load(open(os.path.join(REPO, "tests", "data", "longctx", "prompts.json"), encoding="utf-8"))
    sets["longctx"] = [v["text"] for v in lc["prompts"].values()]
    if not args.quick:
        cc = corpus_cases()
        sets["corpus"] = [t for _, t in cc]
        names["corpus"] = [n for n, _ in cc]
        sets["codepoints"] = codepoint_cases()

    total_ok = True
    summary = {}
    results = {}
    for name, texts in sets.items():
        sys.stderr.write(f"{name}: {len(texts)} cases, {sum(len(t) for t in texts)} chars\n")
        ours = run_cpp(args.exe, texts)
        encs = hf.encode_batch(texts, add_special_tokens=True)
        bad_ids = bad_dec = bad_skip = bad_stream = 0
        n_ids = 0
        first_bad = None
        for i, (t, o, e) in enumerate(zip(texts, ours, encs)):
            n_ids += len(e.ids)
            if o["ids"] != e.ids:
                bad_ids += 1
                if first_bad is None:
                    first_bad = (i, t[:200], e.ids[:40], o["ids"][:40])
            d = hf.decode(e.ids, skip_special_tokens=False)
            ds = hf.decode(e.ids, skip_special_tokens=True)
            bad_dec += o["decode"] != d
            bad_skip += o["decode_skip"] != ds
            bad_stream += o["stream"] != o["decode"]
        ok = bad_ids == 0 and bad_dec == 0 and bad_skip == 0 and bad_stream == 0
        total_ok &= ok
        summary[name] = {"cases": len(texts), "ids": n_ids, "id_mismatch": bad_ids,
                         "decode_mismatch": bad_dec, "decode_skip_mismatch": bad_skip,
                         "stream_mismatch": bad_stream,
                         "fnv64_ids": f"{fnv64([i for e in encs for i in e.ids]):016x}"}
        results[name] = (texts, encs)
        print(f"{name:12s} {len(texts):6d} cases {n_ids:9d} ids  ids {len(texts) - bad_ids}/{len(texts)}"
              f"  decode {len(texts) - bad_dec}/{len(texts)}  skip {len(texts) - bad_skip}/{len(texts)}"
              f"  stream {len(texts) - bad_stream}/{len(texts)}  {'OK' if ok else 'MISMATCH'}")
        if first_bad:
            i, t, a, b = first_bad
            print(f"   first mismatch #{i}: {t!r}\n   hf  {a}\n   c++ {b}")

    if "longctx" in results:
        texts, encs = results["longctx"]
        for (k, v), e in zip(lc["prompts"].items(), encs):
            stored = v["ids"]
            ok = stored == e.ids or stored == [lc["bos_id"]] + e.ids
            print(f"longctx {k}: stored ids {'match' if ok else 'DIFFER'} (len {len(stored)} vs {len(e.ids)})")

    if args.write_golden:
        rng = random.Random(7)
        cases = []
        def add(set_name, idx_list):
            texts, encs = results[set_name]
            for i in idx_list:
                cases.append({"set": set_name, "text": texts[i], "ids": encs[i].ids,
                              "decode": hf.decode(encs[i].ids, skip_special_tokens=False),
                              "decode_skip": hf.decode(encs[i].ids, skip_special_tokens=True)})
        add("adversarial", range(len(results["adversarial"][0])))
        add("encoding", range(len(results["encoding"][0])))
        add("added", list(range(0, 64)) + list(range(len(added), len(added) + 64)))
        add("fuzz", sorted(rng.sample(range(len(results["fuzz"][0])), 600)))
        if "codepoints" in results:
            add("codepoints", sorted(rng.sample(range(len(results["codepoints"][0])), 24)))
        if "corpus" in results:
            texts, encs = results["corpus"]
            for i, t in enumerate(texts):   # the first ~1500 characters of every source
                cut = t[:1500]
                e = hf.encode(cut)
                cases.append({"set": "corpus:" + os.path.basename(names["corpus"][i]), "text": cut,
                              "ids": e.ids, "decode": hf.decode(e.ids, skip_special_tokens=False),
                              "decode_skip": hf.decode(e.ids, skip_special_tokens=True)})
        out = {"generator": "tools/tokenizer_golden.py", "tokenizers_version": __import__("tokenizers").__version__,
               "tokenizer_json_sha256": tok_sha, "summary": summary, "cases": cases}
        gdir = os.path.join(REPO, "tests", "data", "tokenizer")
        os.makedirs(gdir, exist_ok=True)
        with open(os.path.join(gdir, "golden.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, ensure_ascii=False, indent=0)
        print(f"wrote tests/data/tokenizer/golden.json: {len(cases)} cases")
    print("ALL EQUAL" if total_ok else "MISMATCHES FOUND")
    return 0 if total_ok else 1


if __name__ == "__main__":
    if not sys.flags.utf8_mode:   # encoding.py opens its test files with the locale codec
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", "-B"] + sys.argv))
    sys.exit(main())
