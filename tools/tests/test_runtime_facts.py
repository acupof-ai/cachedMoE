#!/usr/bin/env python3
"""CPU shared-fact checks; --check-cpp also compiles an actual header reader."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import runtime_facts


CPP_READER = r'''
#include "core/runtime_facts.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

int main() {
    bool first = true;
    const auto separator = [&] {
        if (!first) std::cout << ',';
        first = false;
    };
    std::cout << '{' << std::setprecision(17);
#define CACHEDMOE_RUNTIME_FACT_INTEGER(type, name, value) \
    static_assert(std::is_same_v<decltype(cachedmoe::configuration::facts::name), const std::type>); \
    separator(); std::cout << '"' << #name << "\":" << cachedmoe::configuration::facts::name;
#define CACHEDMOE_RUNTIME_FACT_REAL(name, value) \
    static_assert(std::is_same_v<decltype(cachedmoe::configuration::facts::name), const double>); \
    separator(); std::cout << '"' << #name << "\":" << std::showpoint \
                          << cachedmoe::configuration::facts::name << std::noshowpoint;
#define CACHEDMOE_RUNTIME_FACT_TEXT(name, value) \
    static_assert(std::is_same_v<decltype(cachedmoe::configuration::facts::name), const std::string_view>); \
    separator(); std::cout << '"' << #name << "\":" << std::quoted(std::string(cachedmoe::configuration::facts::name));
#include "core/runtime_facts.def"
#undef CACHEDMOE_RUNTIME_FACT_INTEGER
#undef CACHEDMOE_RUNTIME_FACT_REAL
#undef CACHEDMOE_RUNTIME_FACT_TEXT
    std::cout << "}\n";
}
'''


def check_cpp_reader(cxx=None):
    """No project build or GPU: compare real compiled constexpr values."""
    selected = cxx or os.environ.get("CXX") or shutil.which("c++")
    if not selected:
        raise RuntimeError("--check-cpp requires a C++20 compiler")
    command = shlex.split(selected)
    with tempfile.TemporaryDirectory(prefix="cachedmoe-facts-reader-") as folder:
        source = Path(folder) / "reader.cpp"
        binary = Path(folder) / "reader"
        source.write_text(CPP_READER, encoding="utf-8")
        child = subprocess.run(command + ["-std=c++20", "-Wall", "-Wextra", "-Werror",
                                          "-I", str(ROOT), str(source), "-o", str(binary)],
                               capture_output=True, text=True, timeout=30)
        if child.returncode:
            raise RuntimeError("actual facts reader compilation failed: " + child.stderr)
        child = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
        if child.returncode:
            raise RuntimeError("actual facts reader failed: " + child.stderr)
        native = json.loads(child.stdout)
        expected = dict(runtime_facts.FACTS)
        if native != expected:
            raise AssertionError(f"actual C++/Python facts differ: {native!r} != {expected!r}")
        # bool is a subclass of int in Python; require exact parsed value types.
        if any(type(native[key]) is not type(value) for key, value in expected.items()):
            raise AssertionError("actual C++/Python fact value types differ")
        return native


class RuntimeFactsTests(unittest.TestCase):
    def test_actual_authority_is_used_and_scopes_are_distinct(self):
        facts = runtime_facts.load_facts()
        self.assertEqual(dict(facts), dict(runtime_facts.FACTS))
        self.assertLess(facts["SESSION_CONTEXT"], facts["RUNTIME_CONTEXT"])
        self.assertLess(facts["RUNTIME_CONTEXT"], facts["MAX_CONTEXT"])
        self.assertLess(facts["NATIVE_MAX_TOKENS"], facts["WEB_MAX_TOKENS"])
        self.assertGreater(facts["PREFILL_REPLAY"], 0)
        self.assertLessEqual(facts["PREFILL_REPLAY"], facts["SESSION_CONTEXT"])
        self.assertLess(facts["RADV_GPU_PREFILL_MIN"], facts["SESSION_GPU_PREFILL_MIN"])
        self.assertIn("KV_DISK_BYTES", facts)
        self.assertIn("PARKED_KV_BYTES", facts)
        self.assertEqual(runtime_facts.TYPES["KV_DISK_BYTES"], "uint64_t")
        self.assertEqual(runtime_facts.TYPES["SAMPLING_TEMPERATURE"], "double")

    def test_values_and_types_are_immutable(self):
        with self.assertRaises(TypeError):
            runtime_facts.FACTS["MAX_CONTEXT"] = 7
        with self.assertRaises(TypeError):
            runtime_facts.TYPES["MAX_CONTEXT"] = "double"
        with self.assertRaises(TypeError):
            runtime_facts.parse_facts("CACHEDMOE_RUNTIME_FACT_INTEGER(uint32_t, A, 1)")["A"] = 2

    def test_both_import_spellings_have_one_module_identity(self):
        sys.path.insert(0, str(ROOT))
        self.addCleanup(sys.path.remove, str(ROOT))
        self.assertIs(importlib.import_module("tools.runtime_facts"), runtime_facts)
        self.assertIs(importlib.import_module("runtime_facts"), runtime_facts)
        code = ("import tools.runtime_facts as a; import runtime_facts as b; "
                "assert a is b; assert a.FACTS is b.FACTS")
        child = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_integer_width_bounds_and_decimal_grammar(self):
        valid = ("CACHEDMOE_RUNTIME_FACT_INTEGER(uint32_t, A, 4294967295)\n"
                 "CACHEDMOE_RUNTIME_FACT_INTEGER(uint64_t, B, 18446744073709551615)")
        parsed = runtime_facts.parse_facts(valid)
        self.assertEqual(parsed["A"], (1 << 32) - 1)
        self.assertEqual(parsed["B"], (1 << 64) - 1)
        for kind, literal in (("uint32_t", "4294967296"), ("uint64_t", "18446744073709551616"),
                              ("uint32_t", "-1"), ("uint32_t", "01"),
                              ("uint32_t", "0x10"), ("int32_t", "1")):
            with self.subTest(kind=kind, literal=literal), self.assertRaises(ValueError):
                runtime_facts.parse_facts(f"CACHEDMOE_RUNTIME_FACT_INTEGER({kind}, A, {literal})")

    def test_real_literals_preserve_double_values_and_reject_nonfinite(self):
        parsed = runtime_facts.parse_facts("CACHEDMOE_RUNTIME_FACT_REAL(A, 0.95)\n"
                                          "CACHEDMOE_RUNTIME_FACT_REAL(B, -2.5e-1)")
        self.assertEqual(parsed["A"], .95)
        self.assertEqual(parsed["B"], -.25)
        for literal in ("NaN", "inf", "1.0e999", "1.0f", "1", ".5", "01.0"):
            with self.subTest(literal=literal), self.assertRaises(ValueError):
                runtime_facts.parse_facts(f"CACHEDMOE_RUNTIME_FACT_REAL(A, {literal})")

    def test_text_literals_support_only_cpp_json_common_escapes(self):
        parsed = runtime_facts.parse_facts(r'CACHEDMOE_RUNTIME_FACT_TEXT(A, "a\\b\"c")')
        self.assertEqual(parsed["A"], 'a\\b"c')
        for literal in (r'"a\nb"', r'"a\/b"', r'"\u0061"', '"中文"'):
            with self.subTest(literal=literal), self.assertRaises(ValueError):
                runtime_facts.parse_facts(f"CACHEDMOE_RUNTIME_FACT_TEXT(A, {literal})")

    def test_duplicate_reserved_and_unknown_keys_are_rejected(self):
        text = ("CACHEDMOE_RUNTIME_FACT_INTEGER(uint32_t, A, 1)\n"
                "CACHEDMOE_RUNTIME_FACT_REAL(A, 1.0)")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runtime_facts.parse_facts(text)
        for name in ("FACTS", "TYPES", "SOURCE"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "reserved"):
                runtime_facts.parse_facts(f"CACHEDMOE_RUNTIME_FACT_INTEGER(uint32_t, {name}, 1)")
        for text in ("", "// only comments", "#define A 1", "CACHEDMOE_RUNTIME_FACT_UNKNOWN(A, 1)"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                runtime_facts.parse_facts(text)

    def test_expressions_and_trailing_code_are_not_evaluated(self):
        for literal in ("1 << 20", "1+1", "__import__('os').system('false')"):
            with self.subTest(literal=literal), self.assertRaises(ValueError):
                runtime_facts.parse_facts(f"CACHEDMOE_RUNTIME_FACT_INTEGER(uint32_t, A, {literal})")
        with self.assertRaises(ValueError):
            runtime_facts.parse_facts("CACHEDMOE_RUNTIME_FACT_INTEGER(uint32_t, A, 1);")

    def test_authority_loading_does_not_consult_environment(self):
        with mock.patch.dict(os.environ, {"CACHEDMOE_RUNTIME_FACTS": "/missing/facts.def",
                                          "CACHEDMOE_MAX_CONTEXT": "1"}):
            self.assertEqual(dict(runtime_facts.load_facts()), dict(runtime_facts.FACTS))
        with tempfile.TemporaryDirectory(prefix="cachedmoe-facts-parser-") as folder:
            path = Path(folder) / "facts.def"
            path.write_text("CACHEDMOE_RUNTIME_FACT_INTEGER(uint32_t, A, 7)\n")
            self.assertEqual(dict(runtime_facts.load_facts(path)), {"A": 7})
            path.write_text("not a fact\n")
            with self.assertRaises(ValueError):
                runtime_facts.load_facts(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-cpp", action="store_true")
    parser.add_argument("--cxx", help="compiler command for --check-cpp")
    args, remaining = parser.parse_known_args()
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
    if args.check_cpp:
        print(json.dumps({"actual_cpp_facts": check_cpp_reader(args.cxx)}, sort_keys=True))
