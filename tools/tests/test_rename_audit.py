#!/usr/bin/env python3
"""CPU regressions for per-hit retention and stable current/committed snapshots."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rename_audit", ROOT / "tools" / "rename_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class RenameAudit(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="cachedmoe-audit-")
        self.addCleanup(self.folder.cleanup)
        self.repo = Path(self.folder.name)
        self.git("init", "-q")

    def git(self, *args):
        child = subprocess.run(["git", "-C", str(self.repo), *args],
                               capture_output=True, text=True, timeout=15)
        self.assertEqual(child.returncode, 0, child.stderr)
        return child.stdout.strip()

    def write(self, name, text):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self):
        self.git("add", ".")
        self.git("-c", "user.name=Audit fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def test_mixed_line_is_classified_per_token_with_utf8_byte_columns(self):
        self.write("live.py", '中文 deepmoe_manifest.json ; deepmoe run\n')
        self.commit()
        result = audit.audit(self.repo, "HEAD", [])
        first, second = result["hits"]
        self.assertEqual((first["column"], first["character_column"]), (8, 4))
        self.assertEqual(first["status"], "RETAIN")
        self.assertEqual(second["status"], "NEEDS_REVIEW")
        self.assertEqual(first["exact_token"], "deepmoe_manifest.json")
        self.assertEqual(first["text"], second["text"])

    def test_status_current_source_requires_review_but_golden_is_retained(self):
        self.write("docs/STATUS.md", "Current usage: deepmoe serve\n")
        self.write("tests/data/golden.json", '{"name":"deepmoe"}\n')
        self.commit()
        result = audit.audit(self.repo, "HEAD", [])
        by_path = {hit["path"]: hit for hit in result["hits"]}
        self.assertEqual(by_path["docs/STATUS.md"]["status"], "NEEDS_REVIEW")
        self.assertEqual(by_path["tests/data/golden.json"]["status"], "RETAIN")

    def test_shipped_rules_do_not_exempt_current_status_instructions(self):
        self.write("docs/STATUS.md", "Current usage: deepmoe serve\n\n"
                   "### 1.0 Machine-recorded measurements (ledger)\n"
                   "Old measurement: deepmoe run\n")
        self.commit()
        rules = audit.load_rules(audit.DEFAULT_RULES)
        result = audit.audit(self.repo, "HEAD", rules)
        self.assertEqual([hit["status"] for hit in result["hits"]],
                         ["NEEDS_REVIEW", "RETAIN"])

    def test_reviewed_rule_is_constrained_to_exact_token_and_line(self):
        self.write("compat.h", "namespace deepmoe = cachedmoe;\nuse deepmoe here\n")
        self.commit()
        rules = [dict(id="source-alias", path_glob="compat.h", token_regex="deepmoe",
                      line_regex="namespace deepmoe = cachedmoe", role="source_alias",
                      reason="Explicit legacy namespace source alias.")]
        result = audit.audit(self.repo, "HEAD", rules)
        self.assertEqual([hit["status"] for hit in result["hits"]], ["RETAIN", "NEEDS_REVIEW"])
        self.assertEqual(result["hits"][0]["role"], "source_alias")

    def test_current_tree_includes_edits_and_nonignored_pending_files(self):
        self.write("live.py", "cachedmoe\n")
        commit = self.commit()
        self.write("live.py", "deepmoe\n")
        self.write("pending.py", "DEEPMOE_X\n")
        committed = audit.audit(self.repo, commit, [])
        current = audit.audit(self.repo, None, [])
        self.assertEqual(committed["counts"]["content_hits"], 0)
        self.assertEqual(current["counts"]["content_hits"], 2)
        self.assertNotEqual(committed["source_fingerprint_sha256"],
                            current["source_fingerprint_sha256"])
        self.assertEqual({hit["path"] for hit in current["hits"]}, {"live.py", "pending.py"})

    def test_current_snapshot_refuses_concurrent_content_change(self):
        self.write("live.py", "cachedmoe\n")
        self.commit()
        scans = iter([
            [("live.py", None, "100644", b"deepmoe\n")],
            [("live.py", None, "100644", b"changed\n")],
        ])
        with mock.patch.object(audit, "working_files", side_effect=lambda _: iter(next(scans))):
            with self.assertRaisesRegex(RuntimeError, "changed during audit"):
                audit.audit(self.repo, None, [])

    def test_compatibility_wrapper_filename_and_binary_accounting(self):
        self.write("cmake/deepmoe_options.cmake",
                   'include("${CMAKE_CURRENT_LIST_DIR}/cachedmoe_options.cmake")\n')
        (self.repo / "binary.bin").write_bytes(b"\x00deepmoe\xff")
        self.commit()
        result = audit.audit(self.repo, "HEAD", [])
        self.assertEqual(result["tracked_path_hits"][0]["status"], "RETAIN")
        self.assertEqual(result["counts"]["skipped_binary_files"], 1)
        self.assertEqual(result["counts"]["content_hits"], 0)

    def test_cli_refuses_to_overwrite_existing_receipt(self):
        self.write("live.py", "cachedmoe\n")
        self.commit()
        out = self.repo / "existing.json"
        out.write_bytes(b"keep original receipt\n")
        child = subprocess.run([sys.executable, str(ROOT / "tools" / "rename_audit.py"),
                                "--repo", str(self.repo), "--ref", "HEAD", "--out", str(out)],
                               capture_output=True, text=True, timeout=15)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("refusing to overwrite", child.stderr)
        self.assertEqual(out.read_bytes(), b"keep original receipt\n")

    def test_metadata_requires_unique_ids_and_concrete_roles(self):
        path = self.repo / "rules.json"
        rule = dict(id="duplicate", path_glob="*.h", token_regex="deepmoe",
                    role="source_alias", reason="Explicit legacy alias.")
        path.write_text(json.dumps({"rules": [rule, rule]}))
        with self.assertRaisesRegex(ValueError, "duplicate retention rule"):
            audit.load_rules(path)

    def test_shipped_retention_metadata_is_valid(self):
        rules = audit.load_rules(audit.DEFAULT_RULES)
        self.assertTrue(rules)
        self.assertTrue(any(rule["role"] == "legacy_source_namespace_alias" for rule in rules))


if __name__ == "__main__":
    unittest.main()
