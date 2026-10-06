"""Directory migration policy and existing transcript access; no GPU."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/web")]
import state_paths
import server


class StateDirectories(unittest.TestCase):
    def test_linux_external_environment_order_and_empty_values(self):
        self.assertEqual(state_paths.cache_home({"XDG_CACHE_HOME": "/xdg", "HOME": "/home"},
                                                windows=False), "/xdg")
        self.assertEqual(state_paths.cache_home({"XDG_CACHE_HOME": "", "HOME": "/home"},
                                                windows=False), "/home/.cache")
        self.assertEqual(state_paths.cache_home({"HOME": ""}, windows=False), "/tmp")
        self.assertEqual(state_paths.cache_home({}, windows=False, temp_dir="/system-temp"),
                         "/system-temp")

    def test_windows_external_environment_order(self):
        self.assertEqual(state_paths.cache_home({"LOCALAPPDATA": "/local", "TEMP": "/temp"},
                                                windows=True), "/local")
        self.assertEqual(state_paths.cache_home({"LOCALAPPDATA": "", "TEMP": "/temp",
                                                "TMPDIR": "/lower"}, windows=True), "/temp")
        self.assertEqual(state_paths.cache_home({}, windows=True, temp_dir="/system-temp"),
                         "/system-temp")

    def test_shared_temp_fallback_order(self):
        order = ("TMPDIR", "TEMP", "TMP", "TEMPDIR")
        for windows in (False, True):
            for first in range(len(order)):
                env = {key: f"/temp-{i}" if i >= first else "" for i, key in enumerate(order)}
                # On Windows TEMP is a higher-priority state base.
                expected = env.get("TEMP") if windows and env.get("TEMP") else env[order[first]]
                self.assertEqual(state_paths.cache_home(env, windows=windows), expected)

    def test_no_existing_root_selects_canonical_without_creating(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = state_paths.choose_directory(tmp)
            self.assertEqual(selection.path, str(Path(tmp) / "cachedmoe"))
            self.assertEqual(selection.source, "canonical-new")
            self.assertFalse(Path(selection.path).exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_legacy_root_is_reused_without_renaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "deepmoe"
            legacy.mkdir()
            (legacy / "keep.bin").write_bytes(b"legacy")
            selection = state_paths.choose_directory(tmp)
            self.assertEqual(selection.path, str(legacy))
            self.assertEqual(selection.source, "legacy-existing")
            self.assertFalse((Path(tmp) / "cachedmoe").exists())
            self.assertEqual((legacy / "keep.bin").read_bytes(), b"legacy")

    def test_both_roots_select_canonical_and_preserve_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            canonical, legacy = Path(tmp) / "cachedmoe", Path(tmp) / "deepmoe"
            canonical.mkdir(); legacy.mkdir()
            (legacy / "keep.bin").write_bytes(b"legacy")
            selection = state_paths.choose_directory(tmp)
            self.assertEqual(selection.path, str(canonical))
            self.assertTrue(selection.both_exist)
            self.assertEqual(selection.source, "canonical-existing")
            self.assertEqual((legacy / "keep.bin").read_bytes(), b"legacy")

    def test_any_default_root_file_collision_is_an_error(self):
        for file_name, other_exists in (("cachedmoe", False), ("deepmoe", False),
                                        ("cachedmoe", True), ("deepmoe", True)):
            with self.subTest(file_name=file_name, other_exists=other_exists), \
                    tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / file_name).write_bytes(b"keep collision")
                if other_exists:
                    (root / ("deepmoe" if file_name == "cachedmoe" else "cachedmoe")).mkdir()
                with self.assertRaisesRegex(ValueError, "not a directory"):
                    state_paths.choose_directory(root)
                self.assertEqual((root / file_name).read_bytes(), b"keep collision")

    def test_dangling_symlink_fails_but_directory_symlink_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "cachedmoe"
            canonical.symlink_to(root / "missing", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "not a directory"):
                state_paths.choose_directory(root)
            canonical.unlink()
            target = root / "existing"
            target.mkdir()
            canonical.symlink_to(target, target_is_directory=True)
            self.assertEqual(state_paths.choose_directory(root).path, str(canonical))

    def test_explicit_cli_is_literal_and_ignores_default_collisions(self):
        with patch.object(state_paths, "_directory_exists", side_effect=AssertionError("unrelated root")):
            choice = state_paths.web_root({}, explicit="./chosen//kv",
                                          environ={"CACHEDMOE_KV_DIR": "/environment"})
        self.assertEqual(choice.path, "./chosen//kv")
        self.assertEqual(choice.source, "explicit")

    def test_ready_default_is_authoritative(self):
        with patch.object(state_paths, "application_root", side_effect=AssertionError("duplicate resolve")):
            choice = state_paths.web_root({"state_root": "/engine-root",
                                            "state_root_source": "legacy-existing"},
                                           environ={"CACHEDMOE_KV_DIR": "/shell-root"})
        self.assertEqual(choice.path, "/engine-root")
        self.assertEqual(choice.source, "legacy-existing")

    def test_environment_explicit_root_uses_new_presence_priority(self):
        for env, expected in (({"DEEPMOE_KV_DIR": "./old//kv"}, "./old//kv"),
                              ({"CACHEDMOE_KV_DIR": "./new//kv"}, "./new//kv"),
                              ({"CACHEDMOE_KV_DIR": "./new//kv", "DEEPMOE_KV_DIR": "/old"}, "./new//kv")):
            with self.subTest(env=env), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(state_paths, "_directory_exists", side_effect=AssertionError("unrelated root")):
                choice = state_paths.web_root({"state_root": None}, environ=env)
                self.assertEqual(choice.path, expected)
                self.assertEqual(choice.source, "environment-explicit")

    def test_empty_new_value_suppresses_old_kv_root_and_still_selects_transcript_root(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            env = {"XDG_CACHE_HOME": tmp, "CACHEDMOE_KV_DIR": "", "DEEPMOE_KV_DIR": "/old"}
            choice = state_paths.web_root({"kv_disk_dir": "", "state_root": None}, environ=env)
            self.assertEqual(choice.path, str(Path(tmp) / "cachedmoe"))
            self.assertEqual(choice.source, "canonical-new")
            old_engine = state_paths.web_root({"kv_disk_dir": ""}, environ=env)
            self.assertEqual(old_engine, choice)

    def test_sibling_checkout_and_windows_paths_use_the_same_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "deepmoe"
            legacy.mkdir()
            self.assertEqual(state_paths.sibling_checkout(root / "cachedmoe-rename"), str(legacy))
            canonical = root / "cachedmoe"
            canonical.mkdir()
            self.assertEqual(state_paths.sibling_checkout(root / "cachedmoe-rename"), str(canonical))
        with patch.object(state_paths, "choose_paths", return_value=state_paths.StateRoot(
                state_paths.LEGACY_WINDOWS_CHECKOUT, "legacy-existing")):
            self.assertEqual(state_paths.windows_checkout_path("traces", "mixed"),
                             state_paths.LEGACY_WINDOWS_CHECKOUT + r"\traces\mixed")


class ExistingWebData(unittest.TestCase):
    @staticmethod
    def hashes(root):
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()}

    def test_four_transcripts_and_two_kv_snapshots_are_preserved_and_legacy_is_accessible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, legacy = Path(tmp), Path(tmp) / "deepmoe"
            chats, kv = legacy / "web_chat", legacy / "kv/model"
            chats.mkdir(parents=True); kv.mkdir(parents=True)
            for index in range(4):
                (chats / f"chat{index}.json").write_text(json.dumps({
                    "messages": [{"role": "user", "content": f"saved-{index}"}],
                    "ctx_ids": [index], "ctx_text": f"saved-{index}", "think": False}))
            for index in range(2):
                (kv / f"kv{index}.pkv").write_bytes(b"existing snapshot" + bytes([index]))
            before = self.hashes(legacy)
            args = SimpleNamespace(kv_dir="", system="", think=False, no_kv_disk=True)
            # Missing state_root models an older engine, including KV disabled.
            serve = SimpleNamespace(ready={"kv_disk_dir": ""})
            with patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}, clear=True), \
                    patch.object(server.threading, "Thread"):
                bridge = server.Bridge(serve, Mock(), args)
                self.assertEqual(bridge.state_root.path, str(legacy))
                for index in range(4):
                    self.assertEqual(bridge.state(f"chat{index}").messages[0]["content"], f"saved-{index}")
                self.assertFalse((root / "cachedmoe").exists())
                (root / "cachedmoe").mkdir()
                current = server.Bridge(serve, Mock(), args)
                self.assertEqual(current.state_root.path, str(root / "cachedmoe"))
                args.kv_dir = str(legacy)
                explicit = server.Bridge(serve, Mock(), args)
                self.assertEqual(explicit.state("chat0").messages[0]["content"], "saved-0")
            self.assertEqual(self.hashes(legacy), before)


if __name__ == "__main__":
    unittest.main()
