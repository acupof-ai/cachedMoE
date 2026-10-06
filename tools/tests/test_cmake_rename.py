#!/usr/bin/env python3
"""CPU-only configure regressions for canonical and legacy CMake cache keys."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "cmake" / "cachedmoe_cache_compat.cmake"

FIXTURE = r'''
cmake_minimum_required(VERSION 3.25)
project(cache_alias_fixture NONE)
include("@HELPER@")
@PRELUDE@
cachedmoe_option(ENABLE_VULKAN "Fixture backend" ON)
cachedmoe_cache(PYTHON FILEPATH "Fixture oracle interpreter" "")
cachedmoe_cache(TEXT STRING "Fixture string" "default-text")
cachedmoe_resolve_cache(FUSION_PYTHON FILEPATH "Fixture shader interpreter")
if(NOT DEFINED CACHEDMOE_FUSION_PYTHON)
    set(CACHEDMOE_FUSION_PYTHON "default-fusion" CACHE FILEPATH "Fixture shader interpreter")
endif()
cachedmoe_resolve_cache(HAVE_DSTORAGE_H INTERNAL "Fixture detected header")
if(NOT DEFINED CACHEDMOE_HAVE_DSTORAGE_H)
    set(CACHEDMOE_HAVE_DSTORAGE_H "detected" CACHE INTERNAL "Fixture detected header")
endif()
# Several shader function scopes can resolve the same key in one configure.
function(resolve_shader_python)
    cachedmoe_resolve_cache(FUSION_PYTHON FILEPATH "Fixture shader interpreter")
endfunction()
resolve_shader_python()
resolve_shader_python()
file(WRITE "${CMAKE_BINARY_DIR}/snapshot.txt" "")
foreach(suffix ENABLE_VULKAN PYTHON TEXT FUSION_PYTHON HAVE_DSTORAGE_H)
    file(APPEND "${CMAKE_BINARY_DIR}/snapshot.txt"
         "${suffix}.effective=${CACHEDMOE_${suffix}}\n")
    foreach(prefix CACHEDMOE DEEPMOE)
        set(key "${prefix}_${suffix}")
        if(DEFINED CACHE{${key}})
            get_property(cache_type CACHE "${key}" PROPERTY TYPE)
            get_property(cache_value CACHE "${key}" PROPERTY VALUE)
            file(APPEND "${CMAKE_BINARY_DIR}/snapshot.txt"
                 "${key}.type=${cache_type}\n${key}.value=${cache_value}\n")
        else()
            file(APPEND "${CMAKE_BINARY_DIR}/snapshot.txt" "${key}.type=ABSENT\n")
        endif()
    endforeach()
endforeach()
'''


class CMakeCacheRename(unittest.TestCase):
    def setUp(self):
        self.cmake = shutil.which("cmake")
        self.assertIsNotNone(self.cmake, "cmake is required for these configure tests")
        self.folder = tempfile.TemporaryDirectory(prefix="cachedmoe-cmake-")
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.source = self.base / "source"
        self.build = self.base / "build"
        self.source.mkdir()
        self.write_source()

    def write_source(self, prelude=""):
        source = FIXTURE.replace("@HELPER@", HELPER.as_posix())
        (self.source / "CMakeLists.txt").write_text(
            source.replace("@PRELUDE@", prelude), encoding="utf-8")

    def configure(self, *cache_args):
        child = subprocess.run(
            [self.cmake, "-S", str(self.source), "-B", str(self.build), *cache_args],
            cwd=self.base, capture_output=True, text=True, timeout=30)
        output = child.stdout + child.stderr
        self.assertEqual(child.returncode, 0, output)
        rows = (self.build / "snapshot.txt").read_text(encoding="utf-8").splitlines()
        return dict(row.split("=", 1) for row in rows), output

    def test_fresh_defaults_are_canonical_cache_entries(self):
        values, output = self.configure()
        self.assertEqual(values["ENABLE_VULKAN.effective"], "ON")
        self.assertEqual(values["CACHEDMOE_ENABLE_VULKAN.type"], "BOOL")
        self.assertEqual(values["PYTHON.effective"], "")
        self.assertEqual(values["CACHEDMOE_PYTHON.type"], "FILEPATH")
        self.assertEqual(values["CACHEDMOE_TEXT.type"], "STRING")
        self.assertEqual(values["CACHEDMOE_HAVE_DSTORAGE_H.type"], "INTERNAL")
        self.assertEqual(values["DEEPMOE_ENABLE_VULKAN.type"], "ABSENT")
        self.assertNotIn("deprecated", output)

    def test_legacy_typed_values_keep_type_and_do_not_create_canonical_cache(self):
        rows = (
            ("ENABLE_VULKAN", "BOOL", "OFF"),
            ("PYTHON", "FILEPATH", "relative-python"),
            ("TEXT", "STRING", "legacy-text"),
            ("FUSION_PYTHON", "STRING", "legacy-fusion"),
            ("HAVE_DSTORAGE_H", "INTERNAL", "0"),
        )
        values, _ = self.configure(*(f"-DDEEPMOE_{key}:{kind}={value}"
                                     for key, kind, value in rows))
        for key, kind, value in rows:
            with self.subTest(key=key):
                self.assertEqual(values[f"{key}.effective"], value)
                self.assertEqual(values[f"DEEPMOE_{key}.type"], kind)
                self.assertEqual(values[f"DEEPMOE_{key}.value"], value)
                self.assertEqual(values[f"CACHEDMOE_{key}.type"], "ABSENT")

    def test_explicit_new_empty_wins_over_nonempty_legacy(self):
        values, output = self.configure(
            "-DCACHEDMOE_PYTHON:FILEPATH=", "-DDEEPMOE_PYTHON:FILEPATH=private-old-path",
            "-DCACHEDMOE_TEXT:STRING=", "-DDEEPMOE_TEXT:STRING=private-old-text")
        self.assertEqual(values["PYTHON.effective"], "")
        self.assertEqual(values["TEXT.effective"], "")
        self.assertEqual(values["DEEPMOE_PYTHON.value"], "private-old-path")
        self.assertNotIn("private-old-path", output)
        self.assertNotIn("private-old-text", output)
        self.assertIn("ignored because CACHEDMOE_PYTHON is defined", " ".join(output.split()))
        self.assertNotIn("Manually-specified variables were not used", output)

    def test_new_typed_cache_wins_without_force_or_type_conversion(self):
        values, _ = self.configure(
            "-DCACHEDMOE_ENABLE_VULKAN:STRING=OFF", "-DDEEPMOE_ENABLE_VULKAN:BOOL=ON",
            "-DCACHEDMOE_HAVE_DSTORAGE_H:STRING=new-result",
            "-DDEEPMOE_HAVE_DSTORAGE_H:INTERNAL=old-result")
        self.assertEqual(values["ENABLE_VULKAN.effective"], "OFF")
        self.assertEqual(values["CACHEDMOE_ENABLE_VULKAN.type"], "STRING")
        self.assertEqual(values["HAVE_DSTORAGE_H.effective"], "new-result")
        self.assertEqual(values["CACHEDMOE_HAVE_DSTORAGE_H.type"], "STRING")
        self.assertEqual(values["DEEPMOE_HAVE_DSTORAGE_H.value"], "old-result")

    def test_untyped_legacy_bool_acquires_the_original_declared_type(self):
        values, _ = self.configure("-DDEEPMOE_ENABLE_VULKAN=OFF")
        self.assertEqual(values["ENABLE_VULKAN.effective"], "OFF")
        self.assertEqual(values["DEEPMOE_ENABLE_VULKAN.type"], "BOOL")
        self.assertEqual(values["CACHEDMOE_ENABLE_VULKAN.type"], "ABSENT")

    def test_untyped_legacy_filepath_uses_cmake_path_normalization(self):
        values, _ = self.configure("-DDEEPMOE_PYTHON=relative-python",
                                   "-DDEEPMOE_FUSION_PYTHON=relative-fusion")
        for key, relative in (("PYTHON", "relative-python"),
                              ("FUSION_PYTHON", "relative-fusion")):
            with self.subTest(key=key):
                self.assertEqual(values[f"{key}.effective"], str(self.base / relative))
                self.assertEqual(values[f"DEEPMOE_{key}.type"], "FILEPATH")
                self.assertEqual(values[f"CACHEDMOE_{key}.type"], "ABSENT")

    def test_untyped_canonical_filepath_bool_and_internal_get_declared_types(self):
        values, _ = self.configure("-DCACHEDMOE_PYTHON=relative-python",
                                   "-DCACHEDMOE_ENABLE_VULKAN=OFF",
                                   "-DCACHEDMOE_HAVE_DSTORAGE_H=0")
        self.assertEqual(values["PYTHON.effective"], str(self.base / "relative-python"))
        self.assertEqual(values["CACHEDMOE_PYTHON.type"], "FILEPATH")
        self.assertEqual(values["CACHEDMOE_ENABLE_VULKAN.type"], "BOOL")
        self.assertEqual(values["CACHEDMOE_HAVE_DSTORAGE_H.type"], "INTERNAL")
        self.assertEqual(values["HAVE_DSTORAGE_H.effective"], "0")

    def test_legacy_only_build_can_be_reconfigured_with_legacy_arguments(self):
        first, _ = self.configure("-DDEEPMOE_ENABLE_VULKAN:BOOL=ON",
                                  "-DDEEPMOE_PYTHON:FILEPATH=first-python",
                                  "-DDEEPMOE_HAVE_DSTORAGE_H:INTERNAL=1")
        second, _ = self.configure("-DDEEPMOE_ENABLE_VULKAN:BOOL=OFF",
                                   "-DDEEPMOE_PYTHON:FILEPATH=second-python",
                                   "-DDEEPMOE_HAVE_DSTORAGE_H:INTERNAL=0")
        for key, first_value, second_value in (("ENABLE_VULKAN", "ON", "OFF"),
                                              ("PYTHON", "first-python", "second-python"),
                                              ("HAVE_DSTORAGE_H", "1", "0")):
            with self.subTest(key=key):
                self.assertEqual(first[f"{key}.effective"], first_value)
                self.assertEqual(second[f"{key}.effective"], second_value)
                self.assertEqual(second[f"CACHEDMOE_{key}.type"], "ABSENT")

    def test_existing_canonical_default_wins_over_a_later_legacy_argument(self):
        self.configure()
        values, _ = self.configure("-DDEEPMOE_ENABLE_VULKAN:BOOL=OFF",
                                   "-DDEEPMOE_PYTHON:FILEPATH=legacy-python")
        self.assertEqual(values["ENABLE_VULKAN.effective"], "ON")
        self.assertEqual(values["PYTHON.effective"], "")
        self.assertEqual(values["DEEPMOE_ENABLE_VULKAN.value"], "OFF")

    def test_canonical_normal_empty_overrides_both_cache_spellings(self):
        self.write_source('set(CACHEDMOE_PYTHON "")')
        values, _ = self.configure("-DCACHEDMOE_PYTHON:FILEPATH=new-cache-python",
                                   "-DDEEPMOE_PYTHON:FILEPATH=legacy-python")
        self.assertEqual(values["PYTHON.effective"], "")
        self.assertEqual(values["CACHEDMOE_PYTHON.value"], "new-cache-python")

    def test_legacy_normal_variable_is_not_a_cache_override(self):
        self.write_source('set(DEEPMOE_ENABLE_VULKAN OFF)')
        values, output = self.configure()
        self.assertEqual(values["ENABLE_VULKAN.effective"], "ON")
        self.assertEqual(values["DEEPMOE_ENABLE_VULKAN.type"], "ABSENT")
        self.assertNotIn("deprecated", output)

    def test_repeated_resolution_warns_once_and_does_not_print_values(self):
        _, output = self.configure("-DDEEPMOE_FUSION_PYTHON:FILEPATH=private-fusion")
        self.assertEqual(output.count("DEEPMOE_FUSION_PYTHON is deprecated"), 1)
        self.assertNotIn("private-fusion", output)

    def test_legacy_module_path_includes_the_canonical_options(self):
        module = (ROOT / "cmake" / "deepmoe_options.cmake").as_posix()
        (self.source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\nproject(options_fixture NONE)\n'
            f'include("{module}")\n'
            'cachedmoe_report()\n'
            'list(LENGTH CACHEDMOE_SHADERS shader_count)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/snapshot.txt" '
            '"standard=${CACHEDMOE_CXX_STANDARD}\\nshaders=${shader_count}\\n")\n',
            encoding="utf-8")
        values, output = self.configure("-DDEEPMOE_BUILD_TESTS:BOOL=OFF")
        self.assertEqual(values, {"standard": "23", "shaders": "52"})
        self.assertIn("cachedmoe: tests=OFF", output)


if __name__ == "__main__":
    unittest.main()
