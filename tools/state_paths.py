"""Select current or existing legacy state directories without moving data."""
from dataclasses import dataclass
import ctypes
import ntpath
import os
from pathlib import Path

import runtime_env


APP_DIRECTORY = "cachedmoe"
LEGACY_DIRECTORY = "deepmoe"
LEGACY_WINDOWS_CHECKOUT = r"C:\Users\Asus\code\deepmoe"
WINDOWS_CHECKOUT = r"C:\Users\Asus\code\cachedmoe"


@dataclass(frozen=True)
class StateRoot:
    path: str
    source: str
    both_exist: bool = False


def cache_home(environ=None, *, windows=None, temp_dir=None):
    """Follow serve's external environment order, including empty values."""
    env = os.environ if environ is None else environ
    windows = os.name == "nt" if windows is None else windows
    if windows:
        base = env.get("LOCALAPPDATA") or env.get("TEMP")
    else:
        base = env.get("XDG_CACHE_HOME")
        if not base and env.get("HOME"):
            base = os.path.join(env["HOME"], ".cache")
    if base:
        return base
    # Keep the same explicit fallback order as the native selector. Neither
    # Python's cached tempfile search nor the STL's own ordering is authority.
    for key in ("TMPDIR", "TEMP", "TMP", "TEMPDIR"):
        if env.get(key):
            return env[key]
    if temp_dir is not None:
        return temp_dir
    if not windows:
        return "/tmp"
    size = 32768
    buffer = ctypes.create_unicode_buffer(size)
    length = ctypes.windll.kernel32.GetTempPathW(size, buffer)
    if not length or length >= size:
        raise OSError("cannot resolve the Windows system temporary directory")
    return buffer.value


def _directory_exists(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    if not path.is_dir():
        raise ValueError(f"state path exists but is not a directory: {path}")
    return True


def choose_paths(current, legacy):
    """Inspect both roots; never create, rename, remove or overwrite either."""
    current_path, legacy_path = Path(current), Path(legacy)
    current_exists = _directory_exists(current_path)
    legacy_exists = _directory_exists(legacy_path)
    if current_exists:
        return StateRoot(os.fspath(current), "canonical-existing", legacy_exists)
    if legacy_exists:
        return StateRoot(os.fspath(legacy), "legacy-existing")
    return StateRoot(os.fspath(current), "canonical-new")


def choose_directory(base, current=APP_DIRECTORY, legacy=LEGACY_DIRECTORY):
    return choose_paths(Path(base) / current, Path(base) / legacy)


def windows_checkout_path(*parts):
    root = choose_paths(WINDOWS_CHECKOUT, LEGACY_WINDOWS_CHECKOUT)
    return ntpath.join(root.path, *parts)


def sibling_checkout(repo):
    return choose_directory(Path(repo).parent).path


def application_root(environ=None, *, windows=None, temp_dir=None):
    return choose_directory(cache_home(environ, windows=windows, temp_dir=temp_dir))


def web_root(ready=None, *, explicit="", environ=None):
    """Explicit directories stay literal; otherwise serve's ready is authority."""
    if explicit:
        return StateRoot(explicit, "explicit")
    ready = ready or {}
    root = ready.get("state_root")
    if isinstance(root, str) and root:
        source = ready.get("state_root_source")
        return StateRoot(root, source if isinstance(source, str) and source else "engine-ready",
                         bool(ready.get("state_root_both_exist", False)))
    kv_dir = runtime_env.getenv("CACHEDMOE_KV_DIR", environ=environ)
    if kv_dir:
        return StateRoot(kv_dir, "environment-explicit")
    # Older executables do not publish state_root. Keep the same filesystem
    # policy for them rather than abandoning the user's existing directory.
    return application_root(environ)
