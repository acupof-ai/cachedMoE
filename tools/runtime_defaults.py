"""Shared factual defaults and named launch policies for Python consumers.

Environment aliases/layers belong to runtime_env. Profiles deliberately keep
production resets separate from benchmark inheritance and resource controls.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
import hashlib
import os
from pathlib import Path
import sys
from types import MappingProxyType

if __package__:
    from . import runtime_env, runtime_facts
else:
    import runtime_env
    import runtime_facts

sys.dont_write_bytecode = True
sys.modules.setdefault("runtime_defaults", sys.modules[__name__])
sys.modules.setdefault("tools.runtime_defaults", sys.modules[__name__])

REPO = Path(__file__).resolve().parents[1]
MODEL_NAME = runtime_facts.MODEL_NAME
MAX_CONTEXT = runtime_facts.MAX_CONTEXT
BENCH_CONTEXT = runtime_facts.SESSION_CONTEXT
CACHE_SLOTS = 5500
GPU_PREFILL_MIN = 16
ACCEPT_TOP_K = runtime_facts.ACCEPT_TOP_K
DRAFT_BLOCK_SIZE = runtime_facts.DRAFT_BLOCK_SIZE
PRODUCTION_DRAFT_TOKENS = 2
KV_DISK_GB = runtime_facts.KV_DISK_BYTES // (1 << 30)
WEB_PORT = 8080
KV_BYTES_PER_TOKEN = 3200
PREFILL_MS_PER_TOKEN = 24.0

REQUEST_DEFAULTS = MappingProxyType({
    "temperature": runtime_facts.SAMPLING_TEMPERATURE,
    "top_p": runtime_facts.TOP_P,
    "max_tokens": runtime_facts.WEB_MAX_TOKENS,
    "reasoning_effort": 75,
})
EFFORT_PRESETS = (("Low", 50), ("High", REQUEST_DEFAULTS["reasoning_effort"]), ("Max", 100))
REQUEST_CONSTRAINTS = MappingProxyType({
    "temperature": MappingProxyType(dict(min=0, max=2, step=.05)),
    "top_p": MappingProxyType(dict(min=0, max=1, step=.01)),
    "max_tokens": MappingProxyType(dict(min=1, max=8192, step=1)),
    "reasoning_effort": MappingProxyType(dict(min=1, max=EFFORT_PRESETS[-1][1], step=1)),
})


def request_policy():
    # JSON consumers get copies: a request/UI cannot mutate startup policy.
    return dict(defaults=dict(REQUEST_DEFAULTS),
                constraints={key: dict(value) for key, value in REQUEST_CONSTRAINTS.items()},
                effort_presets=[dict(value=value, label=f"{label} · {value}")
                                for label, value in EFFORT_PRESETS])


def reasoning_effort(value):
    bounds = REQUEST_CONSTRAINTS["reasoning_effort"]
    if type(value) is not int or not bounds["min"] <= value <= bounds["max"]:
        raise ValueError(f"Reasoning effort must be an integer from {bounds['min']} to {bounds['max']}")
    return value


def model_fallback(*, windows=None):
    windows = os.name == "nt" if windows is None else windows
    return "D:\\models\\" + MODEL_NAME if windows else os.path.expanduser("~/models/" + MODEL_NAME)


def model_directory(environ=None, *, empty_fallback=False, windows=None):
    fallback = model_fallback(windows=windows)
    selected = runtime_env.resolve("MODEL_DIR", environ)
    return selected.value if selected.present and (selected.value or not empty_fallback) else fallback


def load_encoding(model):
    """Load the selected checkpoint's read-only renderer, keyed by its real path.

    ``import encoding`` would silently reuse the first model's module if a later
    benchmark selects another checkpoint in the same Python process.
    """
    path = (Path(model) / "encoding/encoding.py").resolve()
    name = "_cachedmoe_encoding_" + hashlib.sha256(str(path).encode()).hexdigest()
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def executable(repo=REPO, *, windows=None, tests=False):
    windows = os.name == "nt" if windows is None else windows
    name = "cachedmoe_tests" if tests else "cachedmoe"
    return str(Path(repo) / "build" / ("tests" if tests else "") / (name + (".exe" if windows else "")))


def mirror_directory():
    return "/mnt/deepmoe2/models/" + MODEL_NAME


@dataclass(frozen=True)
class LaunchConfig:
    exe: str
    model: str
    shader_dir: str
    environment: Mapping[str, str]
    model_source: str

    def receipt(self):
        return dict(exe=self.exe, model=self.model, shader_dir=self.shader_dir,
                    model_source=self.model_source)


def resolve_launch(exe=None, environ=None, *, model=None, shader_dir=None, repo=REPO):
    env = dict(os.environ if environ is None else environ)
    selected = runtime_env.resolve("MODEL_DIR", env)
    model_value = model if model is not None else (selected.value if selected.present else model_fallback())
    source = "explicit-cli" if model is not None else selected.source if selected.present else "default"
    exe = str(exe) if exe is not None else executable(repo)
    shaders = shader_dir or runtime_env.getenv("SHADER_DIR", environ=env) or str(Path(exe).parent / "shaders")
    return LaunchConfig(exe, model_value, str(shaders), MappingProxyType(env), source)


def cli_value(arguments, flag):
    """The native CLI applies repeated space-separated flags in order."""
    value = None
    for index, item in enumerate(arguments):
        if item == flag and index + 1 < len(arguments):
            value = arguments[index + 1]
    return value


_CONTROL_VALUES = MappingProxyType({
    "DSPARK_ONECB": "1", "BATCH_GPU_ROUTE": "1", "DSPARK_MEGA": "0",
    "DSPARK_PROFILE": "0", "DSPARK_TRIM_TAIL": "1", "SPEC_GPU_READOUT": "1",
    "MASK_DYNAMIC_LRU": "1", "MIRROR_AUTO": "0", "IO_ENGRAM_DEADLINE": "0",
    "SPEC_DIAGNOSTICS": "", "MGT_PAIR_DOT": "0", "MGT_ATTN_CM": "0", "MGT_FOLD_SCALE": "0",
})
_DRAFT = ("DSPARK_ONECB", "BATCH_GPU_ROUTE", "DSPARK_MEGA", "DSPARK_PROFILE")
_READOUT = ("DSPARK_TRIM_TAIL", "SPEC_GPU_READOUT")
_MGT = ("MGT_PAIR_DOT", "MGT_ATTN_CM", "MGT_FOLD_SCALE")
_PROFILE_KEYS = MappingProxyType({
    "production": _DRAFT + _READOUT + _MGT + ("MASK_DYNAMIC_LRU", "MIRROR_AUTO"),
    "spec-resources": _DRAFT + _READOUT + ("MASK_DYNAMIC_LRU", "IO_ENGRAM_DEADLINE", "SPEC_DIAGNOSTICS"),
    "longtest": _DRAFT + _MGT + ("MASK_DYNAMIC_LRU",),
    "thermal": ("MIRROR_AUTO", "DSPARK_MEGA"),
})


def profile_default(name, option):
    """Typed CLI defaults retain each named launch policy's deliberate scope."""
    if option == "spec_k":
        return PRODUCTION_DRAFT_TOKENS if name == "production" else DRAFT_BLOCK_SIZE
    if option == "gpu_route":
        return 0 if name == "production" else int(_CONTROL_VALUES["BATCH_GPU_ROUTE"])
    if option == "onecb":
        return int(_CONTROL_VALUES["DSPARK_ONECB"])
    if option == "mask_cache":
        return "fixed" if name == "longtest" else "dynamic"
    raise KeyError(option)


def profile_controls(name, *, speculative=True, onecb=None, gpu_route=None, mask_cache="dynamic"):
    controls = {key: _CONTROL_VALUES[key] for key in _PROFILE_KEYS[name]}
    if "MASK_DYNAMIC_LRU" in controls:
        controls["MASK_DYNAMIC_LRU"] = (
            "0" if mask_cache == "fixed" else _CONTROL_VALUES["MASK_DYNAMIC_LRU"])
    if not speculative and "DSPARK_ONECB" in controls:
        controls["DSPARK_ONECB"] = controls["BATCH_GPU_ROUTE"] = "0"
    else:
        if onecb is not None:
            controls["DSPARK_ONECB"] = str(onecb)
        if gpu_route is not None:
            controls["BATCH_GPU_ROUTE"] = str(gpu_route)
    return {runtime_env.aliases(key)[0]: value for key, value in controls.items()}


def profile_environment(name, inherited, *, overrides=None, **options):
    if name == "production":
        env = {key: value for key, value in inherited.items() if not runtime_env.is_control(key)}
    elif name == "longtest":
        env = runtime_env.canonicalized(inherited)
        runtime_env.clear(env, "SPEC_DIAGNOSTICS", "ROUTE_DUMP")
    else:
        env = dict(inherited)
    runtime_env.apply_overrides(env, profile_controls(name, **options))
    if overrides:
        runtime_env.apply_overrides(env, overrides)
    return env


def parse_environment(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--env requires KEY=VALUE")
        key, value = item.split("=", 1)
        result[key] = value
    return result
