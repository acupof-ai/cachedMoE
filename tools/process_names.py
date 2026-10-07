"""Executable names used by GPU idle checks and owned-engine guards.

Linux exposes at most 15 bytes in comm. Keep the legacy names during the
executable migration; comm alone never authorizes a signal to a process.
"""
import re


ENGINE_NAMES = ("cachedmoe", "deepmoe")
GPU_EXECUTABLE_NAMES = ENGINE_NAMES + (
    "cachedmoe-tested", "deepmoe-tested", "cachedmoe_tests", "deepmoe_tests",
    "nvme_bench", "io_dst_bench", "bw_matrix", "heap_capacity",
    "kernel_bench", "attn_bench", "prefill_bench", "dspark_bench", "dspark_grid_probe",
    "draft_head_capture", "draft_head_bench",
    "residency_probe", "sharing_probe", "capacity_probe", "hostflag_probe",
    "model_probe", "envcheck", "vulkaninfo",
)
GPU_COMMS = tuple(dict.fromkeys(name[:15] for name in GPU_EXECUTABLE_NAMES))
GPU_COMM_PATTERN = "|".join(re.escape(name) for name in GPU_COMMS)


def is_engine_comm(name):
    return name in ENGINE_NAMES


def is_gpu_process(name):
    """Accept full executable names, Windows .exe names, and Linux comm."""
    name = name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name in GPU_EXECUTABLE_NAMES or name in GPU_COMMS
