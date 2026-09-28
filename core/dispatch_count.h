#pragma once

// Workgroups and dispatches recorded on this thread since the last reset.
// gpu::CommandBuffer::dispatch adds to it and trace::Tracer reads it around
// every traced region, so the trace knows each stage's dispatch geometry
// without any call site passing it -- the workgroup count is what the
// per-dispatch GPU model (tools/gpu_model.py) needs next to the stage's bytes.

#include <cstdint>

namespace deepmoe {

struct DispatchCount {
    uint64_t groups     = 0;
    uint32_t dispatches = 0;
};
inline thread_local DispatchCount g_dispatch_count;

}  // namespace deepmoe
