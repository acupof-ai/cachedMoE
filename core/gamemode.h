#pragma once

// Feral GameMode around the GPU-heavy work of a long-lived process.
//
// Why (docs/STATUS.md §7 0h, bench/results/linux/perf/): a chat decode waits on
// NVMe for a few ms before about a third of its layers, and RADV's automatic
// power management drops the iGPU to its 600 MHz floor in that time. The MoE
// dispatches that follow the wait then run at the low clock while it ramps --
// moe_down 885 us against 263 us, the late gate/up 305 against 145 -- which was
// ~10 ms per chat token that the hot step (never idle) could not see. Pinning
// power_dpm_force_performance_level to "high" removes it, but pinning it for the
// machine's life would keep a laptop's GPU at full clock at the desktop.
//
// GameMode is the scoped version of that pin: gamemoded (D-Bus activated) runs
// its gpuclockctl helper through polkit, which lets the `gamemode` group do so
// without a password, and it drops the pin when the LAST client ends -- or when
// the client process dies, because the daemon watches its pid. So a serve that
// is killed mid-reply cannot leave the GPU pinned. A request costs ~30 ms each
// way. The GPU part is opt-in on the daemon's side (~/.config/gamemode.ini
// [gpu] apply_gpu_optimisations=accept-responsibility, amd_performance_level=
// high); without it this is a no-op request.
//
// Loaded with dlopen, so nothing links against libgamemode and a machine
// without it (Windows, or Linux without the package) just gets a no-op.
// CACHEDMOE_GAMEMODE=0 turns it off.

#include "core/env.h"
#include <cstdio>
#include <cstdlib>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace deepmoe {

class GameModeScope {
public:
    GameModeScope() {
        Api& a = api();
        if (a.start && a.start() == 0) active_ = true;
    }
    ~GameModeScope() {
        if (active_) api().end();
    }
    GameModeScope(const GameModeScope&) = delete;
    GameModeScope& operator=(const GameModeScope&) = delete;

    bool active() const { return active_; }

private:
    struct Api {
        int (*start)() = nullptr;
        int (*end)() = nullptr;
    };
    static Api& api() {
        static Api a = [] {
            Api r;
#if defined(__linux__)
            if (const char* e = ::deepmoe::environment::get("CACHEDMOE_GAMEMODE"); e && *e == '0') return r;
            void* h = dlopen("libgamemode.so.0", RTLD_NOW | RTLD_LOCAL);
            if (!h) return r;
            r.start = reinterpret_cast<int (*)()>(dlsym(h, "real_gamemode_request_start"));
            r.end   = reinterpret_cast<int (*)()>(dlsym(h, "real_gamemode_request_end"));
            if (!r.start || !r.end) r = Api{};
            else std::fprintf(stderr, "[INF] gamemode: GPU work runs inside a GameMode request "
                                      "(CACHEDMOE_GAMEMODE=0 turns this off)\n");
#endif
            return r;
        }();
        return a;
    }
    bool active_ = false;
};

}  // namespace deepmoe
