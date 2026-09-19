Track DX -- E: (USB NVMe) failure diagnosis, 2026-09-19 19:00.

dx_probe.cpp        standalone Win32 probe: N handles FILE_FLAG_NO_BUFFERING|
                    FILE_FLAG_OVERLAPPED on ONE IOCP, QD-limited 4 MiB random
                    reads, optional interleaved 12 KiB reads, 5 s throughput
                    log, stops on the FIRST win32 error (prints code/len/off),
                    and a watchdog that declares a STALL after --stall-s with
                    no completion.  It touches NO deepmoe code: it exists to
                    show the failure is not the engine's.
                    build:
                      zig c++ -target x86_64-windows-gnu -std=c++20 -O2 -w \
                        dx_probe.cpp -o dx_probe.exe

d_control_48h_qd24.txt            D: 48 handles QD 24, 60 s -- PASS 4.53 GB/s
d_control_48h_qd24_small12k.txt   D: same + 12 KiB every 8th read, 30 s -- PASS 4.57 GB/s
e_a_cold_48h_qd24.txt             E: 48 handles QD 24, cold -- WEDGED at t~17 s
winevent_system_48h.txt           System log, storage/USB providers, 48 h
device_state.txt                  disks, UAS bridge, power settings, what could NOT be read

NOT RUN, on purpose: the E: arms (b) 12 KiB interleave and (c) warm repeat.
Arm (a) wedged the device; the task's rule is stop, do not retry, replug.
