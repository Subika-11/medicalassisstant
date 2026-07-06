#pragma once

// Attempts to pin the calling thread to the device's "big" CPU cores
// (heuristically: the cores reporting the highest cpuinfo_max_freq under
// /sys/devices/system/cpu/). Returns true if an affinity mask was
// successfully applied, false if detection or sched_setaffinity failed
// (some OEM kernels block this from unprivileged apps - that's fine, the
// caller should treat failure as non-fatal and just fall back to whatever
// the scheduler does by default).
bool pin_thread_to_big_cores();
