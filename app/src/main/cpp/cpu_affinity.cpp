#include "cpu_affinity.h"

#include <sched.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <android/log.h>

#define LOG_TAG "medrag_cpu_affinity"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace {

// Reads /sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq for each core.
// Returns -1 for a core if the file can't be read (e.g. blocked by SELinux
// policy on some OEM ROMs).
long read_max_freq_khz(int core_id) {
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", core_id);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    long freq = -1;
    if (fscanf(f, "%ld", &freq) != 1) freq = -1;
    fclose(f);
    return freq;
}

} // namespace

bool pin_thread_to_big_cores() {
    long num_cores = sysconf(_SC_NPROCESSORS_CONF);
    if (num_cores <= 0) {
        LOGW("could not determine core count, skipping affinity pinning");
        return false;
    }

    std::vector<std::pair<int, long>> core_freqs;
    for (int i = 0; i < num_cores; ++i) {
        long freq = read_max_freq_khz(i);
        if (freq > 0) {
            core_freqs.emplace_back(i, freq);
        }
    }

    if (core_freqs.empty()) {
        LOGW("cpufreq info unavailable on this device, skipping affinity pinning");
        return false;
    }

    // Heuristic: "big" cores are whichever distinct max-frequency tier is
    // the highest. On most heterogeneous ARM SoCs there are 2-3 tiers
    // (e.g. 2 big + 6 little, or 1 prime + 3 big + 4 little).
    long highest_freq = std::max_element(
        core_freqs.begin(), core_freqs.end(),
        [](auto& a, auto& b) { return a.second < b.second; })->second;

    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    int big_core_count = 0;
    for (auto& [core_id, freq] : core_freqs) {
        if (freq == highest_freq) {
            CPU_SET(core_id, &cpu_set);
            big_core_count++;
        }
    }

    if (big_core_count == 0) {
        LOGW("no big cores identified, skipping affinity pinning");
        return false;
    }

    int result = sched_setaffinity(0 /* calling thread */, sizeof(cpu_set_t), &cpu_set);
    if (result != 0) {
        LOGW("sched_setaffinity failed (errno=%d) - some OEM kernels disallow this", errno);
        return false;
    }

    LOGI("pinned thread to %d big core(s) at %ld kHz", big_core_count, highest_freq);
    return true;
}
