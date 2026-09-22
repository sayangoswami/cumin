//
// Created by Sayan Goswami on 22.09.2026.
//

#ifndef CUMIN_PARAMS_H
#define CUMIN_PARAMS_H

#include "prelude.h"

/**
 * Anchor/index parameters. Stored with the index (see index_t::dump/load)
 * so a query can never drift from the parameters an index was built with.
 */
struct params_t {
    u4 k = 15, s = 8, t = 0, downsample = 2;
    u4 win = 4000;
    float max_occ_pct = 99.9f;
    // Build-time only (not stored in the index -- see index_t::build): true
    // streams anchors through 256 on-disk radix buckets so the whole raw
    // anchor array never has to fit in RAM; false collects everything in one
    // parlay::sequence and sorts it in a single pass. Not persisted because
    // it's purely a build-time resource choice and has no bearing on how a
    // built index is queried.
    bool low_mem = true;

    params_t() = default;

    params_t(u4 k, u4 s, u4 t, u4 downsample, u4 win, float max_occ_pct, bool low_mem = true):
        k(k), s(s), t(t), downsample(downsample), win(win), max_occ_pct(max_occ_pct), low_mem(low_mem) {
        if (s >= k) log_error("s must be smaller than k");
        if (t > k - s) log_error("t must be in [0, %u]", k - s);
    }

    [[nodiscard]] float density() const {
        return 1.0f / (float)((k - s + 1) * downsample);
    }

    void dump(std::ostream &f) const {
        dump_values(f, k, s, t, downsample, win, max_occ_pct);
    }

    void load(std::istream &f) {
        load_values(f, &k, &s, &t, &downsample, &win, &max_occ_pct);
    }

    [[nodiscard]] std::string to_string() const {
        char buf[256];
        snprintf(buf, sizeof(buf), "k=%u s=%u t=%u downsample=%u win=%u density~1/%.0f",
                 k, s, t, downsample, win, 1.0f / density());
        return buf;
    }
};

#endif //CUMIN_PARAMS_H
