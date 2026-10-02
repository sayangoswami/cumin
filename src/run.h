//
// Created by Sayan Goswami on 01.10.2026.
//

#ifndef CUMIN_RUN_H
#define CUMIN_RUN_H

#include "prelude.h"
#include "params.h"
#include "cumin.h"
#include "parlay/sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include <algorithm>
#include <limits>
#include <unordered_map>

/*
 * Pieces shared by the static index (index_t) and the dynamic one
 * (dyn_index_t): the sorted-key/CSR/packed-window-id run layout, the
 * window-id allocator, anchor extraction for a reference record, and the
 * per-read vote tally.
 */

/**
 * A simple frequency counter (heavy-hitter / mode finder), ported from
 * collinearity/src/index.h. One instance lives per worker thread and is
 * reused (via reset()) across every read that thread processes: the
 * overwhelming majority of reads only ever produce a handful of distinct
 * candidate windows, so this keeps a small fixed-capacity inline buffer,
 * scanned linearly (no hashing, no allocation), and only falls back to a
 * hash map for the rare read that overflows it -- see index_t::vote_scalar.
 *
 * @tparam T key type
 * @tparam CAP inline buffer capacity before falling back to the hash map
 */
template <typename T, u4 CAP = 64>
struct heavyhitter_ht_t {
    T keys[CAP];
    u4 counts[CAP];
    u4 n = 0;
    bool overflowed = false;
    std::unordered_map<T, u4> overflow_map;
    T top_key = (T)-1;
    u4 top_count = 0;

    // Ties go to the higher key. For window ids this is what keeps the
    // per-contig padding windows (see index_t) from ever being reported:
    // a padding window's count never exceeds that of the real window just
    // above it. It also makes the winner independent of insertion order.
    inline void bump(const T key, const u4 count) {
        if (count > top_count || (count == top_count && key > top_key)) top_count = count, top_key = key;
    }

    void insert(const T key) {
        if (!overflowed) {
            for (u4 i = 0; i < n; ++i) {
                if (keys[i] == key) {
                    bump(key, ++counts[i]);
                    return;
                }
            }
            if (n < CAP) {
                keys[n] = key;
                counts[n] = 1;
                ++n;
                bump(key, 1);
                return;
            }
            overflowed = true;
            overflow_map.reserve(CAP * 2);
            for (u4 i = 0; i < n; ++i) overflow_map[keys[i]] = counts[i];
            n = 0;
        }
        bump(key, ++overflow_map[key]);
    }

    void reset() {
        n = 0;
        if (overflowed) {
            overflow_map.clear();
            overflowed = false;
        }
        top_key = (T)-1;
        top_count = 0;
    }
};

/**
 * Fixed-width bit-packed array of unsigned integers, 1..32 bits each.
 * Values are packed LSB-first into 64-bit words; one trailing padding word
 * lets get() always do a single unaligned 8-byte load (little-endian only).
 */
struct packed_array_t {
    u4 width = 0;
    u8 n = 0;
    parlay::sequence<u8> words;

    /** Bits needed to represent every value in [0, max_val]. */
    static u4 bits_for(u8 max_val) {
        u4 b = 1;
        while (b < 64 && (max_val >> b) != 0) ++b;
        return b;
    }

    void pack(const parlay::sequence<u4> &vals, u4 width_) {
        expect(width_ >= 1 && width_ <= 32);
        width = width_;
        n = vals.size();
        // 64 values of `width` bits fill exactly `width` whole words, so each
        // block of 64 is packed independently of its neighbours.
        u8 nblocks = (n + 63) / 64;
        words = parlay::sequence<u8>(nblocks * width + 1, 0);
        parlay::parallel_for(0, nblocks, [&](size_t b) {
            u8 *w = words.data() + b * width;
            u8 lo = b * 64, hi = MIN(n, lo + 64);
            for (u8 i = lo; i < hi; ++i) {
                u8 v = vals[i], bit = (i - lo) * width, off = bit & 63;
                w[bit >> 6] |= v << off;
                if (off + width > 64) w[(bit >> 6) + 1] |= v >> (64 - off);
            }
        });
    }

    [[nodiscard]] inline u4 get(u8 i) const {
        u8 bit = i * width, x;
        std::memcpy(&x, reinterpret_cast<const char*>(words.data()) + (bit >> 3), sizeof(x));
        return (u4)((x >> (bit & 7)) & ((1ULL << width) - 1));
    }

    [[nodiscard]] u8 nbytes() const { return words.size() * sizeof(u8); }
};

/**
 * Compressed-sparse-row offsets (U+1 of them), stored 32-bit unless the
 * number of entries they index reaches 2^32, in which case 64-bit.
 */
struct offsets_t {
    bool wide = false;
    parlay::sequence<u4> o32;
    parlay::sequence<u8> o64;

    [[nodiscard]] inline u8 operator[](size_t i) const { return wide ? o64[i] : (u8)o32[i]; }
    [[nodiscard]] size_t size() const { return wide ? o64.size() : o32.size(); }
    [[nodiscard]] u8 nbytes() const { return o32.size() * sizeof(u4) + o64.size() * sizeof(u8); }
};

/**
 * One immutable sorted run: the whole of a static index, or one level of a
 * dynamic one.
 *
 * ufeat    u8        sorted unique anchor hashes                     U
 * offsets  u4 (u8)   CSR: key i's run is windows[offsets[i]..offsets[i+1])   U+1
 * windows  packed    one window id (j_max) per anchor, grouped by key        N
 *
 * In a dynamic index a key may be present with an empty postings range,
 * meaning "blocked": its occurrence count is known to exceed the repeat
 * cutoff (see dyn_index_t). A static index never contains empty ranges.
 */
struct run_t {
    parlay::sequence<u8> ufeat;
    offsets_t offsets;
    packed_array_t windows;

    [[nodiscard]] size_t n_keys() const { return ufeat.size(); }
    [[nodiscard]] u8 n_entries() const { return windows.n; }
    [[nodiscard]] bool empty() const { return ufeat.empty(); }
    [[nodiscard]] u8 nbytes() const { return ufeat.size() * sizeof(u8) + offsets.nbytes() + windows.nbytes(); }

    /** Postings range of `key`, or false if the run doesn't hold it. */
    inline bool find(u8 key, u8 &from, u8 &to) const {
        auto it = std::lower_bound(ufeat.begin(), ufeat.end(), key);
        if (it == ufeat.end() || *it != key) return false;
        size_t i = (size_t)(it - ufeat.begin());
        from = offsets[i];
        to = offsets[i + 1];
        return true;
    }

    /**
     * Build from sorted unique keys, each key's posting count, and all
     * postings concatenated in key order. Offsets are 32-bit unless the
     * entry count reaches 2^32; window ids are packed to `win_bits`.
     */
    static run_t assemble(parlay::sequence<u8> &&keys, const parlay::sequence<u4> &counts,
                          const parlay::sequence<u4> &ws, u4 win_bits) {
        run_t r;
        r.ufeat = std::move(keys);
        auto [off, total] = parlay::scan(parlay::map(counts, [](u4 c) { return (u8)c; }));
        off.push_back(total);
        r.offsets.wide = total > (u8)std::numeric_limits<u4>::max();
        if (r.offsets.wide) r.offsets.o64 = std::move(off);
        else r.offsets.o32 = parlay::map(off, [](u8 o) { return (u4)o; });
        r.windows.pack(ws, win_bits);
        return r;
    }
};

/**
 * Window-id allocator: reference records in the order they were added, the
 * first window id of each, and per-window repeat-filter weights.
 *
 * Windows are `win` wide and start every step = win/T bases (T =
 * params_t::overlap), so an anchor at p lies in exactly the T consecutive
 * windows j_max-T+1 .. j_max, j_max = floor(p/step). Only j_max is stored;
 * the tally votes for all T. Each record's window-id range is prefixed with
 * T-1 padding windows so j_max-T+1 never reaches into the previous record:
 * a padding window holds no anchors of its own, only inferred votes, so its
 * count never exceeds the real window above it, and the higher-id tie-break
 * (heavyhitter_ht_t::bump) means it never wins.
 *
 * Append-only, so the same table serves a static build and a growing
 * dynamic index.
 */
struct window_table_t {
    u4 overlap = 2;
    u4 step = 2000;
    std::vector<std::string> names;
    std::vector<u8> first;   // first window id of each record, padding included
    u8 n_windows = 0;
    // Per window id: how many windows of the every-window layout an anchor
    // with this j_max lies in -- T, except within the first T-1 real windows
    // of a record, where the lower windows don't exist. Padding windows never
    // appear as a stored j_max; they get 0. Summed over a key's postings this
    // is the repeat filter's occurrence count.
    std::vector<u1> weight;

    window_table_t() = default;
    window_table_t(u4 overlap, u4 step): overlap(overlap), step(step) {}

    /** Allocate windows for a record of length L; returns its first *real* window id. */
    u8 alloc(const std::string &name, u8 L) {
        u8 nwin = MAX((u8)1, L / step + 1);
        names.push_back(name);
        first.push_back(n_windows);
        u8 base = n_windows + (overlap - 1); // skip this record's padding windows
        n_windows = base + nwin;
        if (n_windows >= (u8)std::numeric_limits<u4>::max())
            log_error("reference has too many windows (%llu) for 32-bit window ids; increase --win",
                      (unsigned long long)n_windows);
        extend_weights(base, nwin);
        return base;
    }

    /** Window id -> (record name, window start bp). */
    [[nodiscard]] std::pair<std::string, u8> locus(u8 wid) const {
        size_t r = (size_t)(std::upper_bound(first.begin(), first.end(), wid) - first.begin());
        r = (r == 0) ? 0 : r - 1;
        u8 first_real = first[r] + (overlap - 1);
        u8 wstart = (wid > first_real ? wid - first_real : 0) * step;
        return {names[r], wstart};
    }

    /** Recompute `weight` from `first` (after loading a saved table). */
    void rebuild_weights() {
        weight.clear();
        for (size_t r = 0; r < first.size(); ++r) {
            u8 end = (r + 1 < first.size()) ? first[r + 1] : n_windows;
            u8 base = first[r] + (overlap - 1);
            extend_weights(base, end - base);
        }
    }

private:
    void extend_weights(u8 base, u8 nwin) {
        weight.resize(base, 0);
        weight.resize(base + nwin, (u1)overlap);
        for (u8 j = 0; j + 1 < overlap && j < nwin; ++j) weight[base + j] = (u1)(j + 1);
    }
};

/**
 * Every window voted for by one key: the union of [j-T+1, j] over the
 * key's stored ids `ids` (sorted ascending, may repeat), in ascending
 * order and without duplicates -- so one anchor votes at most once per
 * window (the double-voting fix).
 */
template <typename F>
inline void expand_windows(const u4 *ids, size_t n, u4 T, F &&emit) {
    u8 next = 0; // lowest window id not yet emitted
    for (size_t i = 0; i < n; ++i) {
        u8 j = ids[i], lo = MAX(next, j + 1 >= T ? j + 1 - T : 0);
        for (u8 w = lo; w <= j; ++w) emit((u4)w);
        next = MAX(next, j + 1);
    }
}

/**
 * Anchors of one reference record, in pieces: `emit(hashes, window ids)` is
 * called once per chunk with one entry per anchor (its j_max). Records
 * longer than `chunk` are processed in overlapping pieces so the transient
 * extraction arrays stay bounded; features are kept only for owners inside
 * each piece's own span, so nothing is emitted twice. Returns the number of
 * anchors.
 */
template <typename F>
static u8 record_anchors(const std::string &seq, const params_t &p, u4 step, size_t chunk, u8 base, F &&emit) {
    u8 L = seq.size();
    u8 pad = p.k + 4096;
    u8 span = MAX((u8)chunk, pad * 4);
    u8 n_anchor = 0, start = 0;
    while (start < L) {
        u8 end = MIN(L, start + span);
        u8 lo = (start > pad) ? start - pad : 0;
        u8 hi = MIN(L, end + pad);
        auto codes = encode(seq.substr(lo, hi - lo));
        auto a = syncmer_anchors(codes, p.k, p.s, p.t, p.downsample);
        if (!a.positions.empty()) {
            auto keep = parlay::tabulate(a.positions.size(), [&](size_t i) {
                u8 abspos = a.positions[i] + lo;
                return abspos >= start && abspos < end;
            });
            auto idx = parlay::pack_index(keep);
            if (!idx.empty()) {
                auto feat = parlay::map(idx, [&](size_t i) { return a.hashes[i]; });
                auto win = parlay::map(idx, [&](size_t i) { return (u4)(base + (a.positions[i] + lo) / step); });
                n_anchor += idx.size();
                emit(std::move(feat), std::move(win));
            }
        }
        start = end;
    }
    return n_anchor;
}

#endif //CUMIN_RUN_H
