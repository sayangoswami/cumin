//
// Created by Sayan Goswami on 22.09.2026.
//

#ifndef CUMIN_CUMIN_H
#define CUMIN_CUMIN_H

#include "prelude.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include <limits>
#include <string>
#include <array>
#include <vector>
#include <algorithm>

#define BASE_BAD ((u1)255)
static const u8 M_ANCHOR = 0x9E3779B97F4A7C15ULL;
static const u8 M_DOWN   = 0xC2B2AE3D27D4EB4FULL;

static const auto BASE_TABLE = [] {
    std::array<u1, 256> t{};
    t.fill(BASE_BAD);
    t[(u1)'A'] = t[(u1)'a'] = 0;
    t[(u1)'C'] = t[(u1)'c'] = 1;
    t[(u1)'G'] = t[(u1)'g'] = 2;
    t[(u1)'T'] = t[(u1)'t'] = 3;
    return t;
}();

/** Encode a DNA sequence into 2-bit base codes (0-3); anything else (N, ...) becomes BASE_BAD. */
static inline parlay::sequence<u1> encode(const std::string &seq) {
    return parlay::tabulate(seq.size(), [&](size_t i) {
        return BASE_TABLE[(u1)seq[i]];
    });
}

/** Reverse-complement a code sequence; BASE_BAD codes pass through unchanged. */
static inline parlay::sequence<u1> revcomp_codes(const parlay::sequence<u1> &codes) {
    const size_t n = codes.size();
    return parlay::tabulate(n, [&](size_t i) {
        u1 c = codes[n - 1 - i];
        return (c < 4) ? (u1)(3 - c) : c;
    });
}

/** splitmix64 finalizer -- avalanches all bits of x. */
static inline u8 mix64(u8 x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

/**
 * Base-4 value and "window touches an invalid base" flag for every length-m
 * window of `codes`. Mirrors cumin.py's _roll: an invalid base contributes 0
 * to the value (never consulted once `bad` is set), and `bad` is sticky
 * across the whole window.
 */
struct rolled_t {
    parlay::sequence<u8> values;
    parlay::sequence<bool> bad;
};

static inline rolled_t roll(const parlay::sequence<u1> &codes, u4 m) {
    const size_t n = codes.size();
    if (n < m) return {{}, {}};
    const size_t L = n - m + 1;
    auto values = parlay::tabulate(L, [&](size_t i) {
        u8 v = 0;
        for (u4 j = 0; j < m; ++j) {
            u1 c = codes[i + j];
            v = v * 4 + (c < 4 ? c : 0);
        }
        return v;
    });
    auto bad = parlay::tabulate(L, [&](size_t i) {
        for (u4 j = 0; j < m; ++j)
            if (codes[i + j] >= 4) return true;
        return false;
    });
    return {std::move(values), std::move(bad)};
}

struct anchors_t {
    parlay::sequence<u8> positions;
    parlay::sequence<u8> hashes;
};

/**
 * Open syncmer anchors.
 *
 * A k-mer starting at position i is kept iff:
 *  - it and every one of its w = k-s+1 constituent s-mers avoid invalid bases;
 *  - the minimum-hash s-mer among those w s-mers sits at offset t; and
 *  - (if downsample > 1) mix64(anchor_hash ^ M_DOWN) % downsample == 0
 *    (a FracMinHash-style filter, decorrelated from the syncmer test above).
 *
 * @param codes    2-bit-encoded sequence (see encode()), BASE_BAD for invalid bases
 * @param k        k-mer length
 * @param s        s-mer length (s <= k)
 * @param t        required offset of the minimum-hash s-mer within the k-mer
 * @param downsample keep roughly 1/downsample of syncmers; 1 (or 0) disables this
 */
static inline anchors_t syncmer_anchors(const parlay::sequence<u1> &codes, u4 k, u4 s, u4 t, u4 downsample) {
    expect(s <= k && t < k - s + 1);
    const size_t n = codes.size();
    if (n < k) return {{}, {}};
    const u4 w = k - s + 1;
    const size_t nk = n - k + 1;

    auto sroll = roll(codes, s);
    auto sh = parlay::tabulate(sroll.values.size(), [&](size_t i) {
        return sroll.bad[i] ? std::numeric_limits<u8>::max() : mix64(sroll.values[i]);
    });

    auto kroll = roll(codes, k);

    auto kh = parlay::tabulate(nk, [&](size_t i) {
        return mix64(kroll.values[i] * M_ANCHOR);
    });

    auto sel = parlay::tabulate(nk, [&](size_t i) -> bool {
        if (kroll.bad[i]) return false;
        u8 min_hash = sh[i];
        u4 min_off = 0;
        for (u4 j = 1; j < w; ++j) {
            if (sh[i + j] < min_hash) min_hash = sh[i + j], min_off = j;
        }
        if (min_off != t) return false;
        if (downsample > 1 && (mix64(kh[i] ^ M_DOWN) % downsample) != 0) return false;
        return true;
    });

    auto idx = parlay::pack_index(sel);
    auto positions = parlay::map(idx, [](size_t i) { return (u8)i; });
    auto hashes = parlay::map(idx, [&](size_t i) { return kh[i]; });
    return {std::move(positions), std::move(hashes)};
}

struct read_anchors_t {
    parlay::sequence<u8> fwd, rev;
};

/** Anchor hashes for a read, forward and reverse-complement strands. */
static inline read_anchors_t read_anchors(const std::string &seq, u4 k, u4 s, u4 t, u4 downsample) {
    auto codes = encode(seq);
    auto rc = revcomp_codes(codes);
    auto f = syncmer_anchors(codes, k, s, t, downsample);
    auto r = syncmer_anchors(rc, k, s, t, downsample);
    return {std::move(f.hashes), std::move(r.hashes)};
}

// ─────────────────────── scalar (batch-query) path ──────────────────────

static inline void encode_scalar(const std::string &seq, std::vector<u1> &codes) {
    codes.resize(seq.size());
    for (size_t i = 0; i < seq.size(); ++i) codes[i] = BASE_TABLE[(u1)seq[i]];
}

static inline void revcomp_codes_scalar(const std::vector<u1> &codes, std::vector<u1> &out) {
    const size_t n = codes.size();
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        u1 c = codes[n - 1 - i];
        out[i] = (c < 4) ? (u1)(3 - c) : c;
    }
}

/** Incremental rolling m-mer values/bad-flags over `codes` (see roll()). */
static inline void roll_scalar(const std::vector<u1> &codes, u4 m, std::vector<u8> &vals, std::vector<u1> &bad) {
    const size_t n = codes.size();
    if (n < m) { vals.clear(); bad.clear(); return; }
    const size_t L = n - m + 1;
    vals.resize(L);
    bad.resize(L);

    u8 M = 1;
    for (u4 j = 1; j < m; ++j) M *= 4;

    u8 v = 0;
    u4 bad_count = 0;
    for (u4 j = 0; j < m; ++j) {
        u1 c = codes[j];
        v = v * 4 + (c < 4 ? c : 0);
        if (c >= 4) ++bad_count;
    }
    vals[0] = v;
    bad[0] = bad_count > 0;

    for (size_t i = 1; i < L; ++i) {
        u1 out_c = codes[i - 1], in_c = codes[i + m - 1];
        if (out_c >= 4) --bad_count;
        v = (v - (u8)(out_c < 4 ? out_c : 0) * M) * 4 + (in_c < 4 ? in_c : 0);
        if (in_c >= 4) ++bad_count;
        vals[i] = v;
        bad[i] = bad_count > 0;
    }
}

/** Scratch buffers for one thread's worth of scalar anchor extraction, reused across reads. */
struct query_scratch_t {
    std::vector<u1> codes, rc, s_bad, k_bad;
    std::vector<u8> s_vals, k_vals, s_hash;
    std::vector<u8> fwd_hashes, rev_hashes;
    std::vector<u4> dedup_buf; // index_t::vote_scalar's per-key window-id dedup scratch
};

/** Same open-syncmer selection as syncmer_anchors(), scalar, appending hashes to `out`. */
static inline void syncmer_anchors_scalar(const std::vector<u1> &codes, u4 k, u4 s, u4 t, u4 downsample,
                                           query_scratch_t &sc, std::vector<u8> &out) {
    out.clear();
    expect(s <= k && t < k - s + 1);
    const size_t n = codes.size();
    if (n < k) return;
    const u4 w = k - s + 1;
    const size_t nk = n - k + 1;

    roll_scalar(codes, s, sc.s_vals, sc.s_bad);
    roll_scalar(codes, k, sc.k_vals, sc.k_bad);

    sc.s_hash.resize(sc.s_vals.size());
    for (size_t i = 0; i < sc.s_vals.size(); ++i)
        sc.s_hash[i] = sc.s_bad[i] ? std::numeric_limits<u8>::max() : mix64(sc.s_vals[i]);

    for (size_t i = 0; i < nk; ++i) {
        if (sc.k_bad[i]) continue;
        u8 min_hash = sc.s_hash[i];
        u4 min_off = 0;
        for (u4 j = 1; j < w; ++j) {
            if (sc.s_hash[i + j] < min_hash) { min_hash = sc.s_hash[i + j]; min_off = j; }
        }
        if (min_off != t) continue;
        u8 kh = mix64(sc.k_vals[i] * M_ANCHOR);
        if (downsample > 1 && (mix64(kh ^ M_DOWN) % downsample) != 0) continue;
        out.push_back(kh);
    }
}

/** Anchor hashes for a read, forward and reverse-complement strands (scalar). */
static inline void read_anchors_scalar(const std::string &seq, u4 k, u4 s, u4 t, u4 downsample, query_scratch_t &sc) {
    encode_scalar(seq, sc.codes);
    revcomp_codes_scalar(sc.codes, sc.rc);
    syncmer_anchors_scalar(sc.codes, k, s, t, downsample, sc, sc.fwd_hashes);
    syncmer_anchors_scalar(sc.rc, k, s, t, downsample, sc, sc.rev_hashes);
}

#endif //CUMIN_CUMIN_H
