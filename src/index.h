//
// Created by Sayan Goswami on 22.09.2026.
//

#ifndef CUMIN_INDEX_H
#define CUMIN_INDEX_H

#include "prelude.h"
#include "params.h"
#include "cumin.h"
#include "parlay/sequence.h"
#include "parlay/parallel.h"
#include <fstream>
#include <filesystem>
#include <unordered_map>

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
 * Anchors filed under overlapping reference windows.
 *
 * ufeat    u8        sorted unique anchor hashes                     U
 * offsets  u4 (u8)   CSR: key i's run is windows[offsets[i]..offsets[i+1])   U+1
 * windows  packed    one window id per anchor, grouped by key        N
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
 * Direct port of pycumin.cumin.Index (see cumin.py) -- see that docstring
 * for the method. Two build modes, selected by params_t::low_mem:
 *  - low-mem (default): anchors are radix-partitioned into NB on-disk
 *    buckets by the top byte of the feature hash as they are produced, so
 *    the full raw anchor array never has to fit in RAM. Each bucket
 *    (~n_entries/NB) is then sorted once, in turn.
 *  - high-mem: every anchor is kept in memory and sorted in a single pass.
 */
class index_t {
public:
    static constexpr u4 NB = 256;

    static constexpr u4 MAGIC = 0x4E4D5543; // "CUMN"
    static constexpr u2 FORMAT_VERSION = 2;

    params_t p;
    u4 win_step;

    explicit index_t(params_t params = {}, std::string tmpdir = "/tmp",
                      size_t batch = 20'000'000, size_t chunk = 32'000'000);
    ~index_t();

    index_t(const index_t&) = delete;
    index_t& operator=(const index_t&) = delete;
    index_t(index_t&&) = delete;
    index_t& operator=(index_t&&) = delete;

    /** Index one reference record. Returns the number of anchors filed. */
    u8 add_record(const std::string &name, const std::string &seq);

    /** Consolidate everything filed by add_record() into the queryable CSR form. */
    void build(int sample_buckets = 16);

    void save(std::ostream &f) const;
    void load(std::istream &f);

    [[nodiscard]] u8 nbytes() const;

    /** Window id -> (record name, window start bp). */
    [[nodiscard]] std::pair<std::string, u8> window_locus(u8 wid) const;

    struct vote_result_t {
        float score = 0.0f;
        i8 window = -1;
        u4 n_unique = 0;
    };

    /** One orientation. score = best_window_votes / n_unique_anchors. */
    [[nodiscard]] vote_result_t vote(const parlay::sequence<u8> &anchors) const;

    struct query_result_t {
        float score = 0.0f;
        i8 window = -1;
        char strand = '+';
        u4 n_anchors = 0;
    };

    /** Both orientations, better kept. */
    [[nodiscard]] query_result_t query(const std::string &seq) const;

    /**
     * Allocates the per-worker-thread scratch (heavy-hitter counters +
     * anchor-extraction buffers) that query_batch() reuses across reads.
     * Idempotent; called automatically by query_batch() if not done yet.
     */
    void init_query_buffers();

    /**
     * Batch query entry point: parallel *across* reads (one task per read,
     * parlay::parallel_for), each read's own work fully scalar and
     * allocation-free in steady state (see query_scalar/vote_scalar in
     * index.cpp) -- unlike query()/vote() above, which parallelize inside
     * a single read and are better suited to a one-off standalone query
     * than to mapping many reads. See the design note in index.cpp.
     */
    [[nodiscard]] std::vector<query_result_t> query_batch(const std::vector<std::string> &seqs);

    u8 n_entries = 0, n_dropped = 0;
    u4 occ_cutoff = 0;

    [[nodiscard]] u8 n_keys() const { return ufeat.size(); }
    [[nodiscard]] u8 n_windows() const { return base_win; }
    [[nodiscard]] u4 window_bits() const { return windows.width; }
    [[nodiscard]] bool wide_offsets() const { return offsets.wide; }

private:
    std::vector<std::string> rec_names;
    std::vector<u8> rec_first;
    u8 base_win = 0;

    std::string tmpdir;
    size_t batch, chunk;
    std::filesystem::path bucket_dir;
    std::vector<std::ofstream> bucket_ffiles, bucket_wfiles; // low-mem only, NB each

    std::vector<parlay::sequence<u8>> stage_feat;
    std::vector<parlay::sequence<u4>> stage_win;
    size_t pending = 0;

    // built CSR arrays
    parlay::sequence<u8> ufeat;
    offsets_t offsets;
    packed_array_t windows;

    bool buckets_open = false;

    void file_anchors(const parlay::sequence<u8> &pos, const parlay::sequence<u8> &feat, u8 base);
    [[nodiscard]] parlay::sequence<u1> window_weights() const;
    void open_buckets();
    void flush_low_mem();
    [[nodiscard]] std::pair<parlay::sequence<u8>, parlay::sequence<u4>> read_bucket(u4 i) const;

    heavyhitter_ht_t<u4> *hhs = nullptr;
    query_scratch_t *scratch = nullptr;

    [[nodiscard]] vote_result_t vote_scalar(std::vector<u8> &anchors, std::vector<u4> &dedup_buf, heavyhitter_ht_t<u4> &hh) const;
    [[nodiscard]] query_result_t query_scalar(const std::string &seq, query_scratch_t &sc, heavyhitter_ht_t<u4> &hh) const;
};

#endif //CUMIN_INDEX_H
