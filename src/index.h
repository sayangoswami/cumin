//
// Created by Sayan Goswami on 22.09.2026.
//

#ifndef CUMIN_INDEX_H
#define CUMIN_INDEX_H

#include "prelude.h"
#include "params.h"
#include "cumin.h"
#include "parlay/sequence.h"
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

    // NB: this checks the count *after* incorporating this insertion (unlike
    // collinearity's version, which compares the pre-increment count -- that
    // makes top_count always one less than the true max, and a key that only
    // ever occurs once never registers as the leader at all. Harmless there
    // since only the *relative* ranking of candidates is used; here
    // top_count is the vote() score's numerator directly, so it has to be
    // the real count.
    void insert(const T key) {
        if (!overflowed) {
            for (u4 i = 0; i < n; ++i) {
                if (keys[i] == key) {
                    u4 count = ++counts[i];
                    if (count > top_count) top_count = count, top_key = key;
                    return;
                }
            }
            if (n < CAP) {
                keys[n] = key;
                counts[n] = 1;
                ++n;
                if (1 > top_count) top_count = 1, top_key = key;
                return;
            }
            overflowed = true;
            overflow_map.reserve(CAP * 2);
            for (u4 i = 0; i < n; ++i) overflow_map[keys[i]] = counts[i];
            n = 0;
        }
        u4 count = ++overflow_map[key];
        if (count > top_count) top_count = count, top_key = key;
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
 * Anchors filed under overlapping reference windows.
 *
 * ufeat   u8   sorted unique anchor hashes
 * starts  u8   offset of each key's run in wsorted
 * cnt     u4   run length
 * wsorted u4   window ids, grouped by key
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

    params_t p;
    u4 win_step;

    explicit index_t(params_t params = {}, std::string tmpdir = "/tmp",
                      size_t batch = 20'000'000, size_t chunk = 32'000'000);
    ~index_t();

    // Always constructed in place / held by reference -- never moved or
    // copied anywhere in this codebase. Left non-movable rather than
    // implementing a correct move for the owning hhs/scratch raw pointers
    // (see init_query_buffers()) that nothing currently needs.
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

private:
    std::vector<std::string> rec_names;
    std::vector<u8> rec_first;
    u8 base_win = 0;

    std::string tmpdir;
    size_t batch, chunk;
    std::filesystem::path bucket_dir;
    std::vector<std::ofstream> bucket_ffiles, bucket_wfiles; // low-mem only, NB each

    // staging: chunks of (feature,window) pairs waiting to be filed. In
    // low-mem mode these are periodically radix-partitioned to the NB disk
    // buckets (flush_low_mem()); in high-mem mode they just keep growing
    // until build() concatenates and sorts everything in one shot.
    std::vector<parlay::sequence<u8>> stage_feat;
    std::vector<parlay::sequence<u4>> stage_win;
    size_t pending = 0;

    // built CSR arrays
    parlay::sequence<u8> ufeat;
    parlay::sequence<u8> starts;
    parlay::sequence<u4> cnt;
    parlay::sequence<u4> wsorted;

    bool buckets_open = false;

    void file_anchors(const parlay::sequence<u8> &pos, const parlay::sequence<u8> &feat, u8 base, u8 nwin);
    void open_buckets();
    void flush_low_mem();
    [[nodiscard]] std::pair<parlay::sequence<u8>, parlay::sequence<u4>> read_bucket(u4 i) const;

    // query_batch()'s scalar path: one heavy-hitter counter + one anchor
    // scratch buffer per worker thread, allocated once by init_query_buffers()
    // and reused across every read that thread processes.
    heavyhitter_ht_t<u4> *hhs = nullptr;
    query_scratch_t *scratch = nullptr;

    [[nodiscard]] vote_result_t vote_scalar(std::vector<u8> &anchors, heavyhitter_ht_t<u4> &hh) const;
    [[nodiscard]] query_result_t query_scalar(const std::string &seq, query_scratch_t &sc, heavyhitter_ht_t<u4> &hh) const;
};

#endif //CUMIN_INDEX_H
