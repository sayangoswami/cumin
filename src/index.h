//
// Created by Sayan Goswami on 22.09.2026.
//

#ifndef CUMIN_INDEX_H
#define CUMIN_INDEX_H

#include "run.h"
#include <fstream>
#include <filesystem>

/**
 * Anchors filed under overlapping reference windows: one run_t (see run.h
 * for the layout) over windows allocated by a window_table_t.
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

    [[nodiscard]] u8 n_keys() const { return run.n_keys(); }
    [[nodiscard]] u8 n_windows() const { return wt.n_windows; }
    [[nodiscard]] u4 window_bits() const { return run.windows.width; }
    [[nodiscard]] bool wide_offsets() const { return run.offsets.wide; }

private:
    window_table_t wt;

    std::string tmpdir;
    size_t batch, chunk;
    std::filesystem::path bucket_dir;
    std::vector<std::ofstream> bucket_ffiles, bucket_wfiles; // low-mem only, NB each

    std::vector<parlay::sequence<u8>> stage_feat;
    std::vector<parlay::sequence<u4>> stage_win;
    size_t pending = 0;

    run_t run;

    bool buckets_open = false;

    void open_buckets();
    void flush_low_mem();
    [[nodiscard]] std::pair<parlay::sequence<u8>, parlay::sequence<u4>> read_bucket(u4 i) const;

    heavyhitter_ht_t<u4> *hhs = nullptr;
    query_scratch_t *scratch = nullptr;

    [[nodiscard]] vote_result_t vote_scalar(std::vector<u8> &anchors, std::vector<u4> &dedup_buf, heavyhitter_ht_t<u4> &hh) const;
    [[nodiscard]] query_result_t query_scalar(const std::string &seq, query_scratch_t &sc, heavyhitter_ht_t<u4> &hh) const;
};

#endif //CUMIN_INDEX_H
