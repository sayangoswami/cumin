//
// Created by Sayan Goswami on 01.10.2026.
//

#ifndef CUMIN_DYN_INDEX_H
#define CUMIN_DYN_INDEX_H

#include "index.h"
#include "tiered_buffer.h"
#include <memory>

/**
 * Where newly inserted postings live until they are sealed into a run.
 * Always queryable: lookup() sees every entry inserted so far.
 */
class buffer_t {
public:
    virtual ~buffer_t() = default;
    /** Insert a batch of entries, sorted by the caller. */
    virtual void insert(const parlay::sequence<entry_t> &sorted) = 0;
    [[nodiscard]] virtual size_t size() const = 0;
    /** Append the window ids filed under `key` to `out`; returns how many. */
    virtual size_t lookup(u8 key, std::vector<u4> &out) const = 0;
    /** Every entry, sorted; leaves the buffer empty. */
    virtual parlay::sequence<entry_t> drain() = 0;
    [[nodiscard]] virtual u8 nbytes() const = 0;
};

enum class dyn_variant_t { lsm, tiered, lsm_tiered };

dyn_variant_t parse_dyn_variant(const std::string &s);
const char *dyn_variant_name(dyn_variant_t v);

struct dyn_params_t {
    dyn_variant_t variant = dyn_variant_t::lsm;
    size_t buffer = 1 << 20;      // entries buffered before sealing a run (lsm, lsm-tiered)
    u4 ratio = 2;                 // runs per level before they merge into the next level
    u4 max_occ = std::numeric_limits<u4>::max(); // repeat cutoff, weighted (same units as index_t::occ_cutoff)
    float threshold = 0.15f;      // score above which a sequence counts as mapped
    u4 check_bp = 360;            // prefix length queried to decide whether a sequence maps
    bool skip_mapped = true;      // add_record() skips sequences whose prefix already maps
    size_t chunk = 32'000'000;    // max bp of one record processed at a time
};

/**
 * An index that grows while it is queried: the logarithmic method (an LSM
 * tree), a tiered vector, or both. See docs/Status-Sep29-2026.md, "The
 * dynamic index".
 *
 * New postings go into a buffer (buffer_t). For `lsm` and `lsm-tiered`, a
 * full buffer is sealed into an immutable run_t -- the static index layout --
 * and runs merge like a counter in base `ratio`: when a level holds `ratio`
 * runs they merge into one run on the next level. `tiered` never seals: the
 * tiered vector holds everything.
 *
 * Repeat filter: a key is dropped at query time when its weighted
 * occurrence count, summed over the buffer and every run, exceeds max_occ --
 * exactly the static index's filter given the same cutoff. Counts only
 * grow, so a run in which a key already exceeds the cutoff on its own keeps
 * the key with an empty postings range ("blocked") and drops its postings;
 * queries treat a blocked key as filtered.
 *
 * Single writer: add_record() must not run concurrently with queries.
 * Queries may run in parallel with each other (query_batch / decide).
 */
class dyn_index_t {
public:
    using query_result_t = index_t::query_result_t;

    params_t p;
    dyn_params_t dp;

    dyn_index_t(params_t params, dyn_params_t dparams);
    ~dyn_index_t();
    dyn_index_t(const dyn_index_t&) = delete;
    dyn_index_t& operator=(const dyn_index_t&) = delete;

    /**
     * Add one sequence. With skip_mapped, its first check_bp bases are
     * queried first and the whole sequence is skipped if they map (score >
     * threshold). Returns whether it was added.
     */
    bool add_record(const std::string &name, const std::string &seq, bool skip_mapped);
    bool add_record(const std::string &name, const std::string &seq) { return add_record(name, seq, dp.skip_mapped); }

    [[nodiscard]] query_result_t query(const std::string &seq);
    [[nodiscard]] std::vector<query_result_t> query_batch(const std::vector<std::string> &seqs);
    /** query_batch() over each sequence's first check_bp bases: the real-time accept/reject step. */
    [[nodiscard]] std::vector<query_result_t> decide(const std::vector<std::string> &seqs);
    [[nodiscard]] bool is_mapped(const query_result_t &r) const { return r.score > dp.threshold; }

    /** Window id -> (record name, window start bp). */
    [[nodiscard]] std::pair<std::string, u8> window_locus(u8 wid) const { return wt.locus(wid); }

    // ---- stats ----
    [[nodiscard]] u8 n_records() const { return wt.names.size(); }
    [[nodiscard]] u8 n_windows() const { return wt.n_windows; }
    [[nodiscard]] u8 n_entries() const;
    [[nodiscard]] size_t buffer_size() const { return buf->size(); }
    [[nodiscard]] size_t n_runs() const;
    /** Entries per live run, largest level first. */
    [[nodiscard]] std::vector<u8> run_sizes() const;
    [[nodiscard]] u8 nbytes() const;
    /** Keys blocked by the repeat filter, summed over live runs. */
    [[nodiscard]] u8 n_blocked() const;
    u8 n_flushes = 0, n_merges = 0;
    u8 entries_written = 0;      // postings written into runs, flushes and merges included
    double last_flush_s = 0, worst_flush_s = 0; // one flush including its cascade of merges

private:
    window_table_t wt;
    std::unique_ptr<buffer_t> buf;
    std::vector<std::vector<run_t>> levels;

    struct scratch_t {
        query_scratch_t q;
        std::vector<u4> ids;
        std::vector<std::pair<const run_t*, std::pair<u8, u8>>> hits;
        heavyhitter_ht_t<u4> hh;
    };
    std::unique_ptr<scratch_t[]> scratch;
    void init_scratch();

    [[nodiscard]] bool has_runs() const { return dp.variant != dyn_variant_t::tiered; }
    [[nodiscard]] u4 win_bits() const { return packed_array_t::bits_for(MAX(wt.n_windows, (u8)1) - 1); }
    [[nodiscard]] u8 weighted(const u4 *ids, size_t n) const;

    void flush();
    run_t seal(const parlay::sequence<entry_t> &sorted);
    run_t merge(const run_t &a, const run_t &b);

    index_t::vote_result_t vote_scalar(std::vector<u8> &anchors, scratch_t &sc) const;
    query_result_t query_scalar(const std::string &seq, scratch_t &sc) const;
};

#endif //CUMIN_DYN_INDEX_H
