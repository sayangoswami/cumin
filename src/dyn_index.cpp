//
// Created by Sayan Goswami on 01.10.2026.
//

#include "dyn_index.h"
#include "parlay/primitives.h"
#include <chrono>

// ----------------------------- buffers ---------------------------------

/**
 * Sorted vector buffer for the `lsm` variant. A sorted vector cannot take
 * a cheap insert, so it is two sorted segments: new batches merge into a
 * small `tail`, and the tail merges into `main` once it reaches 1/8 of it.
 * Each insert then moves O(tail) entries rather than O(buffer).
 */
class sorted_buffer_t final : public buffer_t {
    std::vector<entry_t> main, tail;

    static void lookup_in(const std::vector<entry_t> &v, u8 key, std::vector<u4> &out, size_t &n) {
        auto it = std::lower_bound(v.begin(), v.end(), entry_t{key, 0});
        for (; it != v.end() && it->key == key; ++it, ++n) out.push_back(it->win);
    }

public:
    void insert(const parlay::sequence<entry_t> &sorted) override {
        size_t mid = tail.size();
        tail.insert(tail.end(), sorted.begin(), sorted.end());
        std::inplace_merge(tail.begin(), tail.begin() + (std::ptrdiff_t)mid, tail.end());
        if (tail.size() > MAX((size_t)4096, main.size() / 8)) {
            mid = main.size();
            main.insert(main.end(), tail.begin(), tail.end());
            std::inplace_merge(main.begin(), main.begin() + (std::ptrdiff_t)mid, main.end());
            tail.clear();
        }
    }

    [[nodiscard]] size_t size() const override { return main.size() + tail.size(); }

    size_t lookup(u8 key, std::vector<u4> &out) const override {
        size_t n = 0;
        lookup_in(main, key, out, n);
        lookup_in(tail, key, out, n);
        return n;
    }

    parlay::sequence<entry_t> drain() override {
        parlay::sequence<entry_t> out(main.size() + tail.size());
        std::merge(main.begin(), main.end(), tail.begin(), tail.end(), out.begin());
        main.clear();
        tail.clear();
        return out;
    }

    [[nodiscard]] u8 nbytes() const override { return (main.capacity() + tail.capacity()) * sizeof(entry_t); }
};

/** Tiered-vector buffer for the `tiered` and `lsm-tiered` variants: one sorted insert per entry. */
class tiered_buffer_t final : public buffer_t {
    tiered_vec_t tv;

public:
    void insert(const parlay::sequence<entry_t> &sorted) override {
        for (auto &e: sorted) tv.insert_sorted(e);
    }

    [[nodiscard]] size_t size() const override { return tv.size(); }

    size_t lookup(u8 key, std::vector<u4> &out) const override {
        size_t n = 0;
        for (size_t i = tv.lower_bound(key); i < tv.size() && tv[i].key == key; ++i, ++n) out.push_back(tv[i].win);
        return n;
    }

    parlay::sequence<entry_t> drain() override {
        auto out = parlay::tabulate(tv.size(), [&](size_t i) { return tv[i]; });
        tv.clear();
        return out;
    }

    [[nodiscard]] u8 nbytes() const override { return tv.size() * sizeof(entry_t); }
};

dyn_variant_t parse_dyn_variant(const std::string &s) {
    if (s == "lsm") return dyn_variant_t::lsm;
    if (s == "tiered") return dyn_variant_t::tiered;
    if (s == "lsm-tiered") return dyn_variant_t::lsm_tiered;
    log_error("unknown dynamic index variant '%s' (expected lsm, tiered or lsm-tiered)", s.c_str());
}

const char *dyn_variant_name(dyn_variant_t v) {
    switch (v) {
        case dyn_variant_t::lsm: return "lsm";
        case dyn_variant_t::tiered: return "tiered";
        case dyn_variant_t::lsm_tiered: return "lsm-tiered";
    }
    return "?";
}

// ------------------------------ index ----------------------------------

dyn_index_t::dyn_index_t(params_t params, dyn_params_t dparams):
    p(params), dp(dparams), wt(params.overlap, params.step()) {
    if (dp.ratio < 2) log_error("ratio must be >= 2");
    if (dp.buffer < 1) log_error("buffer must be >= 1");
    if (dp.variant == dyn_variant_t::lsm) buf = std::make_unique<sorted_buffer_t>();
    else buf = std::make_unique<tiered_buffer_t>();
}

dyn_index_t::~dyn_index_t() = default;

void dyn_index_t::init_scratch() {
    if (!scratch) scratch.reset(new scratch_t[parlay::num_workers()]);
}

u8 dyn_index_t::weighted(const u4 *ids, size_t n) const {
    u8 c = 0;
    for (size_t i = 0; i < n; ++i) c += wt.weight[ids[i]];
    return c;
}

bool dyn_index_t::add_record(const std::string &name, const std::string &seq, bool skip_mapped) {
    if (skip_mapped) {
        init_scratch();
        auto prefix = seq.substr(0, dp.check_bp);
        if (is_mapped(query_scalar(prefix, scratch[(size_t)parlay::worker_id()]))) return false;
    }

    u8 base = wt.alloc(name, seq.size());
    std::vector<parlay::sequence<entry_t>> parts;
    record_anchors(seq, p, p.step(), dp.chunk, base, [&](parlay::sequence<u8> &&feat, parlay::sequence<u4> &&win) {
        parts.push_back(parlay::tabulate(feat.size(), [&](size_t i) { return entry_t{feat[i], win[i]}; }));
    });
    if (!parts.empty()) buf->insert(parlay::sort(parlay::flatten(std::move(parts))));
    if (has_runs() && buf->size() >= dp.buffer) flush();
    return true;
}

// Seal the buffer into a run on level 0, then carry: a level holding
// `ratio` runs merges them into one run on the next level.
void dyn_index_t::flush() {
    auto t0 = std::chrono::steady_clock::now();
    auto entries = buf->drain();
    if (levels.empty()) levels.emplace_back();
    levels[0].push_back(seal(entries));
    ++n_flushes;
    entries_written += entries.size();

    for (size_t i = 0; i < levels.size() && levels[i].size() >= dp.ratio; ++i) {
        run_t merged = std::move(levels[i][0]);
        for (size_t j = 1; j < levels[i].size(); ++j) {
            merged = merge(merged, levels[i][j]);
            ++n_merges;
            entries_written += merged.n_entries();
        }
        levels[i].clear();
        if (i + 1 == levels.size()) levels.emplace_back();
        levels[i + 1].push_back(std::move(merged));
    }

    last_flush_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    worst_flush_s = MAX(worst_flush_s, last_flush_s);
}

// Sorted entries -> run. A key whose weighted count already exceeds the
// cutoff within these entries is kept blocked: empty postings.
run_t dyn_index_t::seal(const parlay::sequence<entry_t> &e) {
    size_t n = e.size();
    if (n == 0) return {};
    auto starts = parlay::pack_index(parlay::tabulate(n, [&](size_t i) {
        return i == 0 || e[i].key != e[i - 1].key;
    }));
    size_t m = starts.size();
    const u8 T = p.overlap;
    auto counts = parlay::tabulate(m, [&](size_t i) {
        size_t s = starts[i], z = (i + 1 < m) ? starts[i + 1] : n;
        u8 c = z - s;
        if (T * c > dp.max_occ) {
            u8 w = 0;
            for (size_t j = s; j < z; ++j) w += wt.weight[e[j].win];
            if (w > dp.max_occ) c = 0;
        }
        return (u4)c;
    });
    auto scanned = parlay::scan(parlay::map(counts, [](u4 c) { return (u8)c; }));
    const auto &offs = scanned.first;
    parlay::sequence<u4> ws(scanned.second);
    parlay::parallel_for(0, m, [&](size_t i) {
        for (u4 j = 0; j < counts[i]; ++j) ws[offs[i] + j] = e[starts[i] + j].win;
    });
    auto keys = parlay::map(starts, [&](size_t s) { return e[s].key; });
    return run_t::assemble(std::move(keys), counts, ws, win_bits());
}

// Linear merge of two runs on their keys, in parallel over key ranges. A
// key in both runs gets both postings lists, unless either run has it
// blocked or the combined weighted count exceeds the cutoff -- then it is
// blocked in the result. A key in only one run is copied as is.
run_t dyn_index_t::merge(const run_t &a, const run_t &b) {
    const run_t &big = a.n_keys() >= b.n_keys() ? a : b;
    size_t nparts = (a.n_keys() + b.n_keys() > (1 << 16)) ? 4 * parlay::num_workers() : 1;
    nparts = MAX((size_t)1, MIN(nparts, big.n_keys()));

    // Part p covers keys in [split[p], split[p+1]), taken from the bigger run.
    auto split_at = [&](const run_t &r, size_t p_) -> size_t {
        if (p_ == 0) return 0;
        if (p_ == nparts) return r.n_keys();
        u8 key = big.ufeat[p_ * big.n_keys() / nparts];
        return (size_t)(std::lower_bound(r.ufeat.begin(), r.ufeat.end(), key) - r.ufeat.begin());
    };

    struct part_t {
        std::vector<u8> keys;
        std::vector<u4> counts, ws;
    };
    std::vector<part_t> parts(nparts);
    const u8 T = p.overlap;

    parlay::parallel_for(0, nparts, [&](size_t pi) {
        auto &out = parts[pi];
        size_t i = split_at(a, pi), ie = split_at(a, pi + 1);
        size_t j = split_at(b, pi), je = split_at(b, pi + 1);
        auto append = [&](const run_t &r, size_t k) {
            for (u8 x = r.offsets[k], z = r.offsets[k + 1]; x < z; ++x) out.ws.push_back(r.windows.get(x));
        };
        auto blocked = [](const run_t &r, size_t k) { return r.offsets[k] == r.offsets[k + 1]; };
        auto copy_key = [&](const run_t &r, size_t k) {
            size_t before = out.ws.size();
            append(r, k);
            out.keys.push_back(r.ufeat[k]);
            out.counts.push_back((u4)(out.ws.size() - before));
        };
        while (i < ie || j < je) {
            if (j == je || (i < ie && a.ufeat[i] < b.ufeat[j])) { copy_key(a, i++); continue; }
            if (i == ie || b.ufeat[j] < a.ufeat[i]) { copy_key(b, j++); continue; }
            // same key in both
            size_t before = out.ws.size();
            if (!blocked(a, i) && !blocked(b, j)) {
                append(a, i);
                append(b, j);
                u8 c = out.ws.size() - before;
                if (T * c > dp.max_occ && weighted(out.ws.data() + before, c) > dp.max_occ) out.ws.resize(before);
            }
            out.keys.push_back(a.ufeat[i]);
            out.counts.push_back((u4)(out.ws.size() - before));
            ++i, ++j;
        }
    });

    auto keys = parlay::flatten(parlay::map(parts, [](part_t &x) { return parlay::to_sequence(x.keys); }));
    auto counts = parlay::flatten(parlay::map(parts, [](part_t &x) { return parlay::to_sequence(x.counts); }));
    auto ws = parlay::flatten(parlay::map(parts, [](part_t &x) { return parlay::to_sequence(x.ws); }));
    return run_t::assemble(std::move(keys), counts, ws, win_bits());
}

// ------------------------------ query ----------------------------------

// Same tally as index_t::vote_scalar, over the buffer and every live run.
index_t::vote_result_t dyn_index_t::vote_scalar(std::vector<u8> &anchors, scratch_t &sc) const {
    if (anchors.empty()) return {0.0f, -1, 0};
    std::sort(anchors.begin(), anchors.end());
    auto &hh = sc.hh;
    auto &ids = sc.ids;
    const u8 T = p.overlap;

    hh.reset();
    u4 nq = 0;
    for (size_t i = 0; i < anchors.size(); ++i) {
        if (i > 0 && anchors[i] == anchors[i - 1]) continue;
        ++nq;
        const u8 key = anchors[i];

        // Occurrence count first, from the runs' offsets, so a filtered key
        // is dropped before any of its postings are gathered.
        sc.hits.clear();
        bool blocked = false;
        u8 occ = 0;
        for (auto &level: levels) {
            for (auto &r: level) {
                u8 from, to;
                if (!r.find(key, from, to)) continue;
                if (from == to) { blocked = true; break; }
                occ += to - from;
                sc.hits.push_back({&r, {from, to}});
            }
            if (blocked) break;
        }
        if (blocked) continue;
        ids.clear();
        occ += buf->lookup(key, ids);
        if (occ == 0 || occ > dp.max_occ) continue; // weights are >= 1, so weighted >= occ

        for (auto &[r, range]: sc.hits)
            for (u8 j = range.first; j < range.second; ++j) ids.push_back(r->windows.get(j));
        if (T * occ > dp.max_occ && weighted(ids.data(), ids.size()) > dp.max_occ) continue;

        std::sort(ids.begin(), ids.end());
        expand_windows(ids.data(), ids.size(), p.overlap, [&](u4 w) { hh.insert(w); });
    }
    if (hh.top_key == (u4)-1) return {0.0f, -1, nq};
    return {(float)hh.top_count / (float)nq, (i8)hh.top_key, nq};
}

dyn_index_t::query_result_t dyn_index_t::query_scalar(const std::string &seq, scratch_t &sc) const {
    read_anchors_scalar(seq, p.k, p.s, p.t, p.downsample, sc.q);
    auto vf = vote_scalar(sc.q.fwd_hashes, sc);
    auto vr = vote_scalar(sc.q.rev_hashes, sc);
    if (vf.score >= vr.score) return {vf.score, vf.window, '+', vf.n_unique};
    return {vr.score, vr.window, '-', vr.n_unique};
}

dyn_index_t::query_result_t dyn_index_t::query(const std::string &seq) {
    init_scratch();
    return query_scalar(seq, scratch[(size_t)parlay::worker_id()]);
}

std::vector<dyn_index_t::query_result_t> dyn_index_t::query_batch(const std::vector<std::string> &seqs) {
    init_scratch();
    std::vector<query_result_t> out(seqs.size());
    parlay::parallel_for(0, seqs.size(), [&](size_t i) {
        out[i] = query_scalar(seqs[i], scratch[(size_t)parlay::worker_id()]);
    });
    return out;
}

std::vector<dyn_index_t::query_result_t> dyn_index_t::decide(const std::vector<std::string> &seqs) {
    init_scratch();
    std::vector<query_result_t> out(seqs.size());
    parlay::parallel_for(0, seqs.size(), [&](size_t i) {
        auto &sc = scratch[(size_t)parlay::worker_id()];
        out[i] = seqs[i].size() > dp.check_bp ? query_scalar(seqs[i].substr(0, dp.check_bp), sc)
                                               : query_scalar(seqs[i], sc);
    });
    return out;
}

// ------------------------------ stats ----------------------------------

u8 dyn_index_t::n_entries() const {
    u8 n = buf->size();
    for (auto &level: levels) for (auto &r: level) n += r.n_entries();
    return n;
}

size_t dyn_index_t::n_runs() const {
    size_t n = 0;
    for (auto &level: levels) n += level.size();
    return n;
}

std::vector<u8> dyn_index_t::run_sizes() const {
    std::vector<u8> out;
    for (size_t i = levels.size(); i-- > 0;)
        for (auto &r: levels[i]) out.push_back(r.n_entries());
    return out;
}

u8 dyn_index_t::n_blocked() const {
    u8 n = 0;
    for (auto &level: levels)
        for (auto &r: level)
            n += parlay::count_if(parlay::iota(r.n_keys()), [&](size_t i) { return r.offsets[i] == r.offsets[i + 1]; });
    return n;
}

u8 dyn_index_t::nbytes() const {
    u8 n = buf->nbytes() + wt.weight.size();
    for (auto &level: levels) for (auto &r: level) n += r.nbytes();
    return n;
}
