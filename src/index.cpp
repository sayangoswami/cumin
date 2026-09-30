//
// Created by Sayan Goswami on 22.09.2026.
//

#include "index.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "parlay/slice.h"
#include <algorithm>
#include <limits>
#include <cmath>

static std::string bucket_fname(char prefix, u4 i) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%c%03u.bin", prefix, i);
    return buf;
}

template <typename T>
static parlay::sequence<T> read_raw(const std::filesystem::path &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = f.tellg();
    if (sz <= 0) return {};
    size_t n = (size_t)sz / sizeof(T);
    parlay::sequence<T> out(n);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(n * sizeof(T)));
    return out;
}

/** Start index (into `vals`) and length of every maximal run of equal, adjacent values. */
template <typename T>
struct runs_t {
    parlay::sequence<size_t> starts;
    parlay::sequence<u4> counts;
};

template <typename T>
static runs_t<T> find_runs(const parlay::sequence<T> &vals) {
    size_t n = vals.size();
    if (n == 0) return {{}, {}};
    auto starts = parlay::pack_index(parlay::tabulate(n, [&](size_t i) {
        return i == 0 || vals[i] != vals[i - 1];
    }));
    size_t m = starts.size();
    auto counts = parlay::tabulate(m, [&](size_t i) {
        size_t e = (i + 1 < m) ? starts[i + 1] : n;
        return (u4)(e - starts[i]);
    });
    return {std::move(starts), std::move(counts)};
}

/**
 * Gather src[range_starts[w]..range_starts[w]+range_counts[w]) for every w in
 * `which`, concatenated in order. Used to compact kept runs into the final
 * window-id array (build()).
 */
template <typename T, typename OffT>
static parlay::sequence<T> gather_ranges(const parlay::sequence<T> &src,
                                          const parlay::sequence<OffT> &range_starts,
                                          const parlay::sequence<u4> &range_counts,
                                          const parlay::sequence<size_t> &which) {
    size_t mk = which.size();
    parlay::sequence<u8> cum(mk + 1);
    cum[0] = 0;
    for (size_t i = 0; i < mk; ++i) cum[i + 1] = cum[i] + range_counts[which[i]];
    u8 total = cum[mk];
    return parlay::tabulate(total, [&](size_t j) {
        size_t hi = (size_t)(std::upper_bound(cum.begin(), cum.end(), (u8)j) - cum.begin()) - 1;
        size_t within = j - cum[hi];
        return src[range_starts[which[hi]] + within];
    });
}

/**
 * Occurrence count of every run for the repeat filter: the number of
 * (anchor, window) memberships, i.e. the run length the index had when it
 * stored every window an anchor lies in. Weighting by window keeps the
 * filter -- and so the set of dropped keys -- exactly what it was before
 * the index stored one id per anchor. See index_t::window_weights().
 */
static parlay::sequence<u4> weighted_run_counts(const runs_t<u8> &runs, const parlay::sequence<u4> &w,
                                                 const parlay::sequence<u1> &weight) {
    return parlay::tabulate(runs.starts.size(), [&](size_t i) {
        u4 c = 0;
        for (size_t j = runs.starts[i], e = j + runs.counts[i]; j < e; ++j) c += weight[w[j]];
        return c;
    });
}

/**
 * Every window voted for by one key: the union of [j-T+1, j] over the
 * key's stored ids `ids` (sorted ascending, may repeat), in ascending
 * order and without duplicates -- so one anchor votes at most once per
 * window (the double-voting fix).
 */
template <typename F>
static inline void expand_windows(const u4 *ids, size_t n, u4 T, F &&emit) {
    u8 next = 0; // lowest window id not yet emitted
    for (size_t i = 0; i < n; ++i) {
        u8 j = ids[i], lo = MAX(next, j + 1 >= T ? j + 1 - T : 0);
        for (u8 w = lo; w <= j; ++w) emit((u4)w);
        next = MAX(next, j + 1);
    }
}

// Occurrence-cutoff percentile over per-feature counts. A heuristic knob -
// drop features occurring above this percentile
static u4 calc_occ_cutoff(parlay::sequence<u4> counts, float pct) {
    if (pct >= 100.0f || counts.empty()) return std::numeric_limits<u4>::max();
    parlay::integer_sort_inplace(counts);
    double rank = (pct / 100.0) * (double)(counts.size() - 1);
    size_t idx = MIN((size_t)std::llround(rank), counts.size() - 1);
    return MAX(counts[idx], (u4)1);
}

// -----------------------------------------------------------------------

index_t::index_t(params_t params, std::string tmpdir_, size_t batch_, size_t chunk_):
    p(params), win_step(params.step()), tmpdir(std::move(tmpdir_)), batch(batch_), chunk(chunk_) {}

index_t::~index_t() {
    for (auto &fh: bucket_ffiles) if (fh.is_open()) fh.close();
    for (auto &fh: bucket_wfiles) if (fh.is_open()) fh.close();
    delete[] hhs;
    delete[] scratch;
}

// Deferred until the first add_record() call, so an index that's only ever
// loaded (query-only, see main.cpp's `map` subcommand) never touches disk
// for build-time scratch space it will never use.
void index_t::open_buckets() {
    if (buckets_open || !p.low_mem) return;
    bucket_dir = std::filesystem::path(tmpdir) /
        ("cumin_buckets_" + std::to_string((u8)time(nullptr)) + "_" + std::to_string((u8)(uintptr_t)this));
    std::filesystem::create_directories(bucket_dir);
    bucket_ffiles.reserve(NB);
    bucket_wfiles.reserve(NB);
    for (u4 i = 0; i < NB; ++i) {
        bucket_ffiles.emplace_back(bucket_dir / bucket_fname('f', i), std::ios::binary);
        bucket_wfiles.emplace_back(bucket_dir / bucket_fname('w', i), std::ios::binary);
        if (!bucket_ffiles.back() || !bucket_wfiles.back())
            log_error("Could not open bucket files in %s", bucket_dir.c_str());
    }
    buckets_open = true;
}

// One entry per anchor: the highest window containing it (j_max). `base`
// is the record's first *real* window id, i.e. already past its padding.
void index_t::file_anchors(const parlay::sequence<u8> &pos, const parlay::sequence<u8> &feat, u8 base) {
    if (pos.empty()) return;
    auto w = parlay::map(pos, [&](u8 p_) { return (u4)(base + p_ / win_step); });
    pending += pos.size();
    stage_feat.push_back(feat);
    stage_win.push_back(std::move(w));
}

// Per window id: how many windows of the old every-window layout an anchor
// with this j_max was filed under -- T, except within the first T-1 real
// windows of a record, where the lower windows don't exist. Padding windows
// never appear as a stored j_max; they get 0.
parlay::sequence<u1> index_t::window_weights() const {
    const u4 T = p.overlap;
    parlay::sequence<u1> wt(base_win, (u1)T);
    for (size_t r = 0; r < rec_first.size(); ++r) {
        u8 first = rec_first[r], end = (r + 1 < rec_first.size()) ? rec_first[r + 1] : base_win;
        for (u8 i = 0; i < T - 1 && first + i < end; ++i) wt[first + i] = 0;
        for (u8 j = 0; j + 1 < T && first + T - 1 + j < end; ++j) wt[first + T - 1 + j] = (u1)(j + 1);
    }
    return wt;
}

u8 index_t::add_record(const std::string &name, const std::string &seq) {
    if (p.low_mem) open_buckets();
    u8 L = seq.size();
    u8 nwin = MAX((u8)1, L / win_step + 1);
    rec_names.push_back(name);
    rec_first.push_back(base_win);
    u8 base = base_win + (p.overlap - 1); // skip this record's padding windows
    base_win = base + nwin;
    if (base_win >= (u8)std::numeric_limits<u4>::max())
        log_error("reference has too many windows (%llu) for 32-bit window ids; increase --win",
                  (unsigned long long)base_win);

    u8 pad = p.k + 4096;
    u8 step = MAX((u8)chunk, pad * 4);
    u8 n_anchor = 0, start = 0;
    while (start < L) {
        u8 end = MIN(L, start + step);
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
                auto pos = parlay::map(idx, [&](size_t i) { return a.positions[i] + lo; });
                auto feat = parlay::map(idx, [&](size_t i) { return a.hashes[i]; });
                n_anchor += pos.size();
                file_anchors(pos, feat, base);
            }
        }
        start = end;
    }
    if (p.low_mem && pending >= batch) flush_low_mem();
    return n_anchor;
}

void index_t::flush_low_mem() {
    if (stage_feat.empty()) return;
    auto feat = parlay::flatten(std::move(stage_feat));
    auto win  = parlay::flatten(std::move(stage_win));
    stage_feat.clear();
    stage_win.clear();
    pending = 0;

    size_t n = feat.size();
    auto perm = parlay::integer_sort(parlay::iota(n), [&](size_t i) { return (u4)(feat[i] >> 56); });
    auto sfeat = parlay::map(perm, [&](size_t i) { return feat[i]; });
    auto swin  = parlay::map(perm, [&](size_t i) { return win[i]; });

    auto buckets = parlay::tabulate(n, [&](size_t i) { return (u4)(sfeat[i] >> 56); });
    std::vector<size_t> boundary(NB + 1);
    for (u4 b = 0; b <= NB; ++b)
        boundary[b] = (size_t)(std::lower_bound(buckets.begin(), buckets.end(), b) - buckets.begin());

    for (u4 b = 0; b < NB; ++b) {
        size_t a = boundary[b], z = boundary[b + 1];
        if (z > a) {
            bucket_ffiles[b].write(reinterpret_cast<const char*>(sfeat.data() + a), (std::streamsize)((z - a) * sizeof(u8)));
            bucket_wfiles[b].write(reinterpret_cast<const char*>(swin.data() + a), (std::streamsize)((z - a) * sizeof(u4)));
        }
    }
    n_entries += n;
}

std::pair<parlay::sequence<u8>, parlay::sequence<u4>> index_t::read_bucket(u4 i) const {
    auto f = read_raw<u8>(bucket_dir / bucket_fname('f', i));
    auto w = read_raw<u4>(bucket_dir / bucket_fname('w', i));
    if (f.empty()) return {std::move(f), std::move(w)};
    size_t n = f.size();
    auto perm = parlay::integer_sort(parlay::iota(n), [&](size_t idx) { return f[idx]; });
    auto sf = parlay::map(perm, [&](size_t idx) { return f[idx]; });
    auto sw = parlay::map(perm, [&](size_t idx) { return w[idx]; });
    return {std::move(sf), std::move(sw)};
}

void index_t::build(int sample_buckets) {
    std::vector<parlay::sequence<u8>> uf_parts;
    std::vector<parlay::sequence<u4>> cn_parts;
    std::vector<parlay::sequence<u4>> ws_parts;
    u8 dropped = 0;
    auto weight = window_weights();

    auto consume_sorted = [&](parlay::sequence<u8> &&f, parlay::sequence<u4> &&w) {
        if (f.empty()) return;
        auto runs = find_runs(f);
        size_t m = runs.starts.size();
        auto occ = weighted_run_counts(runs, w, weight);
        auto keep = parlay::tabulate(m, [&](size_t i) { return occ[i] <= occ_cutoff; });
        auto keep_idx = parlay::pack_index(keep);
        dropped += (m - keep_idx.size());
        if (keep_idx.empty()) return;
        auto uf_i = parlay::map(keep_idx, [&](size_t i) { return f[runs.starts[i]]; });
        auto cn_i = parlay::map(keep_idx, [&](size_t i) { return runs.counts[i]; });
        auto ws_i = gather_ranges(w, runs.starts, runs.counts, keep_idx);
        uf_parts.push_back(std::move(uf_i));
        cn_parts.push_back(std::move(cn_i));
        ws_parts.push_back(std::move(ws_i));
    };

    if (p.low_mem && !buckets_open) {
        // add_record() was never called -- nothing was ever filed.
        ufeat = {}; offsets = {}; windows = {};
        return;
    }

    if (p.low_mem) {
        flush_low_mem();
        for (auto &fh: bucket_ffiles) fh.close();
        for (auto &fh: bucket_wfiles) fh.close();
        bucket_ffiles.clear();
        bucket_wfiles.clear();

        if (p.max_occ_pct >= 100.0f) {
            occ_cutoff = std::numeric_limits<u4>::max();
        } else {
            // Buckets partition by the feature's top byte, so a sample of
            // buckets is an unbiased sample of the whole occurrence
            // distribution -- no need to touch every bucket just to pick a
            // cutoff.
            std::vector<parlay::sequence<u4>> samples;
            u4 stride = MAX((u4)1, NB / (u4)sample_buckets);
            for (u4 i = 0; i < NB; i += stride) {
                auto [f, w] = read_bucket(i);
                if (!f.empty()) samples.push_back(weighted_run_counts(find_runs(f), w, weight));
            }
            occ_cutoff = samples.empty() ? std::numeric_limits<u4>::max()
                                          : calc_occ_cutoff(parlay::flatten(std::move(samples)), p.max_occ_pct);
        }

        for (u4 i = 0; i < NB; ++i) {
            auto [f, w] = read_bucket(i);
            consume_sorted(std::move(f), std::move(w));
            std::error_code ec;
            std::filesystem::remove(bucket_dir / bucket_fname('f', i), ec);
            std::filesystem::remove(bucket_dir / bucket_fname('w', i), ec);
        }
    } else {
        auto feat = parlay::flatten(std::move(stage_feat));
        auto win  = parlay::flatten(std::move(stage_win));
        stage_feat.clear();
        stage_win.clear();
        pending = 0;
        n_entries = feat.size();

        size_t n = feat.size();
        auto perm = parlay::integer_sort(parlay::iota(n), [&](size_t i) { return feat[i]; });
        auto sfeat = parlay::map(perm, [&](size_t i) { return feat[i]; });
        auto swin  = parlay::map(perm, [&](size_t i) { return win[i]; });

        occ_cutoff = (p.max_occ_pct >= 100.0f) ? std::numeric_limits<u4>::max()
                                                : calc_occ_cutoff(weighted_run_counts(find_runs(sfeat), swin, weight), p.max_occ_pct);
        consume_sorted(std::move(sfeat), std::move(swin));
    }

    n_dropped = dropped;
    if (!uf_parts.empty()) {
        ufeat = parlay::flatten(std::move(uf_parts));
        auto cnt = parlay::flatten(std::move(cn_parts));
        auto [off, total] = parlay::scan(parlay::map(cnt, [](u4 c) { return (u8)c; }));
        off.push_back(total);
        offsets = {};
        offsets.wide = total > (u8)std::numeric_limits<u4>::max();
        if (offsets.wide) offsets.o64 = std::move(off);
        else offsets.o32 = parlay::map(off, [](u8 o) { return (u4)o; });

        auto ws = parlay::flatten(std::move(ws_parts));
        windows.pack(ws, packed_array_t::bits_for(MAX(base_win, (u8)1) - 1));
    } else {
        ufeat = {};
        offsets = {};
        windows = {};
    }

    if (p.low_mem) {
        std::error_code ec;
        std::filesystem::remove(bucket_dir, ec);
    }
}

// Header layout (format version 2):
//   magic u4, version u2, key width u1 (64), key encoding u1 (0 = hash),
//   offset width u1 (32/64), window id bits u1, then params_t (incl. overlap).
// The loader refuses anything else rather than misreading it.
void index_t::save(std::ostream &f) const {
    u1 key_bits = 64, key_enc = 0, off_bits = offsets.wide ? 64 : 32, win_bits = (u1)windows.width;
    dump_values(f, MAGIC, FORMAT_VERSION, key_bits, key_enc, off_bits, win_bits);
    p.dump(f);
    dump_strings(f, rec_names);
    dump_seq(f, rec_first);
    dump_values(f, occ_cutoff, n_entries, n_dropped, base_win);
    dump_seq(f, ufeat);
    if (offsets.wide) dump_seq(f, offsets.o64);
    else dump_seq(f, offsets.o32);
    dump_values(f, windows.n);
    dump_seq(f, windows.words);
}

void index_t::load(std::istream &f) {
    u4 magic = 0;
    u2 version = 0;
    u1 key_bits = 0, key_enc = 0, off_bits = 0, win_bits = 0;
    load_values(f, &magic);
    if (!f || magic != MAGIC)
        log_error("not a CUMIN index, or one written before the versioned format (v%u); rebuild it", FORMAT_VERSION);
    load_values(f, &version, &key_bits, &key_enc, &off_bits, &win_bits);
    if (version != FORMAT_VERSION)
        log_error("index format version %u is not supported (expected %u)", version, FORMAT_VERSION);
    if (key_bits != 64 || key_enc != 0 || (off_bits != 32 && off_bits != 64) || win_bits < 1 || win_bits > 32)
        log_error("unsupported index layout: key %u-bit/enc %u, offsets %u-bit, window ids %u-bit",
                  key_bits, key_enc, off_bits, win_bits);

    p.load(f);
    win_step = p.step();
    load_strings(f, rec_names);
    load_seq(f, rec_first);
    load_values(f, &occ_cutoff, &n_entries, &n_dropped, &base_win);
    load_seq(f, ufeat);
    offsets = {};
    offsets.wide = off_bits == 64;
    if (offsets.wide) load_seq(f, offsets.o64);
    else load_seq(f, offsets.o32);
    windows = {};
    windows.width = win_bits;
    load_values(f, &windows.n);
    load_seq(f, windows.words);
    if (!f) log_error("index file is truncated");
}

u8 index_t::nbytes() const {
    return ufeat.size() * sizeof(u8) + offsets.nbytes() + windows.nbytes();
}

std::pair<std::string, u8> index_t::window_locus(u8 wid) const {
    size_t r = (size_t)(std::upper_bound(rec_first.begin(), rec_first.end(), wid) - rec_first.begin());
    r = (r == 0) ? 0 : r - 1;
    u8 first_real = rec_first[r] + (p.overlap - 1);
    u8 wstart = (wid > first_real ? wid - first_real : 0) * win_step;
    return {rec_names[r], wstart};
}

index_t::vote_result_t index_t::vote(const parlay::sequence<u8> &anchors) const {
    if (anchors.empty() || ufeat.empty()) return {0.0f, -1, 0};
    auto sq = parlay::unique(parlay::sort(anchors));
    size_t nq = sq.size();

    auto lo = parlay::tabulate(nq, [&](size_t i) {
        return (size_t)(std::lower_bound(ufeat.begin(), ufeat.end(), sq[i]) - ufeat.begin());
    });
    auto hit = parlay::tabulate(nq, [&](size_t i) {
        return lo[i] < ufeat.size() && ufeat[lo[i]] == sq[i];
    });
    auto hit_pos = parlay::pack_index(hit);
    if (hit_pos.empty()) return {0.0f, -1, (u4)nq};

    auto which = parlay::map(hit_pos, [&](size_t i) { return lo[i]; });

    // Each matched key votes once for every window its postings imply
    const u4 T = p.overlap;
    auto per_key = parlay::map(which, [&](size_t idx) {
        u8 from = offsets[idx], to = offsets[idx + 1];
        auto ids = parlay::sort(parlay::tabulate(to - from, [&](size_t j) { return windows.get(from + j); }));
        parlay::sequence<u4> out;
        expand_windows(ids.data(), ids.size(), T, [&](u4 w) { out.push_back(w); });
        return out;
    });
    auto gathered = parlay::flatten(std::move(per_key));
    if (gathered.empty()) return {0.0f, -1, (u4)nq};
    parlay::integer_sort_inplace(gathered);
    auto runs = find_runs(gathered);

    // Ties go to the higher window id -- see heavyhitter_ht_t::bump
    size_t best = 0;
    for (size_t i = 1; i < runs.counts.size(); ++i)
        if (runs.counts[i] >= runs.counts[best]) best = i;

    u4 best_window = gathered[runs.starts[best]];
    u4 best_count = runs.counts[best];
    return {(float)best_count / (float)nq, (i8)best_window, (u4)nq};
}

index_t::query_result_t index_t::query(const std::string &seq) const {
    auto ra = read_anchors(seq, p.k, p.s, p.t, p.downsample);
    auto vf = vote(ra.fwd);
    auto vr = vote(ra.rev);
    if (vf.score >= vr.score) return {vf.score, vf.window, '+', vf.n_unique};
    return {vr.score, vr.window, '-', vr.n_unique};
}

// ---------------------- scalar batch-query path -------------------------

void index_t::init_query_buffers() {
    if (hhs) return;
    hhs = new heavyhitter_ht_t<u4>[parlay::num_workers()];
    scratch = new query_scratch_t[parlay::num_workers()];
}

index_t::vote_result_t index_t::vote_scalar(std::vector<u8> &anchors, std::vector<u4> &dedup_buf, heavyhitter_ht_t<u4> &hh) const {
    if (anchors.empty() || ufeat.empty()) return {0.0f, -1, 0};
    std::sort(anchors.begin(), anchors.end());

    hh.reset();
    u4 nq = 0;
    for (size_t i = 0; i < anchors.size(); ++i) {
        if (i > 0 && anchors[i] == anchors[i - 1]) continue;
        ++nq;
        auto it = std::lower_bound(ufeat.begin(), ufeat.end(), anchors[i]);
        if (it == ufeat.end() || *it != anchors[i]) continue;
        size_t idx = (size_t)(it - ufeat.begin());
        u8 from = offsets[idx], to = offsets[idx + 1];

        // Vote once for every window this key's postings imply
        dedup_buf.resize(to - from);
        for (u8 j = from; j < to; ++j) dedup_buf[j - from] = windows.get(j);
        std::sort(dedup_buf.begin(), dedup_buf.end());
        expand_windows(dedup_buf.data(), dedup_buf.size(), p.overlap, [&](u4 w) { hh.insert(w); });
    }
    if (hh.top_key == (u4)-1) return {0.0f, -1, nq};
    return {(float)hh.top_count / (float)nq, (i8)hh.top_key, nq};
}

index_t::query_result_t index_t::query_scalar(const std::string &seq, query_scratch_t &sc, heavyhitter_ht_t<u4> &hh) const {
    read_anchors_scalar(seq, p.k, p.s, p.t, p.downsample, sc);
    auto vf = vote_scalar(sc.fwd_hashes, sc.dedup_buf, hh);
    auto vr = vote_scalar(sc.rev_hashes, sc.dedup_buf, hh);
    if (vf.score >= vr.score) return {vf.score, vf.window, '+', vf.n_unique};
    return {vr.score, vr.window, '-', vr.n_unique};
}

std::vector<index_t::query_result_t> index_t::query_batch(const std::vector<std::string> &seqs) {
    init_query_buffers();
    std::vector<query_result_t> out(seqs.size());
    parlay::parallel_for(0, seqs.size(), [&](size_t i) {
        auto w = (size_t)parlay::worker_id();
        out[i] = query_scalar(seqs[i], scratch[w], hhs[w]);
    });
    return out;
}
