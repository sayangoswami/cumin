//
// Created by Sayan Goswami on 22.09.2026.
//

#include "index.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"
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
 * `which`, concatenated in order. Used both to compact kept runs into the
 * final wsorted array (build()) and to gather a query's matched postings
 * (vote()).
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
    p(params), win_step(params.win / 2), tmpdir(std::move(tmpdir_)), batch(batch_), chunk(chunk_) {}

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

void index_t::file_anchors(const parlay::sequence<u8> &pos, const parlay::sequence<u8> &feat, u8 base, u8 nwin) {
    size_t n = pos.size();
    for (int shift : {0, -1}) {
        auto keep = parlay::tabulate(n, [&](size_t i) {
            i8 ww = (i8)(pos[i] / win_step) + shift;
            return ww >= 0 && (u8)ww < nwin;
        });
        auto idx = parlay::pack_index(keep);
        if (idx.empty()) continue;
        auto f = parlay::map(idx, [&](size_t i) { return feat[i]; });
        auto w = parlay::map(idx, [&](size_t i) { return (u4)((i8)(pos[i] / win_step) + shift + (i8)base); });
        pending += idx.size();
        stage_feat.push_back(std::move(f));
        stage_win.push_back(std::move(w));
    }
}

u8 index_t::add_record(const std::string &name, const std::string &seq) {
    if (p.low_mem) open_buckets();
    u8 L = seq.size();
    u8 nwin = MAX((u8)1, L / win_step + 1);
    rec_names.push_back(name);
    rec_first.push_back(base_win);
    u8 base = base_win;
    base_win += nwin;

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
                file_anchors(pos, feat, base, nwin);
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

    auto consume_sorted = [&](parlay::sequence<u8> &&f, parlay::sequence<u4> &&w) {
        if (f.empty()) return;
        auto runs = find_runs(f);
        size_t m = runs.starts.size();
        auto keep = parlay::tabulate(m, [&](size_t i) { return runs.counts[i] <= occ_cutoff; });
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
        ufeat = {}; cnt = {}; wsorted = {}; starts = {};
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
                (void)w;
                if (!f.empty()) samples.push_back(find_runs(f).counts);
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
                                                : calc_occ_cutoff(find_runs(sfeat).counts, p.max_occ_pct);
        consume_sorted(std::move(sfeat), std::move(swin));
    }

    n_dropped = dropped;
    if (!uf_parts.empty()) {
        ufeat = parlay::flatten(std::move(uf_parts));
        cnt = parlay::flatten(std::move(cn_parts));
        wsorted = parlay::flatten(std::move(ws_parts));
        starts = parlay::sequence<u8>(cnt.size());
        u8 running = 0;
        for (size_t i = 0; i < cnt.size(); ++i) { starts[i] = running; running += cnt[i]; }
    } else {
        ufeat = {};
        cnt = {};
        wsorted = {};
        starts = {};
    }

    if (p.low_mem) {
        std::error_code ec;
        std::filesystem::remove(bucket_dir, ec);
    }
}

void index_t::save(std::ostream &f) const {
    p.dump(f);
    dump_strings(f, rec_names);
    dump_seq(f, rec_first);
    dump_values(f, occ_cutoff, n_entries, n_dropped);
    dump_seq(f, ufeat);
    dump_seq(f, starts);
    dump_seq(f, cnt);
    dump_seq(f, wsorted);
}

void index_t::load(std::istream &f) {
    p.load(f);
    win_step = p.win / 2;
    load_strings(f, rec_names);
    load_seq(f, rec_first);
    load_values(f, &occ_cutoff, &n_entries, &n_dropped);
    load_seq(f, ufeat);
    load_seq(f, starts);
    load_seq(f, cnt);
    load_seq(f, wsorted);
}

u8 index_t::nbytes() const {
    return ufeat.size() * sizeof(u8) + starts.size() * sizeof(u8) +
           cnt.size() * sizeof(u4) + wsorted.size() * sizeof(u4);
}

std::pair<std::string, u8> index_t::window_locus(u8 wid) const {
    size_t r = (size_t)(std::upper_bound(rec_first.begin(), rec_first.end(), wid) - rec_first.begin());
    r = (r == 0) ? 0 : r - 1;
    u8 wstart = (wid - rec_first[r]) * win_step;
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
    u8 total = 0;
    for (auto idx: which) total += cnt[idx];
    if (total == 0) return {0.0f, -1, (u4)nq};

    auto gathered = gather_ranges(wsorted, starts, cnt, which);
    parlay::integer_sort_inplace(gathered);
    auto runs = find_runs(gathered);

    size_t best = 0;
    for (size_t i = 1; i < runs.counts.size(); ++i)
        if (runs.counts[i] > runs.counts[best]) best = i;

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
//
// query()/vote() above parallelize *inside* a single read (parlay::sort,
// parlay::tabulate, ..., each allocating its own parlay::sequence) -- fine
// for a one-off standalone query, but at typical read lengths (a few
// hundred bases => a few dozen anchors) that's several small heap
// allocations and scheduler dispatches for essentially no real parallel
// work. The actual embarrassingly-parallel structure of mapping is
// *across* reads, so query_batch() parallelizes there instead (one task
// per read, parlay::parallel_for) and keeps each read's own work scalar
// and allocation-free in steady state via a per-worker-thread reused
// heavy-hitter counter + anchor scratch buffer -- the same split
// collinearity draws between query_fasta/query_batch (batch-parallel) and
// search()'s scalar, heavyhitter_ht_t-based matching.

void index_t::init_query_buffers() {
    if (hhs) return;
    hhs = new heavyhitter_ht_t<u4>[parlay::num_workers()];
    scratch = new query_scratch_t[parlay::num_workers()];
}

index_t::vote_result_t index_t::vote_scalar(std::vector<u8> &anchors, heavyhitter_ht_t<u4> &hh) const {
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
        u8 from = starts[idx], to = from + cnt[idx];
        for (u8 j = from; j < to; ++j) hh.insert(wsorted[j]);
    }
    if (hh.top_key == (u4)-1) return {0.0f, -1, nq};
    return {(float)hh.top_count / (float)nq, (i8)hh.top_key, nq};
}

index_t::query_result_t index_t::query_scalar(const std::string &seq, query_scratch_t &sc, heavyhitter_ht_t<u4> &hh) const {
    read_anchors_scalar(seq, p.k, p.s, p.t, p.downsample, sc);
    auto vf = vote_scalar(sc.fwd_hashes, hh);
    auto vr = vote_scalar(sc.rev_hashes, hh);
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
