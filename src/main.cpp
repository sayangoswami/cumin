//
// Created by Sayan Goswami on 22.09.2026.
//

#include "index.h"
#include "dyn_index.h"
#include <kseq++/seqio.hpp>
#include <argparse/argparse.hpp>
#include <chrono>
#include <sys/resource.h>

using klibpp::SeqStreamIn;
using klibpp::KSeq;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double peak_rss_gb() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return (double)ru.ru_maxrss / 1e9;  // bytes
#else
    return (double)ru.ru_maxrss / 1e6;  // KiB
#endif
}

static const char *HITS_HEADER = "read_id\tquery_bp\tscore\tn_anchors\tstrand\tref\twindow_start\tdecision";

/** One `cumin map` TSV row (without the trailing newline). */
template <typename Index, typename Result>
static void write_hit(std::ostream &fout, const Index &ix, const std::string &name, size_t bp, const Result &r,
                      const char *decision) {
    std::string ref = "*";
    i8 wstart = -1;
    if (r.window >= 0) {
        auto locus = ix.window_locus((u8)r.window);
        ref = locus.first;
        wstart = (i8)locus.second;
    }
    fout << name << '\t' << bp << '\t' << r.score << '\t' << r.n_anchors << '\t'
         << r.strand << '\t' << ref << '\t' << wstart << '\t' << decision;
}

// argparse::Args doesn't support composing another Args struct's fields by
// value, so anchor params are just inlined directly (mirrors config.h's flat
// args_t in collinearity -- one struct per CLI verb, not a shared mixin).
struct build_args_t : argparse::Args {
    std::string &ref = kwarg("ref", "path to reference FASTA (.gz ok)");
    std::string &idx = kwarg("idx", "path to write the built index to");
    int &k = kwarg("k", "anchor k-mer size").set_default(15);
    int &s = kwarg("s", "syncmer s-mer size").set_default(8);
    int &t = kwarg("t", "syncmer offset (0 = open)").set_default(0);
    int &downsample = kwarg("downsample", "density = 1/((k-s+1)*downsample)").set_default(2);
    int &win = kwarg("win", "reference window (bp)").set_default(4000);
    int &overlap = kwarg("overlap", "overlap factor T: windows start every win/T bp; reads up to "
        "win - win/T bp are contained in one window. Costs query votes, not index bytes").set_default(2);
    float &max_occ_pct = kwarg("max-occ-pct", "drop features above this occurrence percentile").set_default(99.9f);
    bool &high_mem = flag("high-mem", "sort the whole reference's anchors in memory in one pass "
        "instead of streaming through on-disk radix buckets; faster if it fits in RAM.");
    std::string &tmpdir = kwarg("tmpdir", "scratch dir for radix buckets (low-mem mode only)").set_default("/tmp");
    int &batch = kwarg("batch", "anchors staged in memory before a low-mem flush to disk").set_default(20'000'000);
    int &chunk = kwarg("chunk", "max bp of one record processed at a time").set_default(32'000'000);

    [[nodiscard]] params_t to_params() const {
        return {(u4)k, (u4)s, (u4)t, (u4)downsample, (u4)win, max_occ_pct, !high_mem, (u4)overlap};
    }

    int run() override {
        log_info("building: %s", to_params().to_string().c_str());
        index_t ix(to_params(), tmpdir, (size_t)batch, (size_t)chunk);

        u8 bp = 0, anchors = 0, nrec = 0;
        SeqStreamIn iss(ref.c_str());
        KSeq record;
        while (iss >> record) {
            bp += record.seq.size();
            anchors += ix.add_record(record.name, record.seq);
            ++nrec;
            if (nrec % 20000 == 0) log_info("  ... %llu records, %.2f Gbp", (unsigned long long)nrec, bp / 1e9);
        }
        if (bp == 0) log_error("no reference sequence loaded from %s", ref.c_str());
        log_info("  %llu bp, %llu anchors (density 1/%.1f)", (unsigned long long)bp, (unsigned long long)anchors,
                 (double)bp / (double)MAX(anchors, (u8)1));

        ix.build();
        log_info("  %llu entries, %llu distinct keys, %llu windows (%u-bit ids, %s offsets)",
                 (unsigned long long)ix.n_entries, (unsigned long long)ix.n_keys(),
                 (unsigned long long)ix.n_windows(), ix.window_bits(), ix.wide_offsets() ? "64-bit" : "32-bit");
        log_info("  occ cutoff %u, %llu repeat features suppressed, %.2f GB (%.3f B/base)",
                 ix.occ_cutoff, (unsigned long long)ix.n_dropped,
                 ix.nbytes() / 1e9, (double)ix.nbytes() / (double)MAX(bp, (u8)1));

        std::ofstream fout(idx, std::ios::binary);
        if (!fout) log_error("could not open %s for writing", idx.c_str());
        ix.save(fout);
        log_info("wrote %s", idx.c_str());
        return 0;
    }
};

struct map_args_t : argparse::Args {
    std::string &idx = kwarg("idx", "path to a built index");
    std::string &reads = kwarg("reads", "path to reads (FASTA/FASTQ, .gz ok)");
    std::string &out = kwarg("out", "output TSV path").set_default("hits.tsv");
    float &threshold = kwarg("threshold", "keep/reject cut; set it from an eval run").set_default(0.15f);
    int &max_reads = kwarg("max-reads", "stop after this many reads (0 = no limit)").set_default(0);
    int &max_query_bp = kwarg("max-query-bp", "truncate each read to its first N bases before querying (0 = no limit)").set_default(0);

    int run() override {
        index_t ix;
        std::ifstream fin(idx, std::ios::binary);
        if (!fin) log_error("could not open %s", idx.c_str());
        ix.load(fin);
        log_info("loaded %s: %llu entries, %.2f GB, %s", idx.c_str(), (unsigned long long)ix.n_entries,
                 ix.nbytes() / 1e9, ix.p.to_string().c_str());

        std::ofstream fout(out);
        if (!fout) log_error("could not open %s for writing", out.c_str());
        fout << HITS_HEADER << '\n';

        // Batched: parallel *across* reads via ix.query_batch() (see
        // index.cpp's design note), rather than one ix.query() call per
        // read -- matters at typical read lengths, where a single query's
        // own work is too small to be worth parallelizing on its own.
        constexpr size_t BATCH_SZ = 4096;
        std::vector<std::string> names, seqs;
        names.reserve(BATCH_SZ);
        seqs.reserve(BATCH_SZ);

        auto flush_batch = [&]() {
            auto results = ix.query_batch(seqs);
            for (size_t i = 0; i < results.size(); ++i) {
                write_hit(fout, ix, names[i], seqs[i].size(), results[i], results[i].score > threshold ? "keep" : "reject");
                fout << '\n';
            }
            names.clear();
            seqs.clear();
        };

        u8 n = 0;
        SeqStreamIn iss(reads.c_str());
        KSeq record;
        while (iss >> record) {
            auto &seq = record.seq;
            if (max_query_bp > 0 && (int)seq.size() > max_query_bp) seq.resize(max_query_bp);
            if (seq.size() < ix.p.k + 5) continue;

            names.push_back(record.name);
            seqs.push_back(std::move(seq));
            ++n;
            if (seqs.size() == BATCH_SZ) flush_batch();
            if (max_reads > 0 && (int)n >= max_reads) break;
        }
        if (!seqs.empty()) flush_batch();
        log_info("%llu reads -> %s", (unsigned long long)n, out.c_str());
        return 0;
    }
};

struct dynbuild_args_t : argparse::Args {
    std::string &ref = kwarg("ref", "reference FASTA inserted record by record, without the skip check "
        "(with --stream: a starting reference)").set_default("");
    std::string &reads = kwarg("reads", "after inserting --ref, map these reads and write a `cumin map` TSV").set_default("");
    std::string &stream = kwarg("stream", "simulate adaptive sampling over these reads: decide on each read's "
        "first --check-bp bases; reject it if it maps, otherwise add the whole read").set_default("");
    std::string &out = kwarg("out", "output TSV path").set_default("hits.tsv");
    std::string &variant = kwarg("variant", "lsm | tiered | lsm-tiered").set_default("lsm");
    int &buffer = kwarg("buffer", "entries buffered before sealing a run (lsm, lsm-tiered)").set_default(1 << 20);
    int &ratio = kwarg("ratio", "runs per level before they merge into the next").set_default(2);
    int &max_occ = kwarg("max-occ", "repeat cutoff (weighted occurrences; a static build reports its "
        "own as `occ cutoff`); 0 = no filter").set_default(0);
    float &threshold = kwarg("threshold", "score above which a read maps").set_default(0.15f);
    int &check_bp = kwarg("check-bp", "prefix length used for the map/reject decision").set_default(360);
    int &batch = kwarg("batch", "--stream: reads decided together, as by concurrent channels").set_default(512);
    int &max_reads = kwarg("max-reads", "stop after this many reads (0 = no limit)").set_default(0);
    int &log_every = kwarg("log-every", "progress line every N records or reads").set_default(100000);
    int &k = kwarg("k", "anchor k-mer size").set_default(15);
    int &s = kwarg("s", "syncmer s-mer size").set_default(8);
    int &t = kwarg("t", "syncmer offset (0 = open)").set_default(0);
    int &downsample = kwarg("downsample", "density = 1/((k-s+1)*downsample)").set_default(2);
    int &win = kwarg("win", "reference window (bp)").set_default(4000);
    int &overlap = kwarg("overlap", "overlap factor T: windows start every win/T bp").set_default(2);

    dyn_index_t *ix = nullptr;
    double add_s = 0, worst_add_s = 0;
    u8 added_bp = 0;

    void log_state(const char *what, u8 n) const {
        auto sizes = ix->run_sizes();
        std::string runs;
        for (auto r: sizes) runs += (runs.empty() ? "" : ",") + std::to_string(r);
        log_info("  %s %llu: %llu entries, buffer %zu, %zu runs [%s], %.3f GB, insert %.2f Mbp/s, "
                 "worst add %.1f ms, %llu flushes / %llu merges (worst %.1f ms, %.1fx write amp), peak RSS %.2f GB",
                 what, (unsigned long long)n, (unsigned long long)ix->n_entries(), ix->buffer_size(), ix->n_runs(),
                 runs.c_str(), ix->nbytes() / 1e9, added_bp / MAX(add_s, 1e-9) / 1e6, worst_add_s * 1e3,
                 (unsigned long long)ix->n_flushes, (unsigned long long)ix->n_merges, ix->worst_flush_s * 1e3,
                 (double)ix->entries_written / (double)MAX(ix->n_entries(), (u8)1), peak_rss_gb());
    }

    bool timed_add(const std::string &name, const std::string &seq, bool skip_mapped) {
        double t0 = now_s();
        bool added = ix->add_record(name, seq, skip_mapped);
        double dt = now_s() - t0;
        add_s += dt;
        worst_add_s = MAX(worst_add_s, dt);
        if (added) added_bp += seq.size();
        return added;
    }

    void insert_ref() {
        u8 nrec = 0;
        double t0 = now_s();
        SeqStreamIn iss(ref.c_str());
        KSeq record;
        while (iss >> record) {
            timed_add(record.name, record.seq, false);
            if (++nrec % (u8)log_every == 0) log_state("records", nrec);
        }
        if (nrec == 0) log_error("no reference sequence loaded from %s", ref.c_str());
        log_state("records", nrec);
        log_info("inserted %llu records, %.2f Mbp in %.2f s", (unsigned long long)nrec, added_bp / 1e6, now_s() - t0);
    }

    void map_reads() {
        std::ofstream fout(out);
        if (!fout) log_error("could not open %s for writing", out.c_str());
        fout << HITS_HEADER << '\n';
        std::vector<std::string> names, seqs;
        auto flush_batch = [&]() {
            auto results = ix->query_batch(seqs);
            for (size_t i = 0; i < results.size(); ++i) {
                write_hit(fout, *ix, names[i], seqs[i].size(), results[i], ix->is_mapped(results[i]) ? "keep" : "reject");
                fout << '\n';
            }
            names.clear();
            seqs.clear();
        };
        u8 n = 0;
        double t0 = now_s();
        SeqStreamIn iss(reads.c_str());
        KSeq record;
        while (iss >> record) {
            if (record.seq.size() < ix->p.k + 5) continue;
            names.push_back(record.name);
            seqs.push_back(std::move(record.seq));
            ++n;
            if (seqs.size() == 4096) flush_batch();
            if (max_reads > 0 && (int)n >= max_reads) break;
        }
        if (!seqs.empty()) flush_batch();
        double dt = now_s() - t0;
        log_info("%llu reads mapped in %.2f s (%.1f us/read) -> %s", (unsigned long long)n, dt, dt / MAX(n, (u8)1) * 1e6,
                 out.c_str());
    }

    void stream_reads() {
        std::ofstream fout(out);
        if (!fout) log_error("could not open %s for writing", out.c_str());
        fout << HITS_HEADER << "\tadded\n";
        std::vector<std::string> names, seqs;
        u8 n = 0, n_reject = 0, n_added = 0, n_recheck = 0;
        double decide_s = 0, worst_decide_s = 0;
        u8 next_log = (u8)log_every;

        auto flush_batch = [&]() {
            double t0 = now_s();
            auto results = ix->decide(seqs);
            double dt = now_s() - t0;
            decide_s += dt;
            worst_decide_s = MAX(worst_decide_s, dt);
            for (size_t i = 0; i < results.size(); ++i) {
                bool mapped = ix->is_mapped(results[i]);
                bool added = false;
                if (mapped) ++n_reject;
                else if ((added = timed_add(names[i], seqs[i], true))) ++n_added;
                else ++n_recheck; // a read earlier in this batch already covered it
                write_hit(fout, *ix, names[i], MIN(seqs[i].size(), (size_t)check_bp), results[i],
                          mapped ? "reject" : "keep");
                fout << '\t' << (added ? 1 : 0) << '\n';
            }
            names.clear();
            seqs.clear();
            if (n >= next_log) {
                log_info("  reads %llu: %llu rejected, %llu added, %llu skipped on re-check; decide %.1f us/read, "
                         "worst batch %.2f ms", (unsigned long long)n, (unsigned long long)n_reject,
                         (unsigned long long)n_added, (unsigned long long)n_recheck, decide_s / MAX(n, (u8)1) * 1e6,
                         worst_decide_s * 1e3);
                log_state("reads", n);
                next_log += (u8)log_every;
            }
        };

        SeqStreamIn iss(stream.c_str());
        KSeq record;
        while (iss >> record) {
            if (record.seq.size() < ix->p.k + 5) continue;
            names.push_back(record.name);
            seqs.push_back(std::move(record.seq));
            ++n;
            if ((int)seqs.size() == batch) flush_batch();
            if (max_reads > 0 && (int)n >= max_reads) break;
        }
        if (!seqs.empty()) flush_batch();
        log_info("streamed %llu reads: %llu rejected, %llu added, %llu skipped on re-check; decide %.1f us/read",
                 (unsigned long long)n, (unsigned long long)n_reject, (unsigned long long)n_added,
                 (unsigned long long)n_recheck, decide_s / MAX(n, (u8)1) * 1e6);
        log_state("reads", n);
    }

    int run() override {
        if (ref.empty() && stream.empty()) log_error("give --ref, --stream, or both");
        if (!reads.empty() && !stream.empty()) log_error("--reads and --stream are separate modes");
        params_t params((u4)k, (u4)s, (u4)t, (u4)downsample, (u4)win, 100.0f, true, (u4)overlap);
        dyn_params_t dparams;
        dparams.variant = parse_dyn_variant(variant);
        dparams.buffer = (size_t)buffer;
        dparams.ratio = (u4)ratio;
        dparams.max_occ = max_occ > 0 ? (u4)max_occ : std::numeric_limits<u4>::max();
        dparams.threshold = threshold;
        dparams.check_bp = (u4)check_bp;
        log_info("dynamic index (%s, buffer %d, ratio %d, max-occ %d): %s", dyn_variant_name(dparams.variant),
                 buffer, ratio, max_occ, params.to_string().c_str());
        dyn_index_t index(params, dparams);
        ix = &index;

        if (!ref.empty()) insert_ref();
        if (!reads.empty()) map_reads();
        if (!stream.empty()) stream_reads();
        ix = nullptr;
        return 0;
    }
};

struct cumin_args_t : argparse::Args {
    build_args_t &build = subcommand("build");
    map_args_t &map = subcommand("map");
    dynbuild_args_t &dynbuild = subcommand("dynbuild");
};

int main(int argc, char *argv[]) {
    auto args = argparse::parse<cumin_args_t>(argc, argv);
    return args.run_subcommands();
}
