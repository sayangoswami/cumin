//
// Created by Sayan Goswami on 22.09.2026.
//

#include "index.h"
#include <kseq++/seqio.hpp>
#include <argparse/argparse.hpp>

using klibpp::SeqStreamIn;
using klibpp::KSeq;

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
    float &max_occ_pct = kwarg("max-occ-pct", "drop features above this occurrence percentile").set_default(99.9f);
    bool &high_mem = flag("high-mem", "sort the whole reference's anchors in memory in one pass "
        "instead of streaming through on-disk radix buckets; faster if it fits in RAM.");
    std::string &tmpdir = kwarg("tmpdir", "scratch dir for radix buckets (low-mem mode only)").set_default("/tmp");
    int &batch = kwarg("batch", "anchors staged in memory before a low-mem flush to disk").set_default(20'000'000);
    int &chunk = kwarg("chunk", "max bp of one record processed at a time").set_default(32'000'000);

    [[nodiscard]] params_t to_params() const {
        return {(u4)k, (u4)s, (u4)t, (u4)downsample, (u4)win, max_occ_pct, !high_mem};
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
        log_info("  %llu entries, occ cutoff %u, %llu repeat features suppressed, %.2f GB (%.2f B/base)",
                 (unsigned long long)ix.n_entries, ix.occ_cutoff, (unsigned long long)ix.n_dropped,
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
        fout << "read_id\tquery_bp\tscore\tn_anchors\tstrand\tref\twindow_start\tdecision\n";

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
                auto &r = results[i];
                std::string ref = "*";
                i8 wstart = -1;
                if (r.window >= 0) {
                    auto locus = ix.window_locus((u8)r.window);
                    ref = locus.first;
                    wstart = (i8)locus.second;
                }
                fout << names[i] << '\t' << seqs[i].size() << '\t' << r.score << '\t' << r.n_anchors << '\t'
                     << r.strand << '\t' << ref << '\t' << wstart << '\t'
                     << (r.score > threshold ? "keep" : "reject") << '\n';
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

struct cumin_args_t : argparse::Args {
    build_args_t &build = subcommand("build");
    map_args_t &map = subcommand("map");
};

int main(int argc, char *argv[]) {
    auto args = argparse::parse<cumin_args_t>(argc, argv);
    return args.run_subcommands();
}
