//
// Created by Sayan Goswami on 22.09.2026.
//

#include "cumin.h"
#include "index.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct SyncmerAnchors {
    std::vector<u8> positions;
    std::vector<u8> hashes;

    SyncmerAnchors() = default;
    explicit SyncmerAnchors(anchors_t &a):
        positions(a.positions.begin(), a.positions.end()),
        hashes(a.hashes.begin(), a.hashes.end()) {}
};

struct ReadAnchors {
    std::vector<u8> fwd, rev;

    ReadAnchors() = default;
    explicit ReadAnchors(read_anchors_t &a):
        fwd(a.fwd.begin(), a.fwd.end()), rev(a.rev.begin(), a.rev.end()) {}
};

static SyncmerAnchors py_syncmer_anchors(std::string &seq, u4 k, u4 s, u4 t, u4 downsample) {
    auto codes = encode(seq);
    auto a = syncmer_anchors(codes, k, s, t, downsample);
    return SyncmerAnchors(a);
}

static ReadAnchors py_read_anchors(std::string &seq, u4 k, u4 s, u4 t, u4 downsample) {
    auto a = read_anchors(seq, k, s, t, downsample);
    return ReadAnchors(a);
}

struct QueryResult {
    float score = 0.0f;
    i8 window = -1;
    char strand = '+';
    u4 n_anchors = 0;

    QueryResult() = default;
    explicit QueryResult(const index_t::query_result_t &r):
        score(r.score), window(r.window), strand(r.strand), n_anchors(r.n_anchors) {}
};

struct QueryRequest {
    std::string id, seq;
    QueryRequest() = default;
    QueryRequest(std::string id, std::string seq): id(std::move(id)), seq(std::move(seq)) {}
};

struct QueryResponse {
    std::string id;
    QueryResult result;
    QueryResponse() = default;
    QueryResponse(std::string id, QueryResult result): id(std::move(id)), result(result) {}
};

/**
 * pybind can't yield lazily from a C++ generator, so (as in
 * collinearity/src/pybindings.cpp's ResponseGenerator) query_stream()
 * still computes the whole batch eagerly -- what this buys over plain
 * query_batch() is just the (id, seq) -> (id, result) bookkeeping, for
 * callers (see pycumin.Aligner.map_reads) that want results paired back up
 * with their own per-read ids rather than relying on input/output order.
 */
struct QueryResponseGenerator {
    std::vector<QueryResponse> responses;
    size_t i = 0;
    explicit QueryResponseGenerator(std::vector<QueryResponse> &&r): responses(std::move(r)) {}
    QueryResponse next() {
        if (i >= responses.size()) throw py::stop_iteration();
        return responses[i++];
    }
    QueryResponseGenerator &iter() { return *this; }
};

/**
 * Thin pybind wrapper around index_t, following pycollinearity.Index's
 * shape: a build-time constructor (anchor/build parameters), then either
 * add_record()+build() or load() from a saved index, then query().
 */
struct Index {
    index_t ix;

    explicit Index(u4 k = 15, u4 s = 8, u4 t = 0, u4 downsample = 2, u4 win = 4000,
                    float max_occ_pct = 99.9f, bool low_mem = true, std::string tmpdir = "/tmp",
                    size_t batch = 20'000'000, size_t chunk = 32'000'000):
        ix(params_t(k, s, t, downsample, win, max_occ_pct, low_mem), std::move(tmpdir), batch, chunk) {}

    u8 add_record(std::string &name, std::string &seq) { return ix.add_record(name, seq); }
    void build(int sample_buckets = 16) { ix.build(sample_buckets); }

    void save(const std::string &path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) log_error("could not open %s for writing", path.c_str());
        ix.save(f);
    }

    void load(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) log_error("could not open %s", path.c_str());
        ix.load(f);
    }

    [[nodiscard]] QueryResult query(std::string &seq) const { return QueryResult(ix.query(seq)); }

    // Batch-parallel scalar path (index_t::query_batch) -- see the design
    // note in index.cpp. Prefer these over calling query() in a Python-side
    // loop for anything more than a handful of reads.
    std::vector<QueryResult> query_batch(const py::list &seqs) {
        std::vector<std::string> v;
        v.reserve(seqs.size());
        for (auto &s: seqs) v.push_back(s.cast<std::string>());
        auto results = ix.query_batch(v);
        return {results.begin(), results.end()};
    }

    QueryResponseGenerator query_stream(const py::iterator &reads) {
        std::vector<std::string> ids, seqs;
        for (auto &item: reads) {
            auto req = item.cast<QueryRequest>();
            ids.push_back(std::move(req.id));
            seqs.push_back(std::move(req.seq));
        }
        auto results = ix.query_batch(seqs);
        std::vector<QueryResponse> responses;
        responses.reserve(results.size());
        for (size_t i = 0; i < results.size(); ++i)
            responses.emplace_back(std::move(ids[i]), QueryResult(results[i]));
        return QueryResponseGenerator(std::move(responses));
    }

    [[nodiscard]] std::pair<std::string, u8> window_locus(i8 wid) const { return ix.window_locus((u8)wid); }
    [[nodiscard]] u8 nbytes() const { return ix.nbytes(); }
    [[nodiscard]] u8 n_entries() const { return ix.n_entries; }
    [[nodiscard]] u8 n_dropped() const { return ix.n_dropped; }
    [[nodiscard]] u4 occ_cutoff() const { return ix.occ_cutoff; }
    [[nodiscard]] std::string params_str() const { return ix.p.to_string(); }
    [[nodiscard]] params_t params() const { return ix.p; }
};

PYBIND11_MODULE(_core, m) {
    py::class_<SyncmerAnchors>(m, "SyncmerAnchors")
            .def(py::init<>())
            .def_readonly("positions", &SyncmerAnchors::positions)
            .def_readonly("hashes", &SyncmerAnchors::hashes);

    py::class_<ReadAnchors>(m, "ReadAnchors")
            .def(py::init<>())
            .def_readonly("fwd", &ReadAnchors::fwd)
            .def_readonly("rev", &ReadAnchors::rev);

    m.def("syncmer_anchors", &py_syncmer_anchors,
          py::arg("seq"), py::arg("k"), py::arg("s"), py::arg("t"), py::arg("downsample"));

    m.def("read_anchors", &py_read_anchors,
          py::arg("seq"), py::arg("k"), py::arg("s"), py::arg("t"), py::arg("downsample"));

    py::class_<QueryResult>(m, "QueryResult")
            .def(py::init<>())
            .def_readonly("score", &QueryResult::score)
            .def_readonly("window", &QueryResult::window)
            .def_readonly("strand", &QueryResult::strand)
            .def_readonly("n_anchors", &QueryResult::n_anchors);

    py::class_<params_t>(m, "params_t")
            .def(py::init<>())
            .def_readonly("k", &params_t::k)
            .def_readonly("s", &params_t::s)
            .def_readonly("t", &params_t::t)
            .def_readonly("downsample", &params_t::downsample)
            .def_readonly("win", &params_t::win)
            .def_readonly("max_occ_pct", &params_t::max_occ_pct)
            .def_readonly("low_mem", &params_t::low_mem)
            .def_property_readonly("density", &params_t::density)
            .def("__repr__", &params_t::to_string);

    py::class_<QueryRequest>(m, "QueryRequest")
            .def(py::init<std::string, std::string>(), py::arg("id"), py::arg("seq"))
            .def_readwrite("id", &QueryRequest::id)
            .def_readwrite("seq", &QueryRequest::seq);

    py::class_<QueryResponse>(m, "QueryResponse")
            .def(py::init<std::string, QueryResult>(), py::arg("id"), py::arg("result"))
            .def_readwrite("id", &QueryResponse::id)
            .def_readwrite("result", &QueryResponse::result);

    py::class_<QueryResponseGenerator>(m, "QueryResponseGenerator")
            .def("__iter__", &QueryResponseGenerator::iter)
            .def("__next__", &QueryResponseGenerator::next);

    py::class_<Index>(m, "Index")
            .def(py::init<u4, u4, u4, u4, u4, float, bool, std::string, size_t, size_t>(),
                 py::arg("k") = 15, py::arg("s") = 8, py::arg("t") = 0, py::arg("downsample") = 2,
                 py::arg("win") = 4000, py::arg("max_occ_pct") = 99.9f, py::arg("low_mem") = true,
                 py::arg("tmpdir") = "/tmp", py::arg("batch") = 20'000'000, py::arg("chunk") = 32'000'000)
            .def("add_record", &Index::add_record, py::arg("name"), py::arg("seq"))
            .def("build", &Index::build, py::arg("sample_buckets") = 16)
            .def("save", &Index::save, py::arg("path"))
            .def("load", &Index::load, py::arg("path"))
            .def("query", &Index::query, py::arg("seq"))
            .def("query_batch", &Index::query_batch, py::arg("seqs"))
            .def("query_stream", &Index::query_stream, py::arg("reads"))
            .def("window_locus", &Index::window_locus, py::arg("wid"))
            .def("nbytes", &Index::nbytes)
            .def_property_readonly("n_entries", &Index::n_entries)
            .def_property_readonly("n_dropped", &Index::n_dropped)
            .def_property_readonly("occ_cutoff", &Index::occ_cutoff)
            .def_property_readonly("p", &Index::params)
            .def("__repr__", &Index::params_str);
}
