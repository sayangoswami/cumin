//
// Created by Sayan Goswami on 22.09.2026.
//

#include "cumin.h"

int main() {
    std::string seq =
            "TGTAACCTCCATGTGATGATCTAAAACAATAACAAATAAATAGTTCCTCCCATATAATAT"
            "TATTTCTTACATAATAAAGAATATCATATATTCTCAAAAAATAACAAATAATATCCTCTT"
            "TCCATTCTCAATTAAGTTCTTAAATGAGAATAAAAGGGTAATCCTCTGTATTTCTTAA";

    const u4 k = 15, s = 9, t = 0, downsample = 1;
    auto codes = encode(seq);
    auto a = syncmer_anchors(codes, k, s, t, downsample);
    log_info("Found %zu syncmer anchors out of %zu k-mers.", a.positions.size(), seq.size() - k + 1);
    for (size_t i = 0; i < a.positions.size(); ++i)
        printf("pos=%llu hash=%llu\n", (unsigned long long)a.positions[i], (unsigned long long)a.hashes[i]);
    return 0;
}
