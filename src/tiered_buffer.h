//
// Created by Sayan Goswami on 01.10.2026.
//

#ifndef CUMIN_TIERED_BUFFER_H
#define CUMIN_TIERED_BUFFER_H

#include "prelude.h"
#include <memory>

/** One (anchor hash, window id) posting, ordered by key then window. */
struct entry_t {
    u8 key;
    u4 win;

    bool operator<(const entry_t &o) const { return key < o.key || (key == o.key && win < o.win); }
    bool operator==(const entry_t &o) const { return key == o.key && win == o.win; }
};

/**
 * Sorted sequence of entries backed by a tiered vector
 * (external/tiered-vector, "Fast Dynamic Arrays", ESA'17): O(n^{1/4})
 * sorted insert, O(1) random access.
 *
 * The tiered-vector header is compile-flag configured and leaks
 * `using namespace std`, macros and a global, so it is confined to
 * tiered_buffer.cpp behind this pimpl.
 */
class tiered_vec_t {
public:
    /** Fixed at compile time by the library: 2^29 entries. */
    static const size_t CAPACITY;

    tiered_vec_t();
    ~tiered_vec_t();
    tiered_vec_t(const tiered_vec_t&) = delete;
    tiered_vec_t& operator=(const tiered_vec_t&) = delete;

    void insert_sorted(const entry_t &e);
    [[nodiscard]] size_t size() const;
    [[nodiscard]] const entry_t &operator[](size_t i) const;
    /** First index whose key is >= key. */
    [[nodiscard]] size_t lower_bound(u8 key) const;
    /** Drop every element; the structure (and its allocated leaves) is reused. */
    void clear();

private:
    struct impl_t;
    std::unique_ptr<impl_t> impl;
};

#endif //CUMIN_TIERED_BUFFER_H
