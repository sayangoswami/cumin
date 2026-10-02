//
// Created by Sayan Goswami on 01.10.2026.
//
// The only translation unit that includes the tiered-vector header. Built
// with -DARRAY -DLEVEL -DPACK (see CMakeLists.txt), the configuration the
// library recommends: array-based tree, level layout, lazily allocated
// leaves with the leaf pointer and offset packed into one word.
//

#include "tiered_buffer.h"
#include "templated_tiered.h"

// Height 4, leaf width 512 (the library's recommended shape), capacity
// 128 * 128 * 64 * 512 = 2^29 entries.
typedef Seq::Tiered<entry_t,
        Seq::LayerItr<Seq::LayerEnd, Seq::Layer<128, Seq::Layer<128, Seq::Layer<64, Seq::Layer<512>>>>>> tiered_impl_t;

struct tiered_vec_t::impl_t {
    tiered_impl_t tv;
};

const size_t tiered_vec_t::CAPACITY = (size_t)128 * 128 * 64 * 512;

tiered_vec_t::tiered_vec_t(): impl(new impl_t()) {}
tiered_vec_t::~tiered_vec_t() = default;

void tiered_vec_t::insert_sorted(const entry_t &e) {
    if (impl->tv.size >= CAPACITY)
        log_error("tiered vector is full (%zu entries); it cannot grow past its compile-time capacity", CAPACITY);
    impl->tv.insert_sorted(e);
}

size_t tiered_vec_t::size() const { return impl->tv.size; }

const entry_t &tiered_vec_t::operator[](size_t i) const { return impl->tv[i]; }

size_t tiered_vec_t::lower_bound(u8 key) const {
    size_t lo = 0, hi = impl->tv.size;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (impl->tv[mid].key < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Equivalent to removing from the back until empty (each such remove only
// decrements the size), and keeps the allocated leaves for reuse. The
// library has no destructor, so recreating it would leak every leaf.
void tiered_vec_t::clear() { impl->tv.size = 0; }
