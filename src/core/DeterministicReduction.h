#ifndef DETERMINISTICREDUCTION_H
#define DETERMINISTICREDUCTION_H

#include <algorithm>

namespace rsh {

struct IndexRange {
    int begin = 0;
    int end = 0;
};

// Canonicalize floating-point reductions to the historical 4-thread OpenMP
// static partition. Physical thread count may vary, but optimizer-critical
// sums see the same contiguous logical lanes and the same final lane order.
inline int canonical_reduction_lanes() {
    return 4;
}

inline IndexRange canonical_static_range(int n, int lane, int lanes) {
    n = std::max(0, n);
    lanes = std::max(1, lanes);
    lane = std::max(0, std::min(lane, lanes - 1));

    const int base = n / lanes;
    const int extra = n % lanes;
    const int begin = lane * base + std::min(lane, extra);
    const int end = begin + base + (lane < extra ? 1 : 0);
    return {begin, end};
}

} // namespace rsh

#endif
