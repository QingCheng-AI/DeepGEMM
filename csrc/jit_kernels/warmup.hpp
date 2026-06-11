#pragma once

#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "../jit/compiler.hpp"
#include "../utils/system.hpp"

namespace deep_gemm {

// Pre-compile every distinct JIT kernel that a GEMM op may select for a given
// weight shape, to avoid runtime compilation jitter during inference.
//
// In LLM inference `n`/`k` (and `num_groups`, majors, dtypes, `compiled_dims`)
// are fixed by the weight, so the only free variable is `m`.
// sweeping the free variable over `[1, DG_WARMUP_MAX_M]`, building each
// distinct kernel.
//
// On the first call for a given (kernel_name, n, k, num_groups, compiled_dims)
// the sweep runs synchronously.
// Set DG_WARMUP_MAX_M=0 (default) to disable.
//
// `gen` is a callable `int(swept_var) -> std::string(code)` supplied by each
// call site.
template <typename GenFn>
static void maybe_warmup(const std::string &kernel_name, const int &n,
                         const int &k, const int &num_groups,
                         const std::string &compiled_dims, GenFn &&gen) {
    const int max_m = get_env<int>("DG_WARMUP_MAX_M", 0);
    if (max_m <= 0)
        return;

    using Key = std::tuple<std::string, int, int, int, std::string>;
    static std::set<Key> warmed_keys;
    static std::mutex mu;

    {
        const Key key{kernel_name, n, k, num_groups, compiled_dims};
        std::lock_guard<std::mutex> lock(mu);
        if (not warmed_keys.insert(key).second)
            return;
    }

    int num_built = 0;
    std::set<std::string> seen_codes;
    for (int x = 1; x <= max_m; ++x) {
        const std::string code = gen(x);
        if (seen_codes.insert(code).second) {
            compiler->build(kernel_name, code);
            ++num_built;
        }
    }

    printf("[DeepGEMM warmup] kernel=%s n=%d k=%d num_groups=%d: "
           "swept x=[1,%d], built %d distinct kernels\n",
           kernel_name.c_str(), n, k, num_groups, max_m, num_built);
}

} // namespace deep_gemm
