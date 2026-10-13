#pragma once
#include "forge/ir/graph.hpp"
#include <cstddef>

namespace forge::ir {

struct ShapeInferenceStats {
    std::size_t nodes_processed{0};
    std::size_t tensors_inferred{0};
    std::size_t nodes_skipped{0};
};

ShapeInferenceStats infer_shapes(Graph& g);

} // namespace forge::ir
