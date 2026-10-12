#include "forge/runtime/runtime.hpp"

namespace forge::runtime {

struct Context::Impl {
    std::size_t nodes{0};
};

Context::Context() : impl_(std::make_unique<Impl>()) {}
Context::~Context() = default;

std::size_t Context::node_count() const {
    return impl_->nodes;
}

} // namespace forge::runtime
