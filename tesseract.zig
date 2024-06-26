pub const std = @import("std");
pub const dtypes = @import("src/dtypes.zig");
pub const graph = @import("src/graph.zig");

const tensor = @import("src/tensor.zig");

pub const Tensor = tensor.tensor;
pub const AnyTensor = @import("src/anytensor.zig").AnyTensor;
pub const scalar = tensor.scalar;

// Expose some of the utility functions that create tensors of specific sizes
pub const constant = tensor.constant;
pub const range = tensor.range;

pub const utils = @import("src/utils.zig");
pub const nn = @import("src/nn.zig");

test "tesseract" {
    _ = tensor;
}
