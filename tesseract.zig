pub const std = @import("std");
pub const dtypes = @import("src/dtypes.zig");

const tensor = @import("src/tensor.zig");

const tesseract = @This();

pub const Tensor = tensor._Tensor;
pub const anytensor = @import("src/anytensor.zig").anytensor;
pub const scalar = tensor.scalar;

// Expose some of the utility functions that create tensors of specific sizes
pub const constant = tensor.constant;
pub const range = tensor.range;

pub const utils = @import("src/utils.zig");

// pub const viz = Graph.viz;
// pub const Fusion = Graph.Fusion;

test "tesseract" {
    _ = tensor;
}
