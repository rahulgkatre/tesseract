pub const std = @import("std");
pub const dtypes = @import("src/dtypes.zig");

const tensor = @import("src/tensor.zig");
const Graph = @import("src/Graph.zig");

const tesseract = @This();

pub const Tensor = tensor.UserTensor;
pub const Scalar = tensor.Scalar;
pub const F = @import("src/functions.zig");

// Expose some of the utility functions that create tensors of specific sizes
pub const constant = tensor.constant;
pub const range = tensor.range;

pub const trace = Graph.trace;
// pub const viz = Graph.viz;
// pub const Fusion = Graph.Fusion;

test "tesseract" {
    _ = tensor;
    _ = Graph;
}

pub fn init() void {
    Graph.init();
}

pub fn deinit() void {
    Graph.deinit();
}
