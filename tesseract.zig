pub const std = @import("std");
pub const dtypes = @import("src/dtypes.zig");

const tensor = @import("src/tensor.zig");
const Graph = @import("src/Graph.zig");
const Program = @import("src/Program.zig");

const tesseract = @This();

// Expose the simple Tensor function rather than the full one
pub const Tensor = tensor.InferredStrides;
pub const Scalar = tensor.Scalar;

// Expose some of the utility functions that create tensors of specific sizes
pub const constant = tensor.constant;
pub const range = tensor.range;

pub const trace = Graph.trace;
pub const viz = Graph.viz;
pub const Fusion = Graph.Fusion;

pub const code = Program.code;

test "tesseract" {
    _ = tensor;
    _ = Graph;
    _ = Program;
}

pub fn init() void {
    Graph.init(std.heap.page_allocator);
    Program.init(std.heap.page_allocator);
}

pub fn deinit() void {
    Graph.deinit();
    Program.deinit();
}
