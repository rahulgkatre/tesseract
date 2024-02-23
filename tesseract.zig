pub const std = @import("std");
pub const dtypes = @import("src/dtypes.zig");

// Expose functions publicly when using Tesseract as a Zig library
const tensor = @import("src/tensor.zig");
pub const Graph = @import("src/Graph.zig");
pub const Program = @import("src/Program.zig");

// Expose the simple Tensor function rather than the full one
pub fn Tensor(comptime dtype: dtypes.DType, comptime shape: anytype) type {
    return tensor.InferredStrides(dtype, shape);
}

pub fn constant(comptime dtype: dtypes.DType, comptime value: anytype) Tensor(dtype, .{1}) {
    return tensor.constant(dtype, value);
}

pub fn range(comptime dtype: dtypes.DType, comptime start: dtype, comptime stop: dtype) Tensor(dtype, .{stop - start}) {
    return tensor.range(dtype, start, stop);
}

test "tesseract" {
    _ = Program;
}

pub fn init() void {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    Graph.init(std.heap.page_allocator);
    Program.init(std.heap.page_allocator);
}
