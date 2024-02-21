pub const std = @import("std");
pub const dtypes = @import("dtypes.zig");

// Expose functions publicly when using Tesseract as a Zig library
const tensor = @import("tensor.zig");

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

test {
    @import("Graph.zig");
    @import("Program.zig");
    @import("tensor.zig");
}
