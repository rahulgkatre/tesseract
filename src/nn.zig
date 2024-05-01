const std = @import("std");
const tensor = @import("tensor.zig");
const anytensor = @import("anytensor.zig").anytensor;
const dtypes = @import("dtypes.zig");

pub fn Linear(comptime in: u64, comptime out: u64, comptime dtype: dtypes.DType) type {
    return struct {
        const W = tensor.tensor(dtype, .{ in, out });
        const B = tensor.tensor(dtype, .{out});
        weight: W = W.param(),
        bias: B = B.param(),

        pub fn forward(comptime self: @This(), comptime x: anytype) @TypeOf(x).MatMul(W) {
            std.debug.assert(tensor.isTensor(@TypeOf(x)));
            return x.matmul(self.weight).add(self.bias);
        }
    };
}

pub fn LazyLinear(comptime out: u64) type {
    return struct {
        pub fn init(_: @This(), comptime x: anytype) Linear(x.shape[x.ndims - 1], out, x.dtype) {
            return .{};
        }

        fn W(comptime x: anytype) type {
            const in = x.shape[x.ndims - 1];
            const dtype = x.dtype;
            return tensor.tensor(dtype, .{ in, out });
        }

        pub fn forward(_: @This(), comptime x: anytype) @TypeOf(x).MatMul(W(x)) {
            const weight = W(x).param();
            const bias = tensor.tensor(x.dtype, .{out}).param();
            return x.matmul(weight).add(bias);
        }
    };
}

test "lazy linear" {
    const x = comptime tensor.tensor(.f32, .{ 16, 784 }).input();
    const linear = comptime LazyLinear(256){};
    const y = comptime linear.init(x).forward(x);
    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try @import("utils.zig").dataflowViz(&[_]*const anytensor{&y.widen()}, writer, std.testing.allocator);
}

test "linear" {
    const x = comptime tensor.tensor(.f32, .{ 16, 784 }).input();
    const linear = comptime Linear(784, 256, .f32){};
    const y = comptime linear.forward(x);
    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try @import("utils.zig").dataflowViz(&[_]*const anytensor{&y.widen()}, writer, std.testing.allocator);
}

test "symbolic using enum literal?" {
    // @compileLog(@typeInfo(@TypeOf(.batch_size)));
    const x: @TypeOf(.enum_literal) = .batch_size;
    try std.testing.expect(x != .n_channels);
    try std.testing.expect(x == .batch_size);

    const y = .batch_size;
    try std.testing.expect(x == y);
}
