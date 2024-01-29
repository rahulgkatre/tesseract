const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("src/tensor.zig").Tensor;
const Backend = @import("src/backend.zig").Backend;

const TestBackend = &Backend{ .Codegen = .{} };

fn fn1() Tensor(i32, .{ 2, 1, 4 }) {
    const tensor1 = Tensor(i32, .{ 2, 1, 4 }).full(TestBackend, 1);
    const tensor2 = Tensor(i32, .{ 2, 3, 1 }).full(TestBackend, 2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) Tensor(i32, .{ 2, 1, 4 }) {
    return comptime blk: {
        const tensor4 = Tensor(i32, .{ 2, 1, 4 }).full(TestBackend, 3);
        const tensor5 = Tensor(i32, .{ 2, 3, 1 }).full(TestBackend, 4);
        const tensor6 = tensor4.mul(tensor5).sum(1).add(input);
        break :blk tensor6;
    };
}

pub fn main() !void {
    // To take advantage of comptime features, all tensor code should be in comptime
    const out = comptime blk: {
        const x1 = fn1();
        const x2 = fn2(x1);
        break :blk x2;
    };

    // Use comptime on the graph call to see the compute graph
    // comptime out.graph();

    // Print the tensors created during compile time, which now exist at runtime
    // as they have memory addresses

    // Initialize the backend which will allow for allocation of tensor storage
    TestBackend.runtime(.{ .filename = "demo_codegen_out.zig" });
    defer TestBackend.finished();

    // Print the storage to show the data
    _ = out.eval();

    // The data is the same as the following numpy code
    // >>> import numpy as np
    // >>> t1 = np.ones((2,1,4))
    // >>> t2 = 2 * np.ones((2,3,1))
    // >>> t3 = (t1 + t2).sum(1)
    // >>> t4 = 3 * t1
    // >>> t5 = 4 * np.ones((2,3,1))
    // >>> t6 = (t4 * t5).sum(1)+t3
    // >>> t6
    // array([[45., 45., 45., 45.],
    //     [45., 45., 45., 45.]])
    // >>>
}
