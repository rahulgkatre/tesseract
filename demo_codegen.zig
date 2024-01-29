const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("src/tensor.zig");

const Tensor = tensor.Tensor;
const Backend = @import("src/backend.zig").Backend;

const TestBackend = &Backend{ .Codegen = .{} };

pub fn main() !void {
    // To take advantage of comptime features, all tensor code should be in comptime
    const out = comptime blk: {
        const x1 = Tensor(f32, .{ 2, 4, 1 }).full(TestBackend, 3);
        const x2 = Tensor(f32, .{ 1, 3 }).full(TestBackend, 2);
        break :blk x1.add(x2);
    };

    // Use comptime on the graph call to see the compute graph
    // comptime out.graph();

    // Print the tensors created during compile time, which now exist at runtime
    // as they have memory addresses

    // Initialize the backend which will allow for allocation of tensor storage
    TestBackend.runtime(.{ .filename = "demo_codegen_out.zig" });
    defer TestBackend.finished();

    // Print the storage to show the data
    tensor.debug = true;
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
