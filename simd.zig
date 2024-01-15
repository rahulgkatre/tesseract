const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("src/tensor.zig").Tensor;
const Backend = @import("src/backend.zig").Backend;

const TestBackend = &Backend{ .Zig = .{} };

pub fn main() !void {
    // To take advantage of comptime features, all tensor code should be in comptime
    const out = comptime blk: {
        const x1 = Tensor(i32, .{ 2, 64 }).constant(TestBackend, 2);
        const x2 = x1.sum(1);
        break :blk x2;
    };

    // Use comptime on the graph call to see the compute graph
    // comptime out.graph();

    // Print the tensors created during compile time, which now exist at runtime
    // as they have memory addresses
    out.graph();

    // Initialize the backend which will allow for allocation of tensor storage
    TestBackend.init(.{});
    defer TestBackend.deinit();

    // Print the storage to show the data
    const eval_out = out.eval();
    std.debug.print("\n{any}\n", .{eval_out.storage});

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
