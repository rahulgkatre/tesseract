const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("src/tensor.zig").Tensor;
const Range = @import("src/tensor.zig").Range;

const Backend = @import("src/backend.zig").Backend;

const TestBackend = &Backend{ .Zig = .{} };

pub fn main() !void {
    // To take advantage of comptime features, all tensor code should be in comptime
    const out = comptime blk: {
        const x1 = Range(TestBackend, i32, 0, 64).view(.{ 8, 1, 4, 1, 2 }).view(.{ 8, 1, 4, 1, 2 });
        const x2 = Range(TestBackend, i32, 64, 128).view(.{ 8, 2, 1, 4, 1 }).view(.{ 8, 4, 1, 2, 1 });
        const x3 = x1.add(x2);
        break :blk x3;
    };

    // Use comptime on the graph call to see the compute graph
    // comptime out.graph();

    // Print the tensors created during compile time, which now exist at runtime
    // as they have memory addresses

    // Initialize the backend which will allow for allocation of tensor storage
    TestBackend.init(.{});
    defer TestBackend.deinit();

    out.graph();

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
