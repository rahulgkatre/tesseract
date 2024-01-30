const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("src/tensor.zig");

const Tensor = tensor.Tensor;
const Backend = @import("src/backend.zig").Backend;

pub fn main() !void {
    // To take advantage of comptime features, all tensor code should be in comptime
    const TestBackend = &Backend{ .Codegen = .{} };
    const out = comptime blk: {
        const x1 = Tensor(i32, .{ 2, 4, 1 }).full(TestBackend, 3);
        const x2 = Tensor(i32, .{ 1, 3 }).full(TestBackend, 2);
        break :blk x1.mul(x2).sum(1);
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
}
