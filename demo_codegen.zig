const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("src/tensor.zig");

const Tensor = tensor.Tensor;
const Backend = @import("src/backend.zig").Backend;

fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.sub(max);
    const exp = x_minus_max.exp();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.div(sumexp);
    return sm;
}

pub fn main() !void {
    // To take advantage of comptime features, all tensor code should be in comptime
    const TestBackend = &Backend{ .Codegen = .{} };
    const out = comptime blk: {
        const x = Tensor(f32, .{ 2, 16 }).full(TestBackend, 3);
        break :blk softmax(x, 1);
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
