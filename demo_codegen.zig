const std = @import("std");
const Allocator = std.mem.Allocator;
const tensor = @import("src/tensor.zig");

const Tensor = tensor.Tensor;

fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.sub(max);
    const exp = x_minus_max.exp();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.div(sumexp);
    return sm;
}

const Graph = @import("src/Graph.zig");
pub fn main() !void {
    // Initialize the global graph
    Graph.init();
    defer Graph.deinit();

    // All tensor code should must be in comptime
    const out = comptime blk: {
        const x = Tensor(f32, .{ 2, 16 }).full(3);
        break :blk softmax(x, 1);
    };

    // Print the storage to show the data
    out.trace();
    Graph.show();
}
