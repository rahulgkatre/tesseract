const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("src/tesseract.zig");
const Tensor = tesseract.Tensor;

// Example of a softmax
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
    // const out = comptime softmax(
    //     tesseract.Tensor(.f32, .{ 2, 16 }).full(3),
    //     1,
    // );
    const out = comptime blk: {
        const a = Tensor(.f32, .{ 2, 3 }).full(2);
        const b = Tensor(.f32, .{ 3, 4 }).full(3);
        break :blk a.matmul(b);
    };

    // Call trace on the output to build its computation graph
    Graph.trace(out);
    Graph.applyGreedyFusion();
    // Show the graph
    Graph.viz();
}
