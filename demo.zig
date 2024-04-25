const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;

const Graph = @import("src/Graph.zig");

pub fn tmain() Tensor(.f16, .{ 16, 10 }) {
    // Here is a small neural network that can be used for MNIST
    const x = Tensor(.u8, .{ 16, 28, 28 }).input();
    const x_flatten = x.flattenPartial(-2, -1);
    const w1 = Tensor(.f16, .{ x_flatten.dimSize(-1), 64 }).param();
    const b1 = Tensor(.f16, .{64}).param();
    const w2 = Tensor(.f16, .{ w1.dimSize(-1), 32 }).param();
    const b2 = Tensor(.f16, .{32}).param();
    const w3 = Tensor(.f16, .{ w2.dimSize(-1), 10 }).param();
    const b3 = Tensor(.f16, .{10}).param();

    const norm_x = x_flatten.div(255).asType(.f16);
    const l1_out = norm_x.matmul(w1).add(b1).relu();
    const l2_out = l1_out.matmul(w2).add(b2).relu();
    const l3_out = l2_out.matmul(w3).add(b3).relu();
    return l3_out.softmax(-1);
}

pub fn main() !void {
    tesseract.init();
    defer tesseract.deinit();

    // All tensor code should run in comptime
    const out = comptime tmain();
    out.trace();

    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try Graph.viz(writer);

    // std.debug.print("\n", .{});
    // try std.json.stringify(Graph{}, .{}, writer);
    // std.debug.print("\n", .{});
}
