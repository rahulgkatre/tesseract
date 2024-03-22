const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;
const F = tesseract.F;

const Graph = @import("src/Graph.zig");

// Example of a softmax
fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.sub(max);
    const exp = x_minus_max.exp();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.div(sumexp);
    return sm;
}

pub fn main() !void {
    tesseract.init();
    defer tesseract.deinit();

    // All tensor code should must be in comptime
    // Here is a small neural network that can be used for MNIST
    const out = comptime blk: {
        const x = Tensor(.u8, .{ 16, 28, 28 }).input();
        const w1 = Tensor(.f16, .{ 28 * 28, 64 }).input();
        const w2 = Tensor(.f16, .{ 64, 32 }).input();
        const w3 = Tensor(.f16, .{ 32, 10 }).input();
        const norm_x = x.div(tesseract.scalar(.f16, 255));
        const l1_out = F.relu(norm_x.view(.{ 16, 28 * 28 }).matmul(w1));
        const l2_out = F.relu(l1_out.matmul(w2));
        const l3_out = F.relu(l2_out.matmul(w3));
        break :blk softmax(l3_out, l3_out.ndims - 1);
    };
    out.trace();

    std.debug.print("\n", .{});
    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try std.json.stringify(Graph{}, .{}, writer);
    std.debug.print("\n", .{});
}
