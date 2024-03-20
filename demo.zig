const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;

const Graph = @import("src/Graph.zig");

fn sigmoid(x: anytype) @TypeOf(x) {
    const x_pos = x.neg().exp().add(tesseract.Scalar(x.dtype).full(1)).recip();
    const x_neg = x.exp().div(x.exp().add(tesseract.Scalar(x.dtype).full(1)));
    const mask = x.lessThan(tesseract.Scalar(x.dtype).full(0));
    return mask.where(x_neg, x_pos);
}

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
    const out = comptime blk: {
        const a = Tensor(.f32, .{ 2, 3, 4 }).full(2);
        const b = Tensor(.f32, .{ 2, 4, 3 }).input();
        // break :blk a.matmul(b);
        var ab = a.matmul(b);
        ab = sigmoid(ab);
        break :blk ab;
    };

    out.trace();

    std.debug.print("\n", .{});
    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try std.json.stringify(Graph{}, .{ .whitespace = .indent_2 }, writer);
    std.debug.print("\n", .{});
}
