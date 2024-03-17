const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;

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
    // All tensor code should must be in comptime
    // const out = comptime softmax(
    //     tesseract.Tensor(.f32, .{ 2, 16 }).full(3),
    //     1,
    // );

    const out = comptime blk: {
        const a = Tensor(.f32, .{ "B", "M", "K" })
            .full(2);
        const b = Tensor(.f32, .{ "B", "K", "N" }).input();
        // break :blk a.matmul(b);
        break :blk softmax(a.matmul(b), 1);
    };

    tesseract.init();
    defer tesseract.deinit();
    tesseract.trace(&out);
    std.debug.print("\n", .{});

    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try std.json.stringify(Graph{}, .{ .whitespace = .indent_2 }, writer);
    std.debug.print("\n", .{});
}
