const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;

const Schedule = @import("src/Schedule.zig");

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
        const a = Tensor(.f32, .{ 2, 3 }).full(2);
        const b = Tensor(.f32, .{ 3, 16 }).input();
        // break :blk a.matmul(b);
        break :blk softmax(a.matmul(b), 1);
    };

    tesseract.init();
    defer tesseract.deinit();
    tesseract.trace(&out);
    tesseract.Fusion.greedyFusion();
    tesseract.viz(std.debug);
    std.debug.print("\n", .{});
    try Schedule.create();
    // try tesseract.code(@import("src/codegen/Zig.zig"), std.debug);
}
