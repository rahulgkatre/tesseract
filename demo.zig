const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;

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
const out = l3_out.softmax(-1);

pub fn main() !void {
    // All tensor code should run in comptime
    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    try tesseract.utils.viz(&[_]*const tesseract.anytensor{@ptrCast(&out)}, writer, gpa.allocator());
}
