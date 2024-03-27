const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;
const F = tesseract.F;

const Graph = @import("src/Graph.zig");

pub fn main() !void {
    tesseract.init();
    defer tesseract.deinit();

    // All tensor code should must be in comptime
    // Here is a small neural network that can be used for MNIST
    const out = comptime blk: {
        const x = Tensor(.u8, .{ 16, 28, 28 }).input();
        const x_flatten = x.flattenPartial(-2, -1);
        const w1 = Tensor(.f16, .{ x_flatten.dimSize(-1), 64 }).input();
        const b1 = Tensor(.f16, .{64}).input();
        const w2 = Tensor(.f16, .{ w1.dimSize(-1), 32 }).input();
        const b2 = Tensor(.f16, .{32}).input();
        const w3 = Tensor(.f16, .{ w2.dimSize(-1), 10 }).input();
        const b3 = Tensor(.f16, .{10}).input();

        const norm_x = x_flatten.div(255);
        const l1_out = F.relu(norm_x.matmul(w1).add(b1));
        const l2_out = F.relu(l1_out.matmul(w2).add(b2));
        const l3_out = F.relu(l2_out.matmul(w3).add(b3));
        break :blk F.softmax(l3_out, -1);
    };
    out.trace();

    // @compileLog(@sizeOf(@TypeOf(out)), @sizeOf(@TypeOf(out.node())) - @sizeOf(@TypeOf(out.node().ptr)));
    // @compileLog(@sizeOf(@TypeOf(out.dtype)), @sizeOf(@TypeOf(out.node().dtype)));
    // @compileLog(@sizeOf(@TypeOf(out.ndims)), @sizeOf(@TypeOf(out.node().ndims)));
    // @compileLog(@sizeOf(@TypeOf(out.shape)), @sizeOf(@TypeOf(out.node().shape)));
    // @compileLog(@sizeOf(@TypeOf(out.strides)), @sizeOf(@TypeOf(out.node().strides)));
    // @compileLog(@sizeOf(@TypeOf(out.offset)), @sizeOf(@TypeOf(out.node().offset)));
    // @compileLog(@sizeOf(@TypeOf(out.op_node)), @sizeOf(@TypeOf(out.node().op_node)));

    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    try Graph.viz(writer);

    // std.debug.print("\n", .{});
    // try std.json.stringify(Graph{}, .{}, writer);
    // std.debug.print("\n", .{});
}
