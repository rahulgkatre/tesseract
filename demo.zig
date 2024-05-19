const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;

// Here is a small neural network that can be used for MNIST
// All tensor code should run in comptime
// This can mean in the top level of a file or in a function that is called at comptime
const x = Tensor(.u8, .{ 16, 28, 28 }).input();
const x_flatten = x.flatten(.{ .start_dim = -2, .end_dim = -1 });
const w1 = Tensor(.f16, .{ x_flatten.dimsize(-1), 64 }).param();
const b1 = Tensor(.f16, .{64}).param();
const w2 = Tensor(.f16, .{ w1.dimsize(-1), 32 }).param();
const b2 = Tensor(.f16, .{32}).param();
const w3 = Tensor(.f16, .{ w2.dimsize(-1), 10 }).param();
const b3 = Tensor(.f16, .{10}).param();

const norm_x = x_flatten.div(255.0).cast(.f16);
const l1_out = norm_x.matmul(w1).add(b1).sigmoid();
const l2_out = l1_out.matmul(w2).add(b2).relu();
const l3_out = l2_out.matmul(w3).add(b3).relu();
const out = l3_out.softmax(-1);

pub fn main() !void {
    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try tesseract.utils.dataflowViz(&[_]*const tesseract.AnyTensor{
        @ptrCast(&out),
    }, writer, gpa.allocator());
    try tesseract.utils.dataflowJson(&[_]*const tesseract.AnyTensor{@ptrCast(&out)}, writer, gpa.allocator());

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var graph = try tesseract.graph.Graph.init(&arena);
    try graph.trace(@ptrCast(&out));
    graph.inlineSingleConsumers();

    // for (graph.node_consumers.keys()) |k| {
    //     const consumers = try gpa.allocator().alloc(tesseract.AnyTensor.JsonFormat, graph.node_consumers.get(k).?.keys().len);
    //     defer gpa.allocator().free(consumers);

    //     for (graph.node_consumers.get(k).?.keys(), consumers) |any, *json| {
    //         json.* = any.toJsonFormat();
    //     }

    //     try std.json.stringify(.{
    //         .producer = k.toJsonFormat(),
    //         .consumers = consumers,
    //     }, .{}, writer);
    //     try writer.print("\n\n", .{});
    // }
}
