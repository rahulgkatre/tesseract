const std = @import("std");
const Allocator = std.mem.Allocator;
const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;

// Here is a small neural network that can be used for MNIST
// All tensor code should run in comptime
// This can mean in the top level of a file or in a function that is called at comptime
// Here the model is defined by a Sequential module, just like in PyTorch
const x = Tensor(.u8, .{ 16, 28, 28 }).input()
    .flatten(.{ .start_dim = -2, .end_dim = -1 })
    .div(255.0)
    .cast(.f16);
const model = tesseract.nn.Sequential(.{
    tesseract.nn.Linear(x.dimsize(-1), 64, .f16){},
    tesseract.nn.ReLU{},
    tesseract.nn.Linear(64, 32, .f16){},
    tesseract.nn.ReLU{},
    tesseract.nn.Linear(32, 10, .f16){},
}){};

const out = model.forward(x).softmax(-1);

pub fn main() !void {
    // Try uncommenting this line to see how the shape of the output has already been determined at compile time!
    // Not shown here, but all the operations that produce this output are also stored as a DAG by the op trackers
    // @compileLog(@TypeOf(out));

    // Now the program is in runtime, as we have some writers and allocators
    const writer = std.io.Writer(std.fs.File, std.fs.File.WriteError, std.fs.File.write){ .context = std.io.getStdOut() };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    // Visualize the dataflow as a GraphViz, and also print the JSON representation of the program
    try tesseract.utils.dataflowViz(&[_]*const tesseract.AnyTensor{@ptrCast(&out)}, writer, gpa.allocator());
    // try tesseract.utils.dataflowJson(&[_]*const tesseract.AnyTensor{@ptrCast(&out)}, writer, gpa.allocator());

    // var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    // defer arena.deinit();
    // var graph = try tesseract.graph.Graph.init(&arena);
    // try graph.trace(@ptrCast(&out));
    // graph.inlineSingleConsumers();
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
