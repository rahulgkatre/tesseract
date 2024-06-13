const std = @import("std");
const Allocator = std.mem.Allocator;

const tesseract = @import("tesseract.zig");
const Tensor = tesseract.Tensor;
const nn = tesseract.nn;
const debug = tesseract.debug;

// Here is a small neural network that can be used for MNIST
// All tensor code should run in comptime
// This can mean in the top level of a file or in a function that is called at comptime
// Here the model is defined by a Sequential module, just like in PyTorch
const x = Tensor([16][28][28]u8).input("mnist_batched_input")
    .flatten(.{ .from = -2, .to = -1 })
    .cast(.f16)
    .div(255.0);

const model = nn.Sequential("model", .{
    nn.LazyLinear(64, .f16, "input"){},
    nn.ReLU{},
    nn.LazyLinear(32, .f16, "fc1"){},
    nn.ReLU{},
    nn.LazyLinear(32, .f16, "fc2"){},
    nn.ReLU{},
    nn.LazyLinear(10, .f16, "output"){},
}){};

const out = model.forward(x).softmax(-1);

pub fn main() !void {
    // Try uncommenting this line to see how the shape of the output has already been determined at compile time!
    // Not shown here, but all the operations that produce this output are also stored as a DAG by the op trackers
    // @compileLog(@TypeOf(out));

    // Now the program is in runtime, as we have some writers and allocators
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    // Visualize the dataflow as a GraphViz, and also print the JSON representation of the program

    // try tesseract.debug.dataflowViz(.{out.toAny()}, debug.debug_writer, gpa.allocator());
    // try tesseract.debug.dataflowJson(.{out.toAny()}, debug.debug_writer, gpa.allocator());

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    var graph = try tesseract.graph.Graph.init(&arena);
    defer graph.deinit();

    try graph.traceForward(out);
}
