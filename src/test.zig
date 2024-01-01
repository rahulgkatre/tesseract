const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const tensor = @import("tensor.zig").tensor;
const Tensor = @import("tensor.zig").Tensor;
const GraphTensor = @import("graph.zig").GraphTensor;

const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const utils = @import("utils.zig");

// TODO: Make it so that the non eval functions are always running in comptime

test "permute shape check" {
    const tensor1 = comptime tensor(i32, .{ 2, 3, 4 });
    const tensor2 = comptime tensor1.permute([_]u8{ 0, 2, 1 });
    try expect(@reduce(.And, @as(@Vector(3, usize), tensor2.shape) == [_]usize{ 2, 4, 3 }));
}

test "zip operation shape check" {
    const tensor1 = tensor(i32, .{ 2, 1, 4 });
    const tensor2 = tensor(i32, .{ 3, 1 });
    const tensor3 = tensor1.zip(ops.ZipOp.Add, tensor2);
    try expect(@reduce(.And, @as(@Vector(3, usize), tensor3.shape) == [_]usize{ 2, 3, 4 }));
    tensor3.graph_tensor.debug_graph();
}
test "reduce operation shape check" {
    const tensor1 = tensor(i32, .{ 2, 3, 4 });
    const tensor2 = tensor1.reduce(ops.ReduceOp.Sum, 1);
    try expect(@reduce(.And, @as(@Vector(3, usize), tensor2.shape) == [_]usize{ 2, 1, 4 }));
    tensor2.graph_tensor.debug_graph();
}
test "zip reduce operation shape check" {
    const tensor1 = tensor(i32, .{ 2, 1, 4 });
    const tensor2 = tensor(i32, .{ 2, 3, 1 });
    const tensor3 = tensor1.zip(ops.ZipOp.Add, tensor2).reduce(ops.ReduceOp.Sum, 1);
    try expect(@reduce(.And, @as(@Vector(3, usize), tensor3.shape) == [_]usize{ 2, 1, 4 }));
    tensor3.graph_tensor.debug_graph();
}
// TODO: Reactivate test once a Zig Backend has been started
// test "lazy with realization" {
//     var tensor1 = tensor(i32, .{ 2, 3, 4 });
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     try tensor1.realize(null, allocator);
//     try expect(tensor1.allocator != null);
//     try expect(tensor1.storage != null);
// }
test "extend shape" {
    const in_shape: @Vector(2, usize) = .{ 3, 1 };
    var out_shape: @Vector(4, usize) = utils.extendShape(2, in_shape, 4);
    const expected_out_shape: @Vector(4, usize) = .{ 1, 1, 3, 1 };
    try expect(@reduce(.And, expected_out_shape == out_shape));
}

fn fn1() Tensor(i32, 3, .{ 2, 1, 4 }, .{ 4, 4, 1 }) {
    const tensor1 = tensor(i32, .{ 2, 1, 4 });
    const tensor2 = tensor(i32, .{ 2, 3, 1 });
    const tensor3 = tensor1.zip(ops.ZipOp.Add, tensor2).reduce(ops.ReduceOp.Sum, 1);
    return tensor3;
}

fn fn2(comptime input: anytype) Tensor(i32, 3, .{ 2, 1, 4 }, .{ 4, 4, 1 }) {
    const tensor4 = tensor(i32, .{ 2, 1, 4 });
    const tensor5 = tensor(i32, .{ 2, 3, 1 });
    const tensor6 = tensor4.zip(ops.ZipOp.Mul, tensor5).reduce(ops.ReduceOp.Sum, 1).zip(ops.ZipOp.Add, input);
    return tensor6;
}

test "tensors with functions" {
    const tensor3 = comptime fn1();
    const tensor6 = comptime fn2(tensor3);
    tensor6.graph_tensor.debug_graph();
}
