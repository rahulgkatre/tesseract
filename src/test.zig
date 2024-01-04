const std = @import("std");
const expectEqual = std.testing.expectEqual;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const backend = @import("backend.zig");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const utils = @import("utils.zig");

const TestBackend = &backend.Backend{ .Zig = .{ .allocator = null } };
const TEST_COMPTIME = false;

test "same tensors assignable" {
    const tensor1 = Tensor(i32, .{ 2, 3, 4 }).init(TestBackend);
    var tensor2 = Tensor(i32, .{ 2, 3, 4 }).init(TestBackend);
    tensor2 = tensor1;
}

test "permute shape check" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).init(TestBackend);
        const tensor2 = tensor1.permute([_]u8{ 0, 2, 1 });
        break :blk tensor2;
    };
    try expectEqual([_]usize{ 2, 4, 3 }, out.shape);
    try expectEqual([_]usize{ 12, 1, 4 }, out.strides);
}
test "zip operation shape check" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).init(TestBackend);
        const tensor2 = Tensor(i32, .{ 3, 1 }).init(TestBackend);
        const tensor3 = tensor1.zip(ops.ZipOp.Add, tensor2);
        break :blk tensor3;
    };
    try expectEqual([_]usize{ 2, 3, 4 }, out.shape);
    if (TEST_COMPTIME) {
        comptime out.eval();
    } else {
        out.eval();
    }
    std.debug.print("\n", .{});
}
test "reduce operation shape check" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).init(TestBackend);
        const tensor2 = tensor1.reduce(ops.ReduceOp.Sum, 1);
        break :blk tensor2;
    };
    try expectEqual([_]usize{ 2, 1, 4 }, out.shape);
    if (TEST_COMPTIME) {
        comptime out.eval();
    } else {
        out.eval();
    }
    std.debug.print("\n", .{});
}
test "zip reduce operation shape check" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).init(TestBackend);
        const tensor2 = Tensor(i32, .{ 2, 3, 1 }).init(TestBackend);
        const tensor3 = tensor1
            .zip(ops.ZipOp.Add, tensor2)
            .reduce(ops.ReduceOp.Sum, 1);
        break :blk tensor3;
    };
    try expectEqual([_]usize{ 2, 1, 4 }, out.shape);
    if (TEST_COMPTIME) {
        comptime out.eval();
    } else {
        out.eval();
    }
    std.debug.print("\n", .{});
}
// TODO: Reactivate test once a Zig Backend has been started
// test "lazy with realization" {
//     var tensor1 = Tensor(i32, .{ 2, 3, 4 });
// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
// const allocator = gpa.allocator();
//     try tensor1.realize(null, allocator);
//     try expectEqual(tensor1.allocator != null);
//     try expectEqual(tensor1.storage != null);
// }

fn fn1() Tensor(i32, .{ 2, 1, 4 }) {
    const tensor1 = Tensor(i32, .{ 2, 1, 4 }).init(TestBackend);
    const tensor2 = Tensor(i32, .{ 2, 3, 1 }).init(TestBackend);
    const tensor3 = tensor1
        .zip(ops.ZipOp.Add, tensor2)
        .reduce(ops.ReduceOp.Sum, 1);
    return tensor3;
}

fn fn2(input: anytype) Tensor(i32, .{ 2, 1, 4 }) {
    const tensor4 = Tensor(i32, .{ 2, 1, 4 }).init(TestBackend);
    const tensor5 = Tensor(i32, .{ 2, 3, 1 }).init(TestBackend);
    const tensor6 = tensor4
        .zip(ops.ZipOp.Mul, tensor5)
        .reduce(ops.ReduceOp.Sum, 1)
        .zip(ops.ZipOp.Add, input);
    return tensor6;
}

test "tensors from functions" {
    const out = comptime blk: {
        const tensor3 = fn1();
        const tensor6 = fn2(tensor3);
        break :blk tensor6;
    };
    if (TEST_COMPTIME) {
        comptime out.eval();
    } else {
        out.eval();
    }
    std.debug.print("\n", .{});
}
