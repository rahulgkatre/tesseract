const std = @import("std");
const expectEqual = std.testing.expectEqual;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const backend = @import("backend.zig");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const utils = @import("utils.zig");

const TestBackend = &backend.Backend{ .Zig = .{ .allocator = null } };
const logging = true;
const compile_log = false;
fn runEval(comptime test_name: anytype, comptime out: anytype) void {
    if (!logging) {
        return;
    }
    if (compile_log) {
        @compileLog(test_name);
        comptime out.eval();
    } else {
        out.eval();
        std.debug.print("\n", .{});
    }
}

test "same tensors assignable" {
    const tensor1 = Tensor(i32, .{ 2, 3, 4 }).input(TestBackend);
    var tensor2 = Tensor(i32, .{ 2, 3, 4 }).input(TestBackend);
    tensor2 = tensor1;
}

test "permute" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).input(TestBackend);
        const tensor2 = tensor1.permute([_]u8{ 0, 2, 1 });
        break :blk tensor2;
    };
    try expectEqual([_]usize{ 2, 4, 3 }, out.shape);
    try expectEqual([_]usize{ 12, 1, 4 }, out.strides);
}
test "zip" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).input(TestBackend);
        const tensor2 = Tensor(i32, .{ 3, 1 }).input(TestBackend);
        const tensor3 = tensor1.add(tensor2);
        break :blk tensor3;
    };
    try expectEqual([_]usize{ 2, 3, 4 }, out.shape);
    runEval("zip", out);
}
test "reduce" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).input(TestBackend);
        const tensor2 = tensor1.sum(1);
        break :blk tensor2;
    };
    try expectEqual([_]usize{ 2, 1, 4 }, out.shape);
    runEval("reduce", out);
}
test "zip reduce" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).input(TestBackend);
        const tensor2 = Tensor(i32, .{ 2, 3, 1 }).input(TestBackend);
        const tensor3 = tensor1
            .add(tensor2)
            .sum(1);
        break :blk tensor3;
    };
    try expectEqual([_]usize{ 2, 1, 4 }, out.shape);
    runEval("zip reduce", out);
}

test "lazy with realization" {
    var NewBackend = backend.Backend{ .Zig = .{ .allocator = null } };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var tensor1 = Tensor(i32, .{ 2, 3, 4 }).input(&NewBackend);
    NewBackend.init(.{ .allocator = &allocator });

    try tensor1.empty();
    try expectEqual(true, tensor1.storage != null);
    try expectEqual(true, tensor1.storage.?.Zig.data.len == 24);

    if (logging) {
        std.debug.print("\n{any}\n", .{tensor1.storage.?});
    }
}

fn fn1() Tensor(i32, .{ 2, 1, 4 }) {
    const tensor1 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend);
    const tensor2 = Tensor(i32, .{ 2, 3, 1 }).constant(TestBackend);
    const tensor3 = tensor1
        .add(tensor2)
        .sum(1);
    return tensor3;
}

fn fn2(input: anytype) Tensor(i32, .{ 2, 1, 4 }) {
    const tensor4 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend);
    const tensor5 = Tensor(i32, .{ 2, 3, 1 }).constant(TestBackend);
    const tensor6 = tensor4
        .mul(tensor5)
        .sum(1)
        .add(input);
    return tensor6;
}

test "tensors from functions" {
    const out = comptime blk: {
        const tensor3 = fn1();
        const tensor6 = fn2(tensor3);
        break :blk tensor6;
    };
    runEval("tensors from functions", out);
}

fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.add(max.neg());
    const exp = x_minus_max.exp2();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.div(sumexp);
    return sm;
}

test "softmax" {
    const out = comptime blk: {
        const x = Tensor(f16, .{ 2, 16 }).input(TestBackend);
        break :blk softmax(x, 1);
    };
    runEval("softmax", out);
}
