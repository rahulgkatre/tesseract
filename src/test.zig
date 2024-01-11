const std = @import("std");
const expectEqual = std.testing.expectEqual;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const backend = @import("backend.zig");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const utils = @import("utils.zig");

const TestBackend = &backend.Backend{ .Zig = .{} };
const comptime_graph = false;
const runtime_graph = true;
const eval_logging = true;
fn runEval(comptime test_name: anytype, comptime out: anytype) void {
    if (comptime_graph) {
        @compileLog(test_name);
        comptime out.graph();
    } else if (runtime_graph) {
        std.debug.print("\n", .{});
        out.graph();
    }

    const eval_out = out.eval();
    if (eval_logging) {
        std.debug.print("\n{any}\n", .{eval_out.storage.Zig.data.?});
    }
}

test "same tensors assignable" {
    // This test catches regressions caused by comptime slices with the same values not being
    // equal to teach other, which would cause this test to not compile
    const tensor1 = Tensor(i32, .{ 2, 3, 4 }).input(TestBackend);
    var tensor2 = Tensor(i32, .{ 2, 3, 4 }).input(TestBackend);
    tensor2 = tensor1;
}

test "permute" {
    comptime {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).input(TestBackend);
        const tensor2 = tensor1.permute(.{ 0, 2, 1 });
        try expectEqual([_]usize{ 2, 4, 3 }, tensor2.shape);
        try expectEqual([_]usize{ 12, 1, 4, 0 }, tensor2.strides);
    }
}
test "view" {
    comptime {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).input(TestBackend);
        const tensor2 = tensor1.view(.{ 12, 2 });
        const tensor3 = tensor2.view(.{24});

        try expectEqual([_]usize{ 12, 2 }, tensor2.shape);
        try expectEqual([_]usize{ 2, 1, 0 }, tensor2.strides);
        try expectEqual([_]usize{24}, tensor3.shape);
        try expectEqual([_]usize{ 1, 0 }, tensor3.strides);
    }
}
test "as strided" {
    // Based on example from https://pytorch.org/docs/stable/generated/torch.as_strided.html
    comptime {
        const tensor1 = Tensor(i32, .{ 3, 3 }).input(TestBackend);
        const tensor2 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 0 });

        try expectEqual([_]usize{ 2, 2 }, tensor2.shape);
        try expectEqual(false, tensor2.isContiguous());

        const test_indices = [_][2]usize{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
        const expected_flat_indices1 = [_]usize{ 0, 2, 1, 3 };
        for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
            try expectEqual(expected_flat_i, tensor2.flattenIndex(test_i));
        }

        const tensor3 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 1 });
        try expectEqual([_]usize{ 2, 2 }, tensor2.shape);
        try expectEqual(false, tensor2.isContiguous());

        const expected_flat_indices2 = [_]usize{ 1, 3, 2, 4 };
        for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
            try expectEqual(expected_flat_i, tensor3.flattenIndex(test_i));
        }
    }
}
test "map" {
    const tensor1 = comptime Tensor(i32, .{ 2, 3, 4 }).constant(TestBackend, 3);
    const tensor2 = comptime tensor1.neg();
    try expectEqual([_]usize{ 2, 3, 4 }, tensor2.shape);
    TestBackend.init(.{});

    runEval("map", tensor2);
}
test "zip" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend, 2);
        const tensor2 = Tensor(i32, .{ 3, 1 }).constant(TestBackend, 3);
        const tensor3 = tensor1.add(tensor2);
        try expectEqual([_]usize{ 2, 3, 4 }, tensor3.shape);
        break :blk tensor3;
    };
    TestBackend.init(.{});
    runEval("zip", out);
}
test "reduce" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).constant(TestBackend, 5);
        const tensor2 = tensor1.sum(1);
        try expectEqual([_]usize{ 2, 1, 4 }, tensor2.shape);
        break :blk tensor2;
    };
    TestBackend.init(.{});
    runEval("reduce", out);
}
test "zip reduce" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend, 2);
        const tensor2 = Tensor(i32, .{ 2, 3, 1 }).constant(TestBackend, 3);
        const tensor3 = tensor1.add(tensor2).sum(1);
        try expectEqual([_]usize{ 2, 1, 4 }, tensor3.shape);
        break :blk tensor3;
    };

    TestBackend.init(.{});

    runEval("zip reduce", out);
}

test "lazy with realization" {
    var NewBackend = backend.Backend{ .Zig = .{} };
    var tensor1 = Tensor(i32, .{ 2, 3, 4 }).constant(&NewBackend, 0);
    NewBackend.init(.{});
    defer NewBackend.deinit();

    tensor1.initStorage();
    try expectEqual(true, tensor1.storage.Zig.data != null);
    try expectEqual(true, tensor1.storage.Zig.data.?.len == 24);
    try expectEqual([_]i32{0} ** 24, tensor1.storage.Zig.data.?[0..24].*);
}

fn fn1() Tensor(i32, .{ 2, 1, 4 }) {
    const tensor1 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend, 1);
    const tensor2 = Tensor(i32, .{ 2, 3, 1 }).constant(TestBackend, 2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) Tensor(i32, .{ 2, 1, 4 }) {
    return comptime blk: {
        const tensor4 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend, 4);
        const tensor5 = Tensor(i32, .{ 2, 3, 1 }).constant(TestBackend, 5);
        const tensor6 = tensor4.mul(tensor5).sum(1).add(input);
        break :blk tensor6;
    };
}

test "tensors from functions" {
    const out = comptime blk: {
        const tensor3 = fn1();
        const tensor6 = fn2(tensor3);
        break :blk tensor6;
    };

    TestBackend.init(.{});

    runEval("tensors from functions", out);
    TestBackend.deinit();
}

fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.sub(max);
    const exp = x_minus_max.exp2();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.div(sumexp);
    return sm;
}

test "softmax" {
    const out = comptime blk: {
        const x = Tensor(f64, .{ 2, 16 }).constant(TestBackend, 5);
        break :blk softmax(x, 1);
    };

    TestBackend.init(.{});

    runEval("softmax", out);
}
