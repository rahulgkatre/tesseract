const std = @import("std");
const expectEqual = std.testing.expectEqual;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig").Tensor;
const backend = @import("backend/backend.zig");
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
        _ = comptime out.eval();
    } else {
        std.debug.print("\n{any}\n", .{out.eval().storage.?.Zig.data});
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
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    @constCast(TestBackend).Zig = .{ .allocator = &allocator };
    _ = runEval("map", tensor2);
}
test "zip" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend, 2);
        const tensor2 = Tensor(i32, .{ 3, 1 }).constant(TestBackend, 3);
        const tensor3 = tensor1.add(tensor2);
        try expectEqual([_]usize{ 2, 3, 4 }, tensor3.shape);
        break :blk tensor3;
    };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    @constCast(TestBackend).Zig = .{ .allocator = &allocator };
    runEval("zip", out);
}
test "reduce" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).constant(TestBackend, 5);
        const tensor2 = tensor1.sum(1);
        try expectEqual([_]usize{ 2, 1, 4 }, tensor2.shape);
        break :blk tensor2;
    };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    @constCast(TestBackend).Zig = .{ .allocator = &allocator };
    runEval("reduce", out);
}
test "zip reduce" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend, 2);
        const tensor2 = Tensor(i32, .{ 2, 3, 1 }).constant(TestBackend, 3);
        const tensor3 = tensor1
            .add(tensor2)
            .sum(1);
        try expectEqual([_]usize{ 2, 1, 4 }, tensor3.shape);
        break :blk tensor3;
    };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    @constCast(TestBackend).Zig = .{ .allocator = &allocator };
    runEval("zip reduce", out);
}

test "lazy with realization" {
    var NewBackend = backend.Backend{ .Zig = .{ .allocator = null } };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var tensor1 = Tensor(i32, .{ 2, 3, 4 }).constant(&NewBackend, 0);
    NewBackend.init(.{ .allocator = &allocator });

    tensor1.make();
    try expectEqual(true, tensor1.storage != null);
    try expectEqual(true, tensor1.storage.?.Zig.data.len == 24);

    if (logging) {
        std.debug.print("\n{any}\n", .{tensor1.storage.?.Zig.data});
    }
}

fn fn1() Tensor(i32, .{ 2, 1, 4 }) {
    const tensor1 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend, 2);
    const tensor2 = Tensor(i32, .{ 2, 3, 1 }).constant(TestBackend, 3);
    const tensor3 = tensor1
        .add(tensor2)
        .sum(1);
    return tensor3;
}

fn fn2(input: anytype) Tensor(i32, .{ 2, 1, 4 }) {
    const tensor4 = Tensor(i32, .{ 2, 1, 4 }).constant(TestBackend, 2);
    const tensor5 = Tensor(i32, .{ 2, 3, 1 }).constant(TestBackend, 3);
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
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    @constCast(TestBackend).Zig = .{ .allocator = &allocator };
    runEval("tensors from functions", out);
}

fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.add(max.neg());
    const exp = x_minus_max.exp2();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.mul(sumexp.recip());
    return sm;
}

test "softmax" {
    const out = comptime blk: {
        const x = Tensor(f64, .{ 2, 16 }).constant(TestBackend, 5);
        break :blk softmax(x, 1);
    };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    @constCast(TestBackend).Zig = .{ .allocator = &allocator };
    runEval("softmax", out);
}
