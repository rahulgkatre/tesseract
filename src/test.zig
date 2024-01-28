const std = @import("std");
const expectEqual = std.testing.expectEqual;
const Allocator = std.mem.Allocator;
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;
const backend = @import("backend.zig");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const utils = @import("utils.zig");

const TestBackend = &backend.Backend{ .Zig = .{} };
const show_graph = true;
const eval_logging = true;
fn runEval(comptime test_name: anytype, comptime out: anytype) void {
    _ = test_name;
    TestBackend.runtime(.{});
    if (show_graph) {
        std.debug.print("\nGRAPH\n", .{});
        out.graph();
    }
    const out_eval = out.eval();
    if (eval_logging) {
        std.debug.print("{any}\n", .{out_eval.storage.?.Zig.data});
    }
    TestBackend.finished();
}

test "same tensors assignable" {
    // This test catches regressions caused by comptime slices with the same values not being
    // equal to teach other, which would cause this test to not compile
    const tensor1 = Tensor(i32, .{ 2, 3, 4 }).full(TestBackend, 0);
    var tensor2 = Tensor(i32, .{ 2, 3, 4 }).full(TestBackend, 0);
    tensor2 = tensor1;
}

test "permute" {
    comptime {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).full(TestBackend, 0);
        const tensor2 = tensor1.permute(.{ 0, 2, 1 });
        try expectEqual([_]usize{ 2, 4, 3 }, tensor2.shape);
        try expectEqual([_]usize{ 12, 1, 4, 0 }, tensor2.strides);
    }
}
test "view" {
    comptime {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).full(TestBackend, 0);
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
        const tensor1 = Tensor(i32, .{ 3, 3 }).full(TestBackend, 0);
        const tensor2 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 0 });

        try expectEqual([_]usize{ 2, 2 }, tensor2.shape);
        try expectEqual(false, tensor2.isContiguous());

        const test_indices = [_][2]usize{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
        const expected_flat_indices1 = [_]usize{ 0, 2, 1, 3 };
        for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
            try expectEqual(expected_flat_i, tensor2.idxToPos(test_i));
        }

        const tensor3 = tensor1.asStrided(.{ 2, 2 }, .{ 1, 2, 1 });
        try expectEqual([_]usize{ 2, 2 }, tensor2.shape);
        try expectEqual(false, tensor2.isContiguous());

        const expected_flat_indices2 = [_]usize{ 1, 3, 2, 4 };
        for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
            try expectEqual(expected_flat_i, tensor3.idxToPos(test_i));
        }
    }
}
test "map" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).full(TestBackend, 3);
        const tensor2 = tensor1.neg();
        break :blk tensor2;
    };
    try expectEqual([_]usize{ 2, 3, 4 }, out.shape);
    runEval("map", out);
}
test "zip" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).full(TestBackend, 2);
        const tensor2 = Tensor(i32, .{ 3, 1 }).full(TestBackend, 3);
        const tensor3 = tensor1.add(tensor2);
        try expectEqual([_]usize{ 2, 3, 4 }, tensor3.shape);
        break :blk tensor3;
    };
    runEval("zip", out);
}
test "reduce" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 3, 4 }).full(TestBackend, 5);
        const tensor2 = tensor1.sum(1);
        try expectEqual([_]usize{ 2, 1, 4 }, tensor2.shape);
        break :blk tensor2;
    };
    runEval("reduce", out);
}
test "zip reduce" {
    const out = comptime blk: {
        const tensor1 = Tensor(i32, .{ 2, 1, 4 }).full(TestBackend, 2);
        const tensor2 = Tensor(i32, .{ 2, 3, 1 }).full(TestBackend, 3);
        const tensor3 = tensor1.add(tensor2).sum(1);
        try expectEqual([_]usize{ 2, 1, 4 }, tensor3.shape);
        break :blk tensor3;
    };
    runEval("zip reduce", out);
}

test "lazy with realization" {
    const NewBackend = &backend.Backend{ .Zig = .{} };
    const tensor1 = comptime Tensor(i32, .{ 2, 3, 4 }).full(NewBackend, 0);
    NewBackend.runtime(.{});
    defer NewBackend.finished();

    const runtime_tensor = tensor1.runtime(0);
    try expectEqual(true, runtime_tensor.storage != null);
    std.debug.print("{any}", .{runtime_tensor.storage.?.Zig.data.len});
    try expectEqual(true, runtime_tensor.storage.?.Zig.data.len == 24);
    try expectEqual([_]i32{0} ** 24, runtime_tensor.storage.?.Zig.data[0..24].*);
}

fn fn1() Tensor(i32, .{ 2, 1, 4 }) {
    const tensor1 = Tensor(i32, .{ 2, 1, 4 }).full(TestBackend, 1);
    const tensor2 = Tensor(i32, .{ 2, 3, 1 }).full(TestBackend, 2);
    const tensor3 = tensor1.add(tensor2).sum(1);
    return tensor3;
}

fn fn2(input: anytype) Tensor(i32, .{ 2, 1, 4 }) {
    return comptime blk: {
        const tensor4 = Tensor(i32, .{ 2, 1, 4 }).full(TestBackend, 4);
        const tensor5 = Tensor(i32, .{ 2, 3, 1 }).full(TestBackend, 5);
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

    runEval("tensors from functions", out);
}

fn softmax(x: anytype, comptime dim: u8) @TypeOf(x) {
    const max = x.max(null);
    const x_minus_max = x.sub(max);
    const exp = x_minus_max.exp();
    const sumexp = exp.sum(dim);
    const sm = x_minus_max.div(sumexp);
    return sm;
}

test "softmax" {
    const out = comptime blk: {
        const x = Tensor(f64, .{ 2, 16 }).full(TestBackend, 5);
        break :blk softmax(x, 1);
    };
    runEval("softmax", out);
}

test "as_type" {
    const out = comptime blk: {
        const tensor1 = Tensor(bool, .{3}).full(TestBackend, true);
        const tensor2 = tensor1.cast(i32);
        const tensor3 = tensor2.cast(u8);
        const tensor4 = tensor3.cast(f16);
        const tensor5 = tensor4.cast(f32);
        break :blk tensor5;
    };
    runEval("as_type", out);
}

// TODO
// Eventually these tests should be called through eval() with CodegenBackend
// but still output the same code

test "cast codegen" {
    const actual = comptime gen: {
        var t1 = tensor.range(TestBackend, f32, 4.0, 16.0);
        var t2 = t1.cast(i32);
        const cg = @import("codegen/ZigCodegen.zig"){};
        break :gen cg.cast(i32, @constCast(&t1), 1, @constCast(&t2), 0);
    };
    _ = actual;
    const expected =
        \\for (0..12) |i| {
        \\    t0[i] = @intFromFloat(t1[i]);  
        \\}
    ;
    _ = expected;
    // try expectEqual(expected[0..expected.len].*, actual[0..actual.len].*);
}

test "map codegen" {
    const actual = comptime gen: {
        var t1 = tensor.range(TestBackend, f32, 4.0, 16.0);
        var t2 = t1.neg();
        const cg = @import("codegen/ZigCodegen.zig"){};
        break :gen cg.map(.Neg, @constCast(&t1), 1, @constCast(&t2), 0);
    };
    _ = actual;
    const expected =
        \\for (0..12) |i| {
        \\    t0[i] = -(t1[i]);  
        \\}
    ;
    _ = expected;
    // try expectEqual(expected[0..expected.len].*, actual[0..actual.len].*);
}

test "zip no broadcast codegen" {
    const actual = comptime gen: {
        var t1 = tensor.range(TestBackend, f32, 4.0, 16.0);
        var t2 = tensor.range(TestBackend, f32, 4.0, 16.0);
        var t3 = t1.add(t2);
        const cg = @import("codegen/ZigCodegen.zig"){};
        break :gen cg.zip(.Add, @constCast(&t1), 1, @constCast(&t2), 2, @constCast(&t3), 0);
    };
    _ = actual;
    const expected =
        \\for (0..12) |i| {
        \\    t0[i] = (t1[i]) + (t2[i]);  
        \\}
    ;
    _ = expected;
    // try expectEqual(expected[0..expected.len].*, actual[0..actual.len].*);
}

test "reduce all codegen" {
    const actual = comptime gen: {
        var t1 = tensor.range(TestBackend, f32, 4.0, 16.0);
        var t2 = t1.sum(null);
        const cg = @import("codegen/ZigCodegen.zig"){};
        break :gen cg.reduce(.Sum, @constCast(&t1), 1, null, @constCast(&t2), 0);
    };
    _ = actual;
    const expected =
        \\{
        \\    var acc = t1[0];
        \\    for (1..12) |i| {
        \\        acc = (acc) + (t1[i]);
        \\    }
        \\    t0[0] = acc;
        \\}
    ;
    _ = expected;
    // try expectEqual(expected[0..expected.len].*, actual[0..actual.len].*);
}
