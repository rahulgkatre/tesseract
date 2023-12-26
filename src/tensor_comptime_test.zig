const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const Tensor = @import("tensor.zig");
const comptimePrint = std.fmt.comptimePrint;

// test "permute shape check" {
//     comptime {
//         var tensor1 = Tensor.tensor(i32, .{ 2, 3, 4 });
//         var tensor2 = Tensor.tensor1.permute([_]usize{ 0, 2, 1 });
//         try expect(@reduce(.And, tensor2.shape == [_]usize{ 2, 4, 3 }));
//     }
// }

test "zip operation shape check" {
    var out = comptime blk: {
        var tensor1 = Tensor.tensor(Tensor.Type.Int32, [_]usize{ 2, 1, 4 });
        var tensor2 = Tensor.tensor(Tensor.Type.Int32, [_]usize{ 3, 1 });
        var tensor3 = tensor1.zip(Tensor.ZipOp.Add, &tensor2);
        try expect(@reduce(.And, tensor3.shape == [_]usize{ 2, 3, 4 }));
        break :blk tensor3;
    };
    _ = out;
    // std.debug.print("\n{any}\n", .{out.history.?});
}

// test "zip reduce operation shape check" {
//     var out = comptime blk: {
//         const reduce_dim: usize = 1;
//         var tensor1 = Tensor.tensor(i32, [_]usize{ 2, 3, 4 });
//         var tensor2 = tensor1.mock_reduce_fn(Op.ReduceOp.Sum, reduce_dim);
//         try expect(@reduce(.And, tensor2.shape == [_]usize{ 2, 1, 4 }));
//         break :blk tensor2;
//     };
//     std.debug.print("\n{any}\n", .{out.history.?});
// }

test "reduce operation shape check" {
    var out = comptime blk: {
        const reduce_dim: usize = 1;
        var tensor1 = Tensor.tensor(Tensor.Type.Int32, [_]usize{ 2, 1, 4 });
        var tensor2 = Tensor.tensor(Tensor.Type.Int32, [_]usize{ 2, 3, 1 });
        var tensor3 = tensor1.zip(Tensor.ZipOp.Add, &tensor2).reduce(Tensor.ReduceOp.Sum, reduce_dim);
        try expect(@reduce(.And, tensor3.shape == [_]usize{ 2, 1, 4 }));
        var t2_1 = Tensor.getInput(tensor3.history.?.inputs[0]);
        var t1  = Tensor.getInput(t2_1.history.?.inputs[0]);
        var t2  = Tensor.getInput(t2_1.history.?.inputs[1]);
        break :blk .{t2_1, t1, t2};
    };
    std.debug.print("\n{any}\n", .{out});
}

test "lazy with realization" {
    var tensor1 = comptime Tensor.tensor(Tensor.Type.Int32, .{ 2, 3, 4 });
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    try tensor1.realize(null, allocator);
    try expect(tensor1.allocator != null);
}

test "extend shape" {
    const in_shape: @Vector(2, usize) = [_]usize{ 3, 1 };
    var out_shape: @Vector(4, usize) = Tensor.extendShape(2, in_shape, 4);
    const expected_out_shape: @Vector(4, usize) = [_]usize{ 1, 1, 3, 1 };
    try expect(@reduce(.And, expected_out_shape == out_shape));
}
