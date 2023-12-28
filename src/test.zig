const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const tensor = @import("tensor.zig").tensor;
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const utils = @import("utils.zig");


test "permute shape check" {
    comptime {
        const tensor1 = tensor(i32, .{ 2, 3, 4 });
        const tensor2 = tensor1.permute([_]u8{ 0, 2, 1 });
        try expect(@reduce(.And, @as(@Vector(3, usize), tensor2.shape) == [_]usize{ 2, 4, 3 }));
    }
}

test "zip operation shape check" {
    const out = comptime blk: {
        const tensor1 = tensor(i32, [_]usize{ 2, 1, 4 });
        const tensor2 = tensor(i32, [_]usize{ 3, 1 });
        const tensor3 = tensor1.zip(ops.ZipOp.Add, &tensor2);
        try expect(@reduce(.And, @as(@Vector(3, usize), tensor3.shape) == [_]usize{ 2, 3, 4 }));
        break :blk tensor3;
    };
    _ = out;
    // std.debug.print("\n{any}\n", .{out.graph_tensor.print_graph()});

}
test "reduce operation shape check" {
    const out = comptime blk: {
        const reduce_dim: usize = 1;
        const tensor1 = tensor(i32, [_]usize{ 2, 3, 4 });
        const tensor2 = tensor1.reduce(ops.ReduceOp.Sum, reduce_dim);
        try expect(@reduce(.And, @as(@Vector(3, usize), tensor2.shape) == [_]usize{ 2, 1, 4 }));
        break :blk tensor2;
    };
    _ = out;
    // std.debug.print("\n{any}\n", .{out.graph_tensor.print_graph()});
}
test "zip reduce operation shape check" {
    const out = comptime blk: {
        const reduce_dim: usize = 1;
        const tensor1 = tensor(i32, [_]usize{ 2, 1, 4 });
        const tensor2 = tensor(i32, [_]usize{ 2, 3, 1 });
        const tensor3 = tensor1.zip(ops.ZipOp.Add, &tensor2).reduce(ops.ReduceOp.Sum, reduce_dim);
        try expect(@reduce(.And, @as(@Vector(3, usize), tensor3.shape) == [_]usize{ 2, 1, 4 }));
        break :blk tensor3;
    };
    _ = out;
    // std.debug.print("\n{any}\n", .{out.graph_tensor.print_graph()});
}
// TODO: Test is currently broken because of comptime in GraphTensor
// Will need fixes to realization logic too
// test "lazy with realization" {
//     var tensor1 = comptime blk: {
//         break :blk tensor(i32, .{ 2, 3, 4 });
//     };
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     try tensor1.realize(null, allocator);
//     try expect(tensor1.allocator != null);
// }
test "extend shape" {
    const in_shape: @Vector(2, usize) = [_]usize{ 3, 1 };
    var out_shape: @Vector(4, usize) = utils.extendShape(2, in_shape, 4);
    const expected_out_shape: @Vector(4, usize) = [_]usize{ 1, 1, 3, 1 };
    try expect(@reduce(.And, expected_out_shape == out_shape));
}
