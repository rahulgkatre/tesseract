const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const tensor = @import("tensor.zig").tensor;
const comptimePrint = std.fmt.comptimePrint;

test "permute shape check" {
    comptime {
        const tensor1 = tensor(i32, .{ 2, 3, 4 });
        const tensor2 = tensor1.permute([_]usize{ 0, 2, 1 });
        try expect(@reduce(.And, tensor2.shape == [_]usize{ 2, 4, 3 }));
    }
}

test "zip operation shape check" {
    comptime {
        const tensor1 = tensor(i32, [_]usize{ 2, 1, 4 });
        const tensor2 = tensor(i32, [_]usize{ 2, 3, 1 });
        const tensor3 = tensor1.mock_zip_fn(null, tensor2);
        try expect(@reduce(.And, tensor3.shape == [_]usize{ 2, 3, 4 }));
    }
}

test "zip reduce operation shape check" {
    comptime {
        const tensor1 = tensor(i32, [_]usize{ 2, 3, 4 });
        const tensor3 = tensor1.mock_reduce_fn(null, 1);
        try expect(@reduce(.And, tensor3.shape == [_]usize{ 2, 1, 4 }));
    }
}

test "reduce operation shape check" {
    comptime {
        const tensor1 = tensor(i32, [_]usize{ 2, 1, 4 });
        const tensor2 = tensor(i32, [_]usize{ 2, 3, 1 });
        const tensor3 = tensor1.mock_zip_fn(null, tensor2).mock_reduce_fn(null, 1);
        try expect(@reduce(.And, tensor3.shape == [_]usize{ 2, 1, 4 }));
    }
}

test "lazy with realization" {
    var tensor1 = comptime tensor(i32, .{ 2, 3, 4 });
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    tensor1 = try tensor1.realize(null, &allocator);
    try expect(tensor1.allocator != null);
}
