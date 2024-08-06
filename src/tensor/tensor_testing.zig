const std = @import("std");
const utils = @import("../utils.zig");
const tensor = @import("tensor.zig");

const Tensor = tensor.Tensor;
const TensorType = @import("tensor_typing.zig").TensorType;

test "view" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Based on example from https://pytorch.org/docs/stable/generated/torch.as_strided.html
    const tensor1 = TensorType(.i32, .{ 3, 3 }).full(
        0,
        allocator,
    );
    const tensor2 = tensor1.view(
        .{ 2, 2 },
        .{ 1, 2 },
        0,
        allocator,
    );

    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqual(false, tensor2.isContiguous());

    const test_indices = [_][2]u64{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 } };
    const expected_flat_indices1 = &[_]u64{ 0, 2, 1, 3 };
    for (expected_flat_indices1, test_indices) |expected_flat_i, test_i| {
        try std.testing.expectEqual(
            expected_flat_i,
            utils.ravelMultiIndex(
                @TypeOf(tensor2.*).ndims,
                tensor2.strides[0..@TypeOf(tensor2.*).ndims].*,
                tensor2.offset,
                test_i,
            ),
        );
    }

    const tensor3 = tensor1.view(
        .{ 2, 2 },
        .{ 1, 2 },
        1,
        allocator,
    );
    try std.testing.expectEqualSlices(u64, &[_]u64{ 2, 2 }, tensor2.shape[0..tensor2.ndims]);
    try std.testing.expectEqual(false, tensor2.isContiguous());

    const expected_flat_indices2 = &[_]u64{ 1, 3, 2, 4 };
    for (expected_flat_indices2, test_indices) |expected_flat_i, test_i| {
        try std.testing.expectEqual(expected_flat_i, utils.ravelMultiIndex(
            @TypeOf(tensor3.*).ndims,
            tensor3.strides[0..@TypeOf(tensor3.*).ndims].*,
            tensor3.offset,
            test_i,
        ));
    }
}

test "cast" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const tensor1 = TensorType(.bool, .{3}).full(true, allocator);
    try std.testing.expect(tensor1.dtype == .bool);
    const tensor2 = tensor1.cast(.i32, allocator);
    try std.testing.expect(tensor2.dtype == .i32);
    const tensor3 = tensor2.cast(.i8, allocator);
    try std.testing.expect(tensor3.dtype == .i8);
    const tensor4 = tensor3.cast(.f16, allocator);
    try std.testing.expect(tensor4.dtype == .f16);
    const tensor5 = tensor4.cast(.f32, allocator);
    try std.testing.expect(tensor5.dtype == .f32);
}

test "ArrayType" {
    // const Tensor1 = TensorType(.i32, .{ 2, 3, 4 });
    // try std.testing.expectEqual(Tensor1.ArrayType(), [2][3][4]i32);
}

test "pad" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    const t4d = Tensor([3][3][4][2]f32).empty(allocator);
    const p1d = comptime .{.{ 1, 1 }};
    const out1 = t4d.pad(p1d, .{ .constant = 0 }, allocator);
    try std.testing.expectEqualDeep(@TypeOf(out1.*).shape, .{ 3, 3, 4, 4 });

    const p2d = comptime .{ .{ 1, 1 }, .{ 2, 2 } };
    const out2 = t4d.pad(p2d, .{ .constant = 0 }, allocator);
    try std.testing.expectEqualDeep(@TypeOf(out2.*).shape, .{ 3, 3, 8, 4 });

    const p3d = comptime .{ .{ 0, 1 }, .{ 2, 1 }, .{ 3, 3 } };
    const out3 = t4d.pad(p3d, .{ .constant = 0 }, allocator);
    try std.testing.expectEqualDeep(@TypeOf(out3.*).shape, .{ 3, 9, 7, 3 });
}

// test "input" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     const tensor1 = Tensor([2][1][4]i32).input("tensor1", allocator);
//     const tensor1_1 = Tensor([2][1][4]i32).input("tensor1", allocator);
//     const tensor2 = Tensor([2][1][4]i32).input("tensor2", allocator);
//     try std.testing.expect(@intFromPtr(&tensor1) == @intFromPtr(&tensor1_1));
//     try std.testing.expect(@intFromPtr(&tensor1) != @intFromPtr(&tensor2));
// }

// test "named dims" {
//     const x = comptime Tensor([64][28][28]f32).empty().setDimNames(.{ "batch", "height", "width" });
//     std.debug.assert(comptime x.namedDim(.batch) == 0);
// }
