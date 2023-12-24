const std = @import("std");
const Allocator = std.mem.Allocator;
const zeros = @import("tensor.zig").zeros;
const lazy = @import("tensor.zig").lazy;
const comptimePrint = std.fmt.comptimePrint;

test "test_permute_compile" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tensor1 = try zeros(i32, [_]usize{ 2, 3, 4 }, &allocator);
    defer tensor1.deinit();
    const tensor2 = try tensor1.permute([_]usize{ 0, 2, 1 });
    defer tensor2.deinit();
    std.debug.print("\ntensor1: {any}\n", .{tensor1.shape});
    std.debug.print("tensor1 is contiguous?: {any}", .{tensor1.is_contiguous()});
    std.debug.print("\ntensor2: {any}\n", .{tensor2.shape});
    std.debug.print("tensor2 is contiguous?: {any}\n\n", .{tensor2.is_contiguous()});
    // Tensor 2 does not own the storage (that belongs to Tensor 1) so when it is freed it will only free its own struct
    // When Tensor 1 is freed, the storage will be freed.
    // The permutation also makes tensor2 no longer contiguous
}

test "test_zip_fn_compile" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // Try changing the shape of the tensors. If the shapes don't broadcast together, the code won't compile.
    // This code simulates calling a zip function on the two tensors (e.g. add, mul, etc)
    // The operation is considered valid iff the tensors broadcast together
    var tensor1 = try zeros(i32, [_]usize{ 2, 1, 4 }, &allocator);
    defer tensor1.deinit();
    var tensor2 = try zeros(i32, [_]usize{ 2, 3, 1 }, &allocator);
    defer tensor2.deinit();
    const tensor3 = try tensor1.mock_zip_fn(null, tensor2);

    // const broadcast_shape = tensor1.shape_broadcast(tensor2);
    std.debug.print("\n{any}.zip_fn({any}) = {any}\n\n", .{ tensor1.shape, tensor2.shape, tensor3.shape });
}

test "test_reduce_fn_compile" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tensor1 = try zeros(i32, [_]usize{ 2, 3, 4 }, &allocator);
    defer tensor1.deinit();
    // Reduce along axis 1. If reduce dim is outside the number of dimensions the tensor has then the code won't compile.
    const tensor2 = try tensor1.mock_reduce_fn(null, 1);
    std.debug.print("\n{any}\n.reduce_fn({any}) = \n{any}\n\n", .{ tensor1, 1, tensor2 });
}

test "test_lazy" {
    var tensor1 = comptime lazy(i32, .{ 2, 3, 4 });
    std.debug.print("\n{any}\n", .{tensor1});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    _ = try tensor1.realize(null, &allocator);
    std.debug.print("\n{any}\n", .{tensor1});
}
