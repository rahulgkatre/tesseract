const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;
const ComptimeUtils = @import("tensor_comptime_utils.zig");
const TensorStorage = @import("tensor_storage.zig").TensorStorage;

pub fn tensor(comptime dtype: type, comptime shape: anytype, allocator: *const Allocator) !ret: {
    break :ret *Tensor(dtype, shape.len, shape, ComptimeUtils.defaultStrides(shape.len, shape));
} {
    return try Tensor(dtype, shape.len, shape, ComptimeUtils.defaultStrides(shape.len, shape)).init(null, allocator);
}

pub fn full(comptime dtype: type, comptime shape: anytype, val: dtype, allocator: *const Allocator) !ret: {
    const strides = ComptimeUtils.defaultStrides(shape.len, shape);
    break :ret *Tensor(dtype, shape.len, shape, strides);
} {
    const t = try tensor(dtype, shape, allocator);
    @memset(t.storage.data, val);
    return t;
}

pub fn zeros(comptime dtype: type, comptime shape: anytype, allocator: *const Allocator) !*Tensor(dtype, shape.len, shape, ComptimeUtils.defaultStrides(shape.len, shape)) {
    return full(dtype, shape, 0, allocator);
}

// TODO: Add a device type param here
// Should be easy to add this type param everywhere as the device will remain the same unless a to(x)
// method is called
fn Tensor(comptime dtype: type, comptime ndims: u8, comptime shape: [ndims]usize, comptime strides: [ndims]usize) type {
    switch (@typeInfo(dtype)) {
        .Bool => {},
        .ComptimeInt => {},
        .Int => {},
        .ComptimeFloat => {},
        .Float => {},
        else => @compileError("Non-numeric or non-bool tensor dtype not supported, received " ++ @typeName(dtype)),
    }
    return struct {
        const Self = @This();
        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        strides: [ndims]usize = strides,
        storage: *TensorStorage(dtype, size()),
        // id: usize = ComptimeUtils.TensorIdGenerator.getAndIncrement(),
        owns_storage: bool,
        allocator: *const Allocator,
        // Used to determine the size of the underlying storage
        pub fn size() usize {
            return comptime get_size: {
                if (ndims > 0) {
                    var _size: usize = 1;
                    for (0..ndims) |dim| {
                        _size *= shape[dim];
                    }
                    break :get_size _size;
                } else {
                    @compileError("Illegal tensor size of 0");
                }
            };
        }
        // Determine the index in the current tensor given an index in the broadcasted tensor
        // If the current tensor has size of 1 in a dimension, then the index must be 0
        // Otherwise it will be what the broadcasted index is
        fn broadcast_index(bc_ndims: u8, bc_index: [bc_ndims]usize) [ndims]usize {
            return comptime get_index: {
                const index: [ndims]usize = undefined;
                for (0..ndims) |i| {
                    index[bc_ndims - i - 1] = if (shape[ndims - i - 1] == 1) 0 else bc_index[bc_ndims - i - 1];
                }
                break :get_index index;
            };
        }
        // Since strides are known at compile time the compiler can simplify this to a function body of "return true" or "return false"
        pub fn is_contiguous(self: *Self) bool {
            _ = self;
            return comptime ComptimeUtils.isContiguous(ndims, strides);
        }
        pub fn init(storage: ?*TensorStorage(dtype, size()), allocator: *const Allocator) !*Self {
            const t = try allocator.create(Self);
            errdefer allocator.destroy(t);
            t.* = .{
                .storage = storage orelse try TensorStorage(dtype, size()).init(allocator),
                .owns_storage = storage == null,
                .allocator = allocator,
            };
            return t;
        }
        pub fn deinit(self: *Self) void {
            if (self.owns_storage) {
                self.storage.deinit();
            }
        }
        pub fn permute(self: *Self, comptime perm: [ndims]usize) !ret: {
            const permute_shape = ComptimeUtils.permuteArray(ndims, shape, perm);
            const permute_strides = ComptimeUtils.permuteArray(ndims, strides, perm);
            break :ret *Tensor(dtype, ndims, permute_shape, permute_strides);
        } {
            const new_shape = comptime ComptimeUtils.permuteArray(ndims, shape, perm);
            const new_strides = comptime ComptimeUtils.permuteArray(ndims, strides, perm);
            return try Tensor(dtype, ndims, new_shape, new_strides).init(self.storage, self.allocator);
        }

        // Mock implementations of three kinds of tensor ops
        // Map is 1 to 1 (neg, log, exp)
        pub fn mock_map_fn(self: *Self, map_op: anytype) !*Self {
            _ = self;
            _ = map_op;
            return try tensor(dtype, ndims, shape);
        }
        // Zip is 2 to 1 (add, mul, xor)
        pub fn mock_zip_fn(self: *Self, zip_op: anytype, other_tensor_ptr: anytype) !ret: {
            const out_shape = ComptimeUtils.shapeBroadcast(Self, @TypeOf(other_tensor_ptr.*));
            const out_ndims = out_shape.len;
            const out_strides = ComptimeUtils.defaultStrides(out_ndims, out_shape);
            break :ret *Tensor(dtype, out_ndims, out_shape, out_strides);
        } {
            _ = zip_op;
            const out_shape = comptime ComptimeUtils.shapeBroadcast(Self, @TypeOf(other_tensor_ptr.*));
            return try tensor(dtype, out_shape, self.allocator);
        }
        // Reduce is many to 1 and collapses a dimension to 1 (sum, prod, max, etc.)
        pub fn mock_reduce_fn(self: *Self, reduce_op: anytype, comptime reduce_dim: usize) !ret: {
            const out_shape = ComptimeUtils.getReducedShape(ndims, shape, reduce_dim);
            const out_strides = ComptimeUtils.defaultStrides(ndims, out_shape);
            break :ret *Tensor(dtype, ndims, out_shape, out_strides);
        } {
            _ = reduce_op;
            const out_shape = comptime ComptimeUtils.getReducedShape(ndims, shape, reduce_dim);
            return try tensor(dtype, out_shape, self.allocator);
        }
    };
}

test "test_broadcast_compile" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // Try changing the shape of the tensors. If the shapes don't broadcast together, the code won't compile.
    var tensor1 = try tensor(i32, [_]usize{ 1, 4 }, &allocator);
    defer tensor1.deinit();
    var tensor2 = try zeros(i32, [_]usize{ 3, 1 }, &allocator);
    defer tensor2.deinit();
    const broadcast_shape = comptime ComptimeUtils.shapeBroadcast(@TypeOf(tensor1.*), @TypeOf(tensor2.*));
    // const broadcast_shape = tensor1.shapeBroadcast(tensor2);
    std.debug.print("\n{any} and {any} broadcast to {any}\n", .{ tensor1.shape, tensor2.shape, broadcast_shape });
}

// test "test_nonnumeric_compile" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     const some_type = struct {
//         a: bool,
//     };
//     _ = some_type;
//     const tensor1 = try zeros(i8, .{ 1, 4 }, &allocator);
//     _ = tensor1;
// }
