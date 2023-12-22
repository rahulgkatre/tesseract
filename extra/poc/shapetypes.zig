const std = @import("std");
const Allocator = std.mem.Allocator;

// A proof of concept showing that it is possible to make tensor shape a generic
// type parameter to enable compile time verification of broadcasts

fn TensorStorage(comptime dtype: type, comptime size: usize) type {
    return struct {
        const Self = @This();
        data: []dtype,
        // This allocator is a placeholder for a Device struct that provides an allocator object
        // so that the tensor storage can be allocated on device memory
        allocator: *const Allocator,

        pub fn init(allocator: *const Allocator) !*Self {
            const storage = try allocator.create(Self);
            storage.* = .{
                .data = try allocator.alloc(dtype, size),
                .allocator = allocator,
            };
            @memset(storage.data, 0);
            return storage;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }
    };
}

pub fn Tensor(comptime dtype: type, comptime shape: anytype) type {
    switch (@typeInfo(dtype)) {
        .Bool => {},
        .ComptimeInt => {},
        .Int => {},
        .ComptimeFloat => {},
        .Float => {},
        else => @compileError("Non-numeric or non-bool tensor dtype not supported, received " ++ @typeName(dtype)),
    }

    switch (@typeInfo(@TypeOf(shape))) {
        .Array => {},
        else => @compileError("Shape parameter must be an array with fixed size, received " ++ @typeName(@TypeOf(shape))),
    }

    return struct {
        const Self = @This();
        const size = shape_to_size(shape.len, shape);
        ndims: u8 = shape.len,
        shape: [shape.len]usize = shape,
        storage: *TensorStorage(dtype, size),
        owns_storage: bool,
        allocator: *const Allocator,

        fn shape_to_size(comptime ndims: u8, _shape: [ndims]usize) usize {
            if (ndims > 0) {
                var _size: usize = 1;
                inline for (0..ndims) |dim| {
                    _size *= _shape[dim];
                }
                return _size;
            } else {
                return 0;
            }
        }

        // NOTE: These two functions are a bit unsafe due to the usage of reflection to access fields of a tensor at compile time
        // It makes 2 critical assumptions about the input object:
        // - ndims: u8 is the first field of the struct
        // - shape: [ndims]usize is the second field of the struct
        // As long as these assumptions are valid there should not be any problems as if they are not valid then the code will still fail to compile
        // It would be better if a compile error could be generated to explain that the code was attempting to operate on something that is not a tensor
        fn comptime_get_tensor_ndims(comptime tensor: anytype) u8 {
            const info = @typeInfo(tensor);
            const ndims_field = info.Struct.fields[0];
            const ndims_default_value_aligned: *align(ndims_field.alignment) const anyopaque = @alignCast(@ptrCast(ndims_field.default_value));
            return @as(*const ndims_field.type, @ptrCast(ndims_default_value_aligned)).*;
        }

        fn comptime_get_tensor_shape(comptime ndims: u8, comptime tensor: anytype) [ndims]usize {
            const info = @typeInfo(tensor);
            const shape_field = info.Struct.fields[1];
            const shape_default_value_aligned: *align(shape_field.alignment) const anyopaque = @alignCast(@ptrCast(shape_field.default_value));
            return @as(*const shape_field.type, @ptrCast(shape_default_value_aligned)).*;
        }

        pub fn init(storage: ?*TensorStorage(dtype, size), allocator: *const Allocator) !*Self {
            const tensor = try allocator.create(Self);
            errdefer allocator.destroy(tensor);
            tensor.* = .{
                // .ndims = ndims,
                // .shape = shape,
                .storage = storage orelse try TensorStorage(dtype, shape_to_size(shape.len, shape)).init(allocator),
                .owns_storage = storage == null,
                .allocator = allocator,
            };
            return tensor;
        }

        pub fn deinit(self: *Self) void {
            if (self.owns_storage) {
                self.storage.deinit();
            }
        }

        pub fn permute(self: *Self, comptime perm: [shape.len]usize) !*Tensor(dtype, shape_permute(shape.len, shape, perm)) {
            const new_shape = comptime shape_permute(shape.len, shape, perm);
            const perm_tensor: *Tensor(dtype, new_shape) = try Tensor(dtype, new_shape).init(self.storage, self.allocator);
            return perm_tensor;
        }

        fn shape_permute(comptime ndims: u8, old_shape: [ndims]usize, perm: [ndims]usize) [ndims]usize {
            var new_shape: [ndims]usize = undefined;
            for (0..ndims) |dim| {
                new_shape[dim] = old_shape[perm[dim]];
            }
            return new_shape;
        }

        pub fn broadcast_with(self: *Self, other_ptr: anytype) [@max(shape.len, comptime_get_tensor_ndims(@TypeOf(other_ptr.*)))]usize {
            _ = self;
            return comptime shape_check: {
                const ndims1 = comptime_get_tensor_ndims(Self);
                const shape1 = comptime_get_tensor_shape(ndims1, Self);
                const tensor2 = @TypeOf(other_ptr.*);
                const ndims2 = comptime_get_tensor_ndims(tensor2);
                const shape2 = comptime_get_tensor_shape(ndims2, tensor2);

                const broadcast_ndims = @max(ndims1, ndims2);
                var broadcast_shape: [broadcast_ndims]usize = undefined;

                var dim1: usize = undefined;
                var dim2: usize = undefined;
                for (0..broadcast_ndims) |i| {
                    dim1 = if (i >= ndims1) 1 else shape1[ndims1 - i - 1];
                    dim2 = if (i >= ndims2) 1 else shape2[ndims2 - i - 1];
                    if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                        @compileError("Cannot broadcast tensors of shapes " ++ std.fmt.comptimePrint("{any}", .{shape1}) ++ " and " ++ std.fmt.comptimePrint("{any}", .{shape2}));
                    }
                    broadcast_shape[broadcast_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
                }
                break :shape_check broadcast_shape;
            };
        }
    };
}

test "test_permute_compile" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tensor1 = try Tensor(i32, .{ 2, 3, 4 }).init(null, &allocator);
    defer tensor1.deinit();
    const tensor2 = try tensor1.permute(.{ 0, 2, 1 });
    defer tensor2.deinit();
    std.debug.print("\ntensor1: {any}\n", .{tensor1});
    std.debug.print("\ntensor2: {any}\n", .{tensor2});
    // Tensor 2 does not own the storage (that belongs to Tensor 1) so when it is freed it will only free its own struct
    // When Tensor 1 is freed, the storage will be freed.
}

test "test_broadcast_compile" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    // Try changing the shape of the tensors. If the shapes don't broadcast together, the code won't compile.
    var tensor1 = try Tensor(i32, .{ 1, 4 }).init(null, &allocator);
    defer tensor1.deinit();
    var tensor2 = try Tensor(i32, .{ 3, 1 }).init(null, &allocator);
    defer tensor2.deinit();
    const broadcast_shape = tensor1.broadcast_with(tensor2);
    std.debug.print("\n{any} and {any} broadcast to {any}\n", .{ tensor1.shape, tensor2.shape, broadcast_shape });
}

test "test_nonnumeric_compile" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const some_type = struct {
        a: bool,
    };
    _ = some_type;
    const tensor1 = try Tensor(c_int, .{ 1, 4 }).init(null, &allocator);
    _ = tensor1;
}
