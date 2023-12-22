const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;

// Used to infer the default (contiguous) strides for the shape
fn comptime_get_tensor_default_strides(comptime ndims: u8, shape: [ndims]usize) [ndims]usize {
    return comptime default_strides: {
        var stride: usize = undefined;
        var offset: usize = 1;
        var strides: [ndims]usize = undefined;
        for (0..ndims - 1) |i| {
            stride = shape[ndims - i - 1] * offset;
            strides[ndims - i - 2] = stride;
            offset = stride;
        }
        strides[ndims - 1] = 1;
        break :default_strides strides;
    };
}

// Wrapper class for a reserved section of memory where the size is part of the type
// This prevents the memory from being used with tensors with an incompatible shape.
fn Buffer(comptime dtype: type, comptime size: usize) type {
    return struct {
        const Self = @This();
        data: []dtype,
        // This allocator is a placeholder for a Device struct that provides an allocator object
        // so that the tensor buffer can be allocated on device memory
        allocator: *const Allocator,
        pub fn init(allocator: *const Allocator) !*Self {
            const buffer = try allocator.create(Self);
            buffer.* = .{
                .data = try allocator.alloc(dtype, size),
                .allocator = allocator,
            };
            return buffer;
        }
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }
    };
}

// TODO: Add a device type param here
// Should be easy to add this type param everywhere as the device will remain the same unless a to(x)
// method is called
pub fn Tensor(comptime dtype: type, comptime shape: anytype) type {
    // Some type checks to make sure that the dtype and shape have valid typing
    // const ndims: u8 = undefined;
    // const shape = undefined;
    // const strides = undefined;

    switch (@typeInfo(dtype)) {
        .Bool => {},
        .ComptimeInt => {},
        .Int => {},
        .ComptimeFloat => {},
        .Float => {},
        else => @compileError("Non-numeric or non-bool tensor dtype not supported, received " ++ @typeName(dtype)),
    }
    // switch (@typeInfo(@TypeOf(shape))) {
    //     .Array => {
    //         shape = shapestrides;
    //         ndims = @truncate(shape.len);
    //         strides = comptime_get_tensor_default_strides(shape);
    //     },
    //     .Struct => |info| {
    //         if (info.fields.len == 2) { //and info.fields[0].name "shape" and info.fields[1].name == "strides") {
    //             shape = shapestrides.shape;
    //             ndims = @truncate(shape.len);
    //             strides = shapestrides.strides;
    //         } else {
    //             @compileError("Invalid shapes and strides object provided");
    //         }
    //     },
    //     else => @compileError("Shape parameter must be an array with fixed size or a struct containing shape stride fields, received " ++ @typeName(@TypeOf(shapestrides))),
    // }

    return struct {
        const Self = @This();
        // comptime const used for array sizing
        const size = comptime_get_tensor_size();
        const ndims = shape.len;
        // runtime accessible field
        ndims: u8 = ndims,
        shape: [ndims]usize = shape,
        strides: [ndims]usize = comptime_get_tensor_default_strides(ndims, shape),
        buffer: *Buffer(dtype, size),
        owns_storage: bool,
        allocator: *const Allocator,

        // Used to determine the size of the underlying buffer
        fn comptime_get_tensor_size() usize {
            return comptime get_size: {
                if (ndims > 0) {
                    var _size: usize = 1;
                    for (0..ndims) |dim| {
                        _size *= shape[dim];
                    }
                    break :get_size _size;
                } else {
                    break :get_size 0;
                }
            };
        }
        // NOTE: These functions are a bit unsafe due to the usage of reflection to access fields of a tensor at compile time
        // It makes 2 critical assumptions about the input object:
        // - ndims: u8 is the first field of the struct
        // - shape: [ndims]usize is the second field of the struct
        // As long as these assumptions are valid there should not be any problems as if they are not valid then the code will still fail to compile
        // It would be better if a compile error could be generated to explain that the code was attempting to operate on something that is not a tensor
        fn comptime_get_tensor_ndims(comptime tensor_type: anytype) u8 {
            const info = @typeInfo(tensor_type);
            const ndims_field = info.Struct.fields[0];
            const ndims_default_value_aligned: *align(ndims_field.alignment) const anyopaque = @alignCast(@ptrCast(ndims_field.default_value));
            return @as(*const ndims_field.type, @ptrCast(ndims_default_value_aligned)).*;
        }
        fn comptime_get_tensor_shape(comptime tensor_type: anytype) [ndims]usize {
            const info = @typeInfo(tensor_type);
            const shape_field = info.Struct.fields[1];
            const shape_default_value_aligned: *align(shape_field.alignment) const anyopaque = @alignCast(@ptrCast(shape_field.default_value));
            return @as(*const shape_field.type, @ptrCast(shape_default_value_aligned)).*;
        }
        fn comptime_get_tensor_strides(comptime tensor_type: anytype) [ndims]usize {
            const info = @typeInfo(tensor_type);
            const strides_field = info.Struct.fields[2];
            const strides_default_value_aligned: *align(strides_field.alignment) const anyopaque = @alignCast(@ptrCast(strides_field.default_value));
            return @as(*const strides_field.type, @ptrCast(strides_default_value_aligned)).*;
        }
        // Utility function for permuting the dimensions of a tensor
        // It runs in comptime to determine the return tensor shape, and also at runtime to get the actual new shape
        fn comptime_shape_permute(old_shape: [ndims]usize, perm: [ndims]usize) [ndims]usize {
            var new_shape: [ndims]usize = undefined;
            for (0..ndims) |dim| {
                new_shape[dim] = old_shape[perm[dim]];
            }
            return new_shape;
        }
        // Gets the broadcast shape between current tensor and other tensor if comptime 1possible
        fn shape_broadcast(self: *Self, other_tensor_ptr: anytype) [@max(shape.len, comptime_get_tensor_ndims(@TypeOf(other_tensor_ptr.*)))]usize {
            _ = self;
            return comptime shape_check: {
                const other_tensor_type = @TypeOf(other_tensor_ptr.*);
                const other_ndims = comptime_get_tensor_ndims(other_tensor_type);
                const other_shape = comptime_get_tensor_shape(other_tensor_type);
                const broadcast_ndims = @max(ndims, other_ndims);
                var broadcast_shape: [broadcast_ndims]usize = undefined;
                var dim1: usize = undefined;
                var dim2: usize = undefined;
                for (0..broadcast_ndims) |i| {
                    dim1 = if (i >= ndims) 1 else shape[ndims - i - 1];
                    dim2 = if (i >= other_ndims) 1 else other_shape[other_ndims - i - 1];
                    if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                        @compileError("Cannot broadcast tensors of shapes " ++ comptimePrint("{any}", .{shape}) ++ " and " ++ comptimePrint("{any}", .{other_shape}));
                    }
                    broadcast_shape[broadcast_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
                }
                break :shape_check broadcast_shape;
            };
        }
        // Determine the index in the current tensor given an index in the broadcasted tensor
        // If the current tensor has size of 1 in a dimension, then the index must be 0
        // Otherwise it will be what the broadcasted index is
        fn comptime_broadcast_index(comptime bc_ndims: u8, bc_index: [bc_ndims]usize) [shape.len]usize {
            return comptime get_index: {
                const index: [ndims]usize = undefined;
                for (0..ndims) |i| {
                    index[bc_ndims - i - 1] = if (shape[ndims - i - 1] == 1) 0 else bc_index[bc_ndims - i - 1];
                }
                break :get_index index;
            };
        }
        // fn comptime_is_contiguous() bool {
        //     return comptime is_contiguous: {
        //         for (strides) {

        //         }

        //     }
        // }

        pub fn init(buffer: ?*Buffer(dtype, size), allocator: *const Allocator) !*Self {
            const tensor = try allocator.create(Self);
            errdefer allocator.destroy(tensor);
            tensor.* = .{
                .buffer = buffer orelse try Buffer(dtype, comptime_get_tensor_size()).init(allocator),
                .owns_storage = buffer == null,
                .allocator = allocator,
            };
            return tensor;
        }

        pub fn deinit(self: *Self) void {
            if (self.owns_storage) {
                self.buffer.deinit();
            }
        }

        pub fn empty(allocator: *const Allocator) !*Self {
            return init(null, allocator);
        }

        pub fn full(val: dtype, allocator: *const Allocator) !*Self {
            const tensor = try empty(allocator);
            @memset(tensor.buffer.data, val);
            return tensor;
        }

        pub fn zeros(allocator: *const Allocator) !*Self {
            return full(0, allocator);
        }

        pub fn permute(self: *Self, comptime perm: [shape.len]usize) !*Tensor(dtype, comptime_shape_permute(shape, perm)) {
            const new_shape = comptime comptime_shape_permute(shape, perm);
            const perm_tensor: *Tensor(dtype, new_shape) = try Tensor(dtype, new_shape).init(self.buffer, self.allocator);
            return perm_tensor;
        }

        // pub fn shape_broadcast(self: *Self, other: anytype) {

        // }
    };
}

test "test_permute_compile" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    var tensor1 = try Tensor(i32, [_]usize{ 2, 3, 4 }).zeros(&allocator);
    defer tensor1.deinit();
    const tensor2 = try tensor1.permute([_]usize{ 0, 2, 1 });
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
    var tensor1 = try Tensor(i32, [_]usize{ 1, 4 }).zeros(&allocator);
    defer tensor1.deinit();
    var tensor2 = try Tensor(i32, [_]usize{ 3, 1 }).zeros(&allocator);
    defer tensor2.deinit();
    const broadcast_shape = tensor1.shape_broadcast(tensor2);
    std.debug.print("\n{any} and {any} broadcast to {any}\n", .{ tensor1.shape, tensor2.shape, broadcast_shape });
}

test "test_nonnumeric_compile" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const some_type = struct {
        a: bool,
    };
    _ = some_type;
    const tensor1 = try Tensor(c_int, [_]usize{ 1, 4 }).init(null, &allocator);
    _ = tensor1;
}
