const std = @import("std");
const Allocator = std.mem.Allocator;
const comptimePrint = std.fmt.comptimePrint;

// NOTE: These functions are a bit unsafe due to the usage of reflection to access fields of a tensor at compile time
// Reflection is needed here because shapes are part of the type, but it seems like for generics the param info is lost
// It makes 2 critical assumptions about the input objecret:
// - ndims: u8 is the first field of the struct
// - shape: [ndims]usize is the second field of the struct
// As long as these assumptions are valid there should not be any problems as if they are not valid then the code will still fail to compile
// It would be better if a compile error could be generated to explain that the code was attempting to operate on something that is not a tensor
fn unsafe_get_ndims(comptime tensor_type: type) u8 {
    const info = @typeInfo(tensor_type);
    const ndims_field = info.Struct.fields[0];
    const ndims_default_value_aligned: *align(ndims_field.alignment) const anyopaque = @alignCast(@ptrCast(ndims_field.default_value));
    return @as(*const ndims_field.type, @ptrCast(ndims_default_value_aligned)).*;
}
fn unsafe_get_shape(comptime ndims: u8, comptime tensor_type: type) [ndims]usize {
    const info = @typeInfo(tensor_type);
    const shape_field = info.Struct.fields[1];
    const shape_default_value_aligned: *align(shape_field.alignment) const anyopaque = @alignCast(@ptrCast(shape_field.default_value));
    return @as(*const shape_field.type, @ptrCast(shape_default_value_aligned)).*;
}
fn unsafe_get_strides(comptime ndims: u8, comptime tensor_type: type) [ndims]usize {
    const info = @typeInfo(tensor_type);
    const strides_field = info.Struct.fields[2];
    const strides_default_value_aligned: *align(strides_field.alignment) const anyopaque = @alignCast(@ptrCast(strides_field.default_value));
    return @as(*const strides_field.type, @ptrCast(strides_default_value_aligned)).*;
}
// Utility function for permuting an array (tensor shape or strides)
// It runs in comptime to determine the return tensor shape/strides, and also at runtime to get the actual new shape/strides
fn permute_array(comptime ndims: u8, comptime array: [ndims]usize, perm: [ndims]usize) [ndims]usize {
    var new_array: [ndims]usize = undefined;
    for (0..ndims) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
}
// Used to infer the default (contiguous) strides for the shape
fn default_strides(comptime ndims: u8, comptime shape: [ndims]usize) [ndims]usize {
    var stride: usize = undefined;
    var offset: usize = 1;
    var strides: [ndims]usize = undefined;
    for (0..ndims - 1) |i| {
        stride = shape[ndims - i - 1] * offset;
        strides[ndims - i - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = 1;
    return strides;
}
fn is_strides_contiguous(comptime ndims: u8, comptime strides: [ndims]usize) bool {
    var prev = strides[0];
    for (strides[1..]) |s| {
        if (prev <= s) {
            return false;
        }
        prev = s;
    }
    return true;
}
fn shape_broadcast(comptime tensor1_type: type, comptime tensor2_type: type) [@max(unsafe_get_ndims(tensor1_type), unsafe_get_ndims(tensor2_type))]usize {
    return comptime ret: {
        const tensor1_ndims = unsafe_get_ndims(tensor1_type);
        const tensor1_shape = unsafe_get_shape(tensor1_ndims, tensor1_type);

        const tensor2_ndims = unsafe_get_ndims(tensor2_type);
        const tensor2_shape = unsafe_get_shape(tensor2_ndims, tensor2_type);

        const bc_ndims = @max(tensor1_ndims, tensor2_ndims);
        var bc_shape: [bc_ndims]usize = undefined;

        var dim1: usize = undefined;
        var dim2: usize = undefined;
        for (0..bc_ndims) |i| {
            dim1 = if (i >= tensor1_ndims) 1 else tensor1_shape[tensor1_ndims - i - 1];
            dim2 = if (i >= tensor2_ndims) 1 else tensor2_shape[tensor2_ndims - i - 1];
            if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                @compileError("Cannot broadcast tensors of shapes " ++ comptimePrint("{any}", .{tensor1_shape}) ++ " and " ++ comptimePrint("{any}", .{tensor2_shape}));
            }
            bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
        }
        break :ret bc_shape;
    };
}
fn get_reduced_shape(comptime ndims: u8, comptime shape: [ndims]usize, comptime reduce_dim: usize) [ndims]usize {
    var out_shape: [ndims]usize = undefined;
    @memcpy(&out_shape, &shape);
    out_shape[reduce_dim] = 1;
    return out_shape;
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

pub fn tensor(comptime dtype: type, comptime shape: anytype, allocator: *const Allocator) !ret: {
    break :ret *Tensor(dtype, shape.len, shape, default_strides(shape.len, shape));
} {
    return try Tensor(dtype, shape.len, shape, default_strides(shape.len, shape)).init(null, allocator);
}

pub fn full(comptime dtype: type, comptime shape: anytype, val: dtype, allocator: *const Allocator) !ret: {
    const strides = default_strides(shape.len, shape);
    break :ret *Tensor(dtype, shape.len, shape, strides);
} {
    const t = try tensor(dtype, shape, allocator);
    @memset(t.buffer.data, val);
    return t;
}

pub fn zeros(comptime dtype: type, comptime shape: anytype, allocator: *const Allocator) !*Tensor(dtype, shape.len, shape, default_strides(shape.len, shape)) {
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
        buffer: *Buffer(dtype, size()),
        owns_storage: bool,
        allocator: *const Allocator,
        // Used to determine the size of the underlying buffer
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
        // Gets the broadcast shape between current tensor and other tensor if comptime possible

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
            return comptime is_strides_contiguous(ndims, strides);
        }
        pub fn init(buffer: ?*Buffer(dtype, size()), allocator: *const Allocator) !*Self {
            const t = try allocator.create(Self);
            errdefer allocator.destroy(t);
            t.* = .{
                .buffer = buffer orelse try Buffer(dtype, size()).init(allocator),
                .owns_storage = buffer == null,
                .allocator = allocator,
            };
            return t;
        }
        pub fn deinit(self: *Self) void {
            if (self.owns_storage) {
                self.buffer.deinit();
            }
        }
        pub fn permute(self: *Self, comptime perm: [ndims]usize) !ret: {
            const permute_shape = permute_array(ndims, shape, perm);
            const permute_strides = permute_array(ndims, strides, perm);
            break :ret *Tensor(dtype, ndims, permute_shape, permute_strides);
        } {
            const new_shape = comptime permute_array(ndims, shape, perm);
            const new_strides = comptime permute_array(ndims, strides, perm);
            return try Tensor(dtype, ndims, new_shape, new_strides).init(self.buffer, self.allocator);
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
            const out_shape = shape_broadcast(Self, @TypeOf(other_tensor_ptr.*));
            const out_ndims = out_shape.len;
            const out_strides = default_strides(out_ndims, out_shape);
            break :ret *Tensor(dtype, out_ndims, out_shape, out_strides);
        } {
            _ = zip_op;
            const out_shape = comptime shape_broadcast(Self, @TypeOf(other_tensor_ptr.*));
            return try tensor(dtype, out_shape, self.allocator);
        }
        // Reduce is many to 1 and collapses a dimension to 1 (sum, prod, max, etc.)
        pub fn mock_reduce_fn(self: *Self, reduce_op: anytype, comptime reduce_dim: usize) !ret: {
            const out_shape = get_reduced_shape(ndims, shape, reduce_dim);
            const out_strides = default_strides(ndims, out_shape);
            break :ret *Tensor(dtype, ndims, out_shape, out_strides);
        } {
            _ = reduce_op;
            const out_shape = comptime get_reduced_shape(ndims, shape, reduce_dim);
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
    const broadcast_shape = comptime shape_broadcast(@TypeOf(tensor1.*), @TypeOf(tensor2.*));
    // const broadcast_shape = tensor1.shape_broadcast(tensor2);
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
