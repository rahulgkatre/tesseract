const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;

// Generate the ID used for autodifferentiation at compile time
// https://stackoverflow.com/questions/68555025/global-comptime-var-in-zig
// TODO: not working
// pub const TensorIdGenerator = id: {
//     comptime var nextId: usize = 0;
//     const ptr: *usize = &nextId;
//     const NextIdResult = struct {
//         pub fn getAndIncrement() usize {
//             const ret = ptr.*;
//             ptr.* += 1;
//             return ret;
//         }
//     };
//     break :id NextIdResult;
// };

// NOTE: These functions are a bit unsafe due to the usage of reflection to access fields of a tensor at compile time
// Reflection is needed here because shapes are part of the type, but it seems like for generics the param info is lost
// It makes 2 critical assumptions about the input objecret:
// - ndims: u8 is the first field of the struct
// - shape: [ndims]usize is the second field of the struct
// As long as these assumptions are valid there should not be any problems as if they are not valid then the code will still fail to compile
// It would be better if a compile error could be generated to explain that the code was attempting to operate on something that is not a tensor
fn unsafeGetNdims(comptime tensor_type: type) u8 {
    const info = @typeInfo(tensor_type);
    const ndims_field = info.Struct.fields[0];
    const ndims_default_value_aligned: *align(ndims_field.alignment) const anyopaque = @alignCast(@ptrCast(ndims_field.default_value));
    return @as(*const ndims_field.type, @ptrCast(ndims_default_value_aligned)).*;
}
fn unsafeGetShape(comptime ndims: u8, comptime tensor_type: type) [ndims]usize {
    const info = @typeInfo(tensor_type);
    const shape_field = info.Struct.fields[1];
    const shape_default_value_aligned: *align(shape_field.alignment) const anyopaque = @alignCast(@ptrCast(shape_field.default_value));
    return @as(*const shape_field.type, @ptrCast(shape_default_value_aligned)).*;
}
fn unsafeGetStrides(comptime ndims: u8, comptime tensor_type: type) [ndims]usize {
    const info = @typeInfo(tensor_type);
    const strides_field = info.Struct.fields[2];
    const strides_default_value_aligned: *align(strides_field.alignment) const anyopaque = @alignCast(@ptrCast(strides_field.default_value));
    return @as(*const strides_field.type, @ptrCast(strides_default_value_aligned)).*;
}
// Utility function for permuting an array (tensor shape or strides)
// It runs in comptime to determine the return tensor shape/strides, and also at runtime to get the actual new shape/strides
pub fn permuteArray(comptime ndims: u8, comptime array: [ndims]usize, perm: [ndims]usize) [ndims]usize {
    var new_array: [ndims]usize = undefined;
    for (0..ndims) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
}
// Used to infer the default (contiguous) strides for the shape
pub fn defaultStrides(comptime ndims: u8, comptime shape: [ndims]usize) [ndims]usize {
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
// Check if the strides are contiguous (decreasing order)
pub fn isContiguous(comptime ndims: u8, comptime strides: [ndims]usize) bool {
    var prev = strides[0];
    for (strides[1..]) |s| {
        if (s > prev) {
            return false;
        }
        prev = s;
    }
    return true;
}
// Gets the broadcast shape between two tensors if one exists
// If the two tensors do not broadcast, the code won't compile
pub fn shapeBroadcast(comptime tensor1_type: type, comptime tensor2_type: type) [@max(unsafeGetNdims(tensor1_type), unsafeGetNdims(tensor2_type))]usize {
    return comptime ret: {
        const tensor1_ndims = unsafeGetNdims(tensor1_type);
        const tensor1_shape = unsafeGetShape(tensor1_ndims, tensor1_type);

        const tensor2_ndims = unsafeGetNdims(tensor2_type);
        const tensor2_shape = unsafeGetShape(tensor2_ndims, tensor2_type);

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
// Return a shape where the reduced dim is 1
pub fn getReducedShape(comptime ndims: u8, comptime shape: [ndims]usize, comptime reduce_dim: usize) [ndims]usize {
    var out_shape: [ndims]usize = undefined;
    @memcpy(&out_shape, &shape);
    out_shape[reduce_dim] = 1;
    return out_shape;
}
