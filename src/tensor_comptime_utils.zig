const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;

pub fn permuteArray(comptime ndims: u8, comptime array: [ndims]usize, perm: [ndims]usize) [ndims]usize {
    // Utility function for permuting an array (tensor shape or strides)
    // It runs in comptime to determine the return tensor shape/strides, and also at runtime to get the actual new shape/strides
    var new_array: [ndims]usize = undefined;
    for (0..ndims) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
}
pub fn defaultStrides(comptime ndims: u8, comptime shape: [ndims]usize) [ndims]usize {
    // Used to infer the default (contiguous) strides for the shape
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
pub fn isContiguous(comptime ndims: u8, comptime strides: [ndims]usize) bool {
    // Check if the strides are contiguous (decreasing order)
    var prev = strides[0];
    for (strides[1..]) |s| {
        if (s > prev) {
            return false;
        }
        prev = s;
    }
    return true;
}

pub fn shapeBroadcast(comptime tensor1_t: type, comptime tensor2_t: type) [@max(@field(tensor1_t, "ndims"), @field(tensor2_t, "ndims"))]usize {
    // Gets the broadcast shape between two tensors if one exists
    // If the two tensors do not broadcast, the code won't compile
    return comptime blk: {
        const tensor1_ndims = @field(tensor1_t, "ndims");
        const tensor1_shape = @field(tensor1_t, "shape");
        const tensor2_ndims = @field(tensor2_t, "ndims");
        const tensor2_shape = @field(tensor2_t, "shape");
        const bc_ndims = @max(tensor1_ndims, tensor2_ndims);
        var bc_shape: [bc_ndims]usize = undefined;
        var dim1: usize = undefined;
        var dim2: usize = undefined;
        inline for (0..bc_ndims) |i| {
            dim1 = if (i >= tensor1_ndims) 1 else tensor1_shape[tensor1_ndims - i - 1];
            dim2 = if (i >= tensor2_ndims) 1 else tensor2_shape[tensor2_ndims - i - 1];
            if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
                @compileError("Cannot broadcast tensors of shapes " ++ comptimePrint("{any}", .{tensor1_shape}) ++ " and " ++ comptimePrint("{any}", .{tensor2_shape}));
            }
            bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
        }
        break :blk bc_shape;
    };
}
pub fn getReducedShape(comptime ndims: u8, comptime shape: [ndims]usize, comptime reduce_dim: usize) [ndims]usize {
    // Return a shape where the reduced dim is 1
    var out_shape: [ndims]usize = undefined;
    @memcpy(&out_shape, &shape);
    out_shape[reduce_dim] = 1;
    return out_shape;
}
