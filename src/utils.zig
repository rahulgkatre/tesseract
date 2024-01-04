const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const Tensor = @import("tensor.zig").BaseTensor;

pub fn bufferSizeForTensor(comptime ndims: u8, shape: [ndims]usize, strides: [ndims]usize) usize {
    // Determine the size of the underlying storage
    if (isContiguous(ndims, strides)) {
        // Size is the product of the shape
        var prod: usize = 1;
        for (shape) |dim_size| prod *= dim_size;
        return prod;
    } else {
        // TODO: Verify this is correct
        // If the stride is not contiguous then the buffer size is 1 + last index
        // last index is the sum of the strides
        var sum: usize = 1;
        for (0..ndims) |d| sum += shape[d] * strides[d];
        return sum + 1;
    }
}
pub fn permuteArray(comptime ndims: u8, comptime array: [ndims]usize, perm: [ndims]u8) [ndims]usize {
    // Utility function for permuting an array (tensor shape or strides)
    // TODO: Add checks to make sure dim is not being reused.
    var new_array: [ndims]usize = undefined;
    for (0..ndims) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
}
pub fn defaultStrides(comptime ndims: u8, comptime shape: [ndims]usize) [ndims]usize {
    // Infer the default (contiguous) strides for the shape
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
pub fn isContiguous(comptime ndims: u8, strides: [ndims]usize) bool {
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
    return bc_shape;
}
pub fn reducedShape(comptime ndims: u8, comptime shape: [ndims]usize, comptime reduce_dim: usize) [ndims]usize {
    // Return a shape where the reduced dim is 1
    var out_shape: [ndims]usize = undefined;
    @memcpy(&out_shape, &shape);
    out_shape[reduce_dim] = 1;
    return out_shape;
}
// TODO: Add functions to expand along one dim, I don't think we need extend
// pub fn extendShape(comptime in_ndims: u8, in_shape: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
//     // Extend shape by 1 padding it in the new dimensions
//     var out_shape: [out_ndims]usize = undefined;
//     @memset(&out_shape, 1);
//     @memcpy(out_shape[(out_ndims - in_ndims)..], &in_shape);
//     return out_shape;
// }
// pub fn extendStrides(comptime in_ndims: u8, in_strides: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
//     // Extend strides by 0 padding it in the new dimensions
//     var out_strides: [out_ndims]usize = undefined;
//     @memset(&out_strides, 0);
//     @memcpy(out_strides[(out_ndims - in_ndims)..], &in_strides);
//     return out_strides;
// }
