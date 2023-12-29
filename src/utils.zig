const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const Tensor = @import("tensor.zig").Tensor;

pub fn defaultStridedTensor(comptime dtype: type, comptime ndims: u8, comptime shape: [ndims]usize) type {
    return Tensor(dtype, ndims, shape, defaultStrides(ndims, shape));
}
pub fn size(comptime ndims: u8, comptime shape: [ndims]usize) usize {
    // Used to determine the size of the underlying storage
    const shape_vec: @Vector(ndims, usize) = shape;
    var _size: usize = @reduce(.Mul, shape_vec);
    if (_size == 0) @compileError("Illegal tensor size of 0");
    return _size;
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
pub fn permutedTensor(comptime tensor_t: type, comptime perm: [@field(tensor_t, "ndims")]u8) type {
    const dtype = @field(tensor_t, "dtype");
    const ndims = @field(tensor_t, "ndims");
    const shape = @field(tensor_t, "shape");
    const strides = @field(tensor_t, "strides");
    const permute_shape = permuteArray(ndims, shape, perm);
    const permute_strides = permuteArray(ndims, strides, perm);
    return Tensor(dtype, ndims, permute_shape, permute_strides);
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
pub fn flatIndex(comptime ndims: u8, index: [ndims]usize, strides: [ndims]usize) usize {
    // Convert a multidimensional index into a single dimensional index        
    var flat_index: usize = 0;
    for (0..ndims) |d| flat_index += index[d] * strides[d];
    return flat_index;
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
pub fn broadcastedTensor(comptime tensor1_t: type, comptime tensor2_t: type) type {
    const shape = shapeBroadcast(tensor1_t, tensor2_t);
    return Tensor(@field(tensor1_t, "dtype"), shape.len, shape, defaultStrides(shape.len, shape));
}   
pub fn broadcastIndex(comptime ndims: u8, shape: [ndims]usize, comptime bc_ndims: u8, bc_index: [bc_ndims]usize) [ndims]usize {
    // Determine the index in the current tensor given an index in the broadcasted tensor
    // If the current tensor has size of 1 in a dimension, then the index must be 0
    // Otherwise it will be what the broadcasted index is
    const index: [ndims]usize = undefined;
    for (0..ndims) |d| index[bc_ndims - d - 1] = if (shape[ndims - d - 1] == 1) 0 else bc_index[bc_ndims - d - 1];
    return index;
}

pub fn reducedShape(comptime ndims: u8, comptime shape: [ndims]usize, comptime reduce_dim: usize) [ndims]usize {
    // Return a shape where the reduced dim is 1
    var out_shape: [ndims]usize = undefined;
    @memcpy(&out_shape, &shape);
    out_shape[reduce_dim] = 1;
    return out_shape;
}
pub fn reducedTensor(comptime tensor_t: type, comptime reduce_dim: usize) type {
    const dtype = @field(tensor_t, "dtype");
    const ndims = @field(tensor_t, "ndims");
    const shape = reducedShape(ndims, @field(tensor_t, "shape"), reduce_dim);
    const strides = defaultStrides(ndims, shape);
    return Tensor(dtype, ndims, shape, strides);
}
pub fn extendShape(comptime in_ndims: u8, in_shape: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
    // Extend shape by 1 padding it in the new dimensions
    var out_shape: [out_ndims]usize = undefined;
    @memset(&out_shape, 1);
    @memcpy(out_shape[(out_ndims - in_ndims)..], &in_shape);
    return out_shape;
}
pub fn extendStrides(comptime in_ndims: u8, in_strides: [in_ndims]usize, comptime out_ndims: u8) [out_ndims]usize {
    // Extend strides by 0 padding it in the new dimensions
    var out_strides: [out_ndims]usize = undefined;
    @memset(&out_strides, 0);
    @memcpy(out_strides[(out_ndims - in_ndims)..], &in_strides);
    return out_strides;
}
