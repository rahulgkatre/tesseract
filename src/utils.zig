const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const Tensor = @import("tensor.zig").BaseTensor;

pub fn storageSizeForTensor(comptime ndims: u8, shape: [ndims]usize, strides: [ndims + 1]usize) usize {
    if (true) {
        // Size is the product of the shape for contiguous tensors
        var prod: usize = 1;
        for (shape) |dim_size| {
            prod *= dim_size;
        }
        // Add the storage offset
        return prod + strides[ndims];
    } else {
        // TODO: Verify this is correct
        // If the stride is not contiguous then the storage size is 1 + last index
        // last index is the sum of the strides
        var sum: usize = strides[ndims];
        for (0..ndims) |d| {
            sum += shape[d] * strides[d];
        }
        return sum;
    }
}
pub fn stridesFromShape(shape: anytype) [shape.len + 1]usize {
    const ndims = shape.len;
    var offset: usize = 1;
    var strides: [ndims + 1]usize = undefined;
    for (0..ndims - 1) |d| {
        const stride = shape[ndims - d - 1] * offset;
        strides[ndims - d - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = 1;
    strides[ndims] = 0;
    return strides;
}
pub fn permuteArray(comptime len: u8, array: [len]usize, perm: [len]u8) [len]usize {
    var used: [len]bool = [_]bool{false} ** len;
    for (perm) |p| {
        if (p < len and !used[p]) {
            used[p] = true;
        } else {
            const msg = comptimePrint("Invalid permutation {any}", .{perm});
            if (@inComptime()) {
                @compileError(msg);
            } else {
                @panic(msg);
            }
        }
    }
    for (used) |u| {
        if (!u) {
            const msg = "Invalid permutation: " ++ comptimePrint("{any}", .{perm});
            if (@inComptime()) {
                @compileError(msg);
            } else {
                @panic(msg);
            }
        }
    }
    var new_array: [len]usize = undefined;
    for (0..len) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
}
pub fn isContiguous(comptime ndims: u8, strides: [ndims + 1]usize) bool {
    // Check if the strides are contiguous (decreasing order)
    var prev = strides[0];
    for (strides[1..ndims]) |s| {
        if (s > prev and s > 0) {
            return false;
        }
        prev = s;
    }
    return true;
}
// TODO: Add functions to expand along one dim
