const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const Tensor = @import("tensor.zig").BaseTensor;

pub fn bufferSizeForTensor(comptime ndims: u8, shape: [ndims]usize, strides: [ndims]usize) usize {
    if (isContiguous(ndims, strides)) {
        // Size is the product of the shape for contiguous tensors
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
pub fn permuteArray(comptime ndims: u8, array: [ndims]usize, perm: [ndims]u8) [ndims]usize {
    var used: [ndims]bool = [_]bool{false} ** ndims;
    for (perm) |p| {
        if (p < ndims and !used[p]) {
            used[p] = true;
        } else {
            const msg = "Invalid permutation: " ++ comptimePrint("{any}", .{perm});
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
    var new_array: [ndims]usize = undefined;
    for (0..ndims) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
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
// TODO: Add functions to expand along one dim
