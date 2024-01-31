const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;

pub fn tensorString(comptime TensorType: type) []const u8 {
    return comptime blk: {
        var tmp: []const u8 = "tensor<";
        for (0..TensorType.ndims) |d| {
            tmp = tmp ++ comptimePrint("{d}x", .{TensorType.shape[d]});
        }
        tmp = tmp ++ @typeName(TensorType.dtype) ++ ">";
        break :blk tmp;
    };
}
pub fn storageSizeForTensor(comptime ndims: u8, shape: [ndims]usize, strides: [ndims + 1]usize) usize {
    // The storage size is 1 + last index calculated by the strides and shape
    // shape[d] - 1 is the last index in dimension d
    // Also incorporate the storage offset
    var size: usize = strides[ndims] + 1;
    for (0..ndims) |d| {
        size += (shape[d] - 1) * strides[d];
    }
    // The result is the size of the storage needed to visit all indices of the tensor
    return size;
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
    for (0..ndims) |d| {
        if (shape[d] == 0 or shape[d] == 1) {
            strides[d] = 0;
        }
    }
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
            const msg = comptimePrint("Invalid permutation: {any}", .{perm});
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
    var prev = strides[0];
    for (strides[1..ndims]) |s| {
        if (s > prev and s > 0) {
            return false;
        }
        prev = s;
    }
    return true;
}
