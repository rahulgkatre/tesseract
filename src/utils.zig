const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;

pub fn arrayPermute(comptime T: type, comptime len: u8, array: [len]T, perm: [len]u8) [len]T {
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
    var new_array: [len]T = undefined;
    for (0..len) |dim| {
        new_array[dim] = array[perm[dim]];
    }
    return new_array;
}

pub fn arrayInsert(comptime T: type, comptime len: u8, array: [len]T, index: usize, val: T) [len + 1]T {
    var new_array: [len + 1]T = undefined;
    for (0..index) |i| {
        new_array[i] = array[i];
    }
    new_array[index] = val;
    for (index..len) |i| {
        new_array[i + 1] = array[i];
    }
    return new_array;
}

pub fn arrayDelete(comptime T: type, comptime len: u8, array: [len]T, index: usize) [len - 1]T {
    var new_array: [len - 1]T = undefined;
    for (0..index) |i| {
        new_array[i] = array[i];
    }
    for (index + 1..len) |i| {
        new_array[i - 1] = array[i];
    }
    return new_array;
}

pub fn ravelMultiIndex(comptime ndims: u8, strides: [ndims + 1]usize, multi_idx: [ndims]usize) usize {
    var flat_idx = strides[ndims];
    for (0..ndims) |d| {
        flat_idx += multi_idx[d] * strides[d];
    }
    return flat_idx;
}
