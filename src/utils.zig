const std = @import("std");
const comptimePrint = std.fmt.comptimePrint;
const ops = @import("ops.zig");
const dtypes = @import("dtypes.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const tensor = @import("tensor.zig");

pub fn arrayPermute(comptime T: type, comptime len: u8, array: [len]u64, comptime perm: [len]u8) [len]T {
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
            std.log.err("Invalid permutation {any}", .{perm});
            const msg = "An error occurred in tensor validation";
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

pub fn arrayInsert(comptime len: u8, array: [len]u64, index: usize, val: u64) [len + 1]u64 {
    var new_array: [len + 1]u64 = undefined;
    for (0..index) |i| {
        new_array[i] = array[i];
    }
    new_array[index] = val;
    for (index..len) |i| {
        new_array[i + 1] = array[i];
    }
    return new_array;
}

pub fn arrayDelete(comptime len: u8, array: [len]u64, index: usize) [len - 1]u64 {
    var new_array: [len - 1]u64 = undefined;
    for (0..index) |i| {
        new_array[i] = array[i];
    }
    for (index + 1..len) |i| {
        new_array[i - 1] = array[i];
    }
    return new_array;
}

pub fn ravelMultiIndex(comptime ndims: u8, strides: [ndims]u64, offset: u64, multi_idx: [ndims]u64) usize {
    var flat_idx = offset;
    for (multi_idx, strides) |idx, stride| {
        flat_idx += idx * stride;
    }
    return flat_idx;
}

pub fn isContiguous(strides: []const u64) bool {
    var prev: u64 = std.math.maxInt(u64);
    for (strides) |stride| {
        if (stride > 0) {
            if (stride > prev) {
                return false;
            }
            prev = stride;
        }
    }
    return true;
}

// Infer the contiguous stride pattern from the shape
// This is the default stride pattern unless a stride is manually provided
// using asStrided
pub fn contiguousStrides(comptime shape: []const u64) [shape.len]u64 {
    const ndims = shape.len;
    if (ndims == 0) {
        return .{};
    }

    var offset: u64 = 1;
    var strides: [ndims]u64 = undefined;
    for (0..ndims - 1) |d| {
        const stride = shape[ndims - d - 1] * offset;
        strides[ndims - d - 2] = stride;
        offset = stride;
    }
    strides[ndims - 1] = 1;
    for (0..ndims) |d| {
        if (shape[d] == 0 or shape[d] == 1) {
            strides[d] = 0;
        }
    }
    return strides;
}

pub fn broadcastShape(shape1: anytype, shape2: anytype) [@max(shape1.len, shape2.len)]u64 {
    if (std.mem.eql(u64, &shape1, &shape2)) {
        return shape1;
    }
    const bc_ndims = @max(shape1.len, shape2.len);
    var bc_shape: [bc_ndims]u64 = undefined;
    for (0..bc_ndims) |i| {
        const dim1 = if (i >= shape1.len) 1 else shape1[shape1.len - i - 1];
        const dim2 = if (i >= shape2.len) 1 else shape2[shape2.len - i - 1]; // orelse dim1;
        if (dim1 != 1 and dim2 != 1 and dim1 != dim2) {
            @compileError(comptimePrint(
                \\Shapes are not comaptible for broadcasting
                \\Shape 1: {any}
                \\Shape 2: {any}
            ,
                .{ shape1, shape2 },
            ));
        }
        bc_shape[bc_ndims - i - 1] = if (dim1 == dim2 or dim2 == 1) dim1 else dim2;
    }
    return bc_shape;
}

pub fn numEntries(comptime ndims: u8, shape: [ndims]u64) u128 {
    var prod: u128 = 1;
    for (shape) |s| {
        prod *= s;
    }
    return prod;
}

pub fn extractDType(comptime Type: type) dtypes.DType {
    switch (@typeInfo(Type)) {
        .Array => |info| return extractDType(info.child),
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => return @field(dtypes.DType, rawTypeName(Type)),
        .Struct => |info| if (info.backing_integer) |_| return @field(dtypes.DType, rawTypeName(Type)),
        else => {},
    }
    @compileError("ArrayType input for Tensor must be a array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{Type}));
}

pub fn extractNdims(comptime ArrayType: type) u8 {
    switch (@typeInfo(ArrayType)) {
        .Array => |info| return 1 + extractNdims(info.child),
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => return 0,
        .Struct => |info| if (info.backing_integer) |_| return 0,
        else => {},
    }
    @compileError("ArrayType input for Tensor must be a array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{ArrayType}));
}

pub fn extractShape(comptime ArrayType: type) [extractNdims(ArrayType)]u64 {
    switch (@typeInfo(ArrayType)) {
        .Array => |info| return .{info.len} ++ extractShape(info.child),
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => return .{},
        .Struct => |info| if (info.backing_integer) |_| return .{},
        else => {},
    }
    @compileError("ArrayType input for Tensor must be a array type (e.g. [M][N][P]DType), received " ++ std.fmt.comptimePrint("{any}", .{ArrayType}));
}

pub fn rawTypeName(comptime T: type) []const u8 {
    const name = @typeName(T);
    for (0..name.len) |i| {
        if (name[name.len - i - 1] == '.') {
            return name[name.len - i ..];
        }
    }
    return name;
}

pub fn rawTagName(tagged: anytype) []const u8 {
    const name = @tagName(tagged);
    for (0..name.len) |i| {
        if (name[name.len - i - 1] == '.') {
            return name[name.len - i ..];
        }
    }
    return name;
}

pub fn signedToUnsignedDim(ndims: u8, dim: i16) u8 {
    const value = if (dim < 0) ndims + dim else dim;
    if (value < 0 or value > ndims) {
        @compileError(std.fmt.comptimePrint(
            "Dimension index {d} is out of bounds {d}",
            .{ value, ndims },
        ));
    }
    return @intCast(value);
}

pub fn simplifiedView(input: anytype) struct {
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
} {
    // TODO: Simplify contiguous or broadcasted sub intervals of the view
    const t = tensor.asTensor(input);
    if (t.ndims == 0) {
        return .{
            .ndims = 0,
            .shape = &.{},
            .strides = &.{},
            .offset = 0,
        };
    }

    var start_dim: u8 = 0;
    var end_dim = t.ndims;
    var simplified_shape = t.shape.*;
    var simplified_strides = t.strides.*;

    for (1..t.ndims) |dim| {
        if (simplified_strides[t.ndims - dim - 1] == 0) {
            simplified_shape[t.ndims - dim - 1] *= simplified_shape[t.ndims - dim];
            simplified_strides[t.ndims - dim - 1] = 0;
            start_dim += 1;
        } else if (isContiguous(simplified_strides[t.ndims - dim - 1 ..])) {
            simplified_shape[t.ndims - dim - 1] *= simplified_shape[t.ndims - dim];
            simplified_strides[t.ndims - dim - 1] = 1;
            end_dim -= 1;
        } else {
            break;
        }
    }

    return .{
        .ndims = end_dim - start_dim,
        .shape = simplified_shape[start_dim..end_dim],
        .strides = simplified_strides[start_dim..end_dim],
        .offset = t.offset,
    };
}

// test "simplify view" {
//     const s1 = comptime blk: {
//         const t1 = tensor.Tensor([2][2][2]f32).empty();
//         const s1 = simplifiedView(t1);
//         break :blk s1;
//     };
//     try std.testing.expectEqual(s1.shape[0..s1.ndims].*, .{8});
//     try std.testing.expectEqual(s1.strides[0..s1.ndims].*, .{1});

//     const s2 = comptime blk: {
//         const t2 = tensor.Tensor([2]f32).empty().expand(.{ 2, 2, 2 });
//         const s2 = simplifiedView(t2);
//         break :blk s2;
//     };
//     @compileLog(s2.ndims, s2.shape, s2.strides);

//     try std.testing.expectEqual(s2.shape[0..s2.ndims].*, .{ 4, 2 });
//     try std.testing.expectEqual(s2.strides[0..s2.ndims].*, .{ 0, 1 });
// }
