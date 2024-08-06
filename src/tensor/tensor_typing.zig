const std = @import("std");
const tensor = @import("tensor.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const Tensor = tensor.Tensor;
const ops = @import("../ops.zig");
const utils = @import("../utils.zig");
const dtypes = @import("../dtypes.zig");

// =============================================================================
// Type level utilities specifically for Tensor types
// Equivalents for @as, @TypeOf, @Type
// =============================================================================

/// Like @TypeOf but any must either be a scalar (to construct a scalar Tensor)
/// or a tensor (to reconstruct the Tensor from its dtype shape and ndims)
pub fn AsTensorType(comptime Type: type) type {
    var T = Type;
    switch (@typeInfo(T)) {
        .Pointer => |info| T = info.child,
        else => {},
    }
    switch (@typeInfo(T)) {
        .Struct => {},
        .Int,
        .Float,
        .Bool,
        .ComptimeInt,
        .ComptimeFloat,
        => return Tensor(T),
        else => @compileError(std.fmt.comptimePrint("Cannot convert {any} to a tensor type", .{T})),
    }
    return TensorType(T.dtype, T.shape[0..T.ndims]);
}

/// Like @as but for casting to matching Tensor type
/// Used for wrapping immediate values in single size tensors with the same dtype as the current tensor
/// Will cause a compile error if any is not a Tensor or a scalar number.
pub fn asTensor(any: anytype, allocator: std.mem.Allocator) *const AsTensorType(@TypeOf(any)) {
    return switch (@typeInfo(@TypeOf(any))) {
        .Pointer => any,
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => AsTensorType(@TypeOf(any)).full(any, allocator) catch unreachable,
        else => unreachable,
    };
}

/// Like @Type but for constructing a Tensor type from its "type info"
/// Given dtype and shape, recreate the array type and return the corresponing Tensor type
pub fn TensorType(dtype: dtypes.DType, shape: anytype) type {
    var Array = dtypes.ZigType(dtype);
    for (0..shape.len) |dim| {
        Array = [shape[shape.len - dim - 1]]Array;
    }
    return Tensor(Array);
}

pub fn ArrayType(comptime T: type) type {
    var Child = dtypes.ZigType(T.dtype);
    for (0..T.ndims) |dim| {
        Child = [T.shape[T.ndims - dim - 1]]Child;
    }
    return Child;
}

pub fn TensorTuple(comptime tensors: anytype) type {
    var types: [tensors.len]type = undefined;
    for (tensors, 0..) |in, i| {
        types[i] = AsTensorType(in);
    }
    return std.meta.Tuple(&types);
}

/// Utility function to enforce that T must be float-like
pub fn FloatTensor(comptime T: type) type {
    if (dtypes.isFloat(T.dtype)) {
        return T;
    }
    return TensorType(dtypes.default_float, T.shape);
}

/// Utility function to enforce that T must be bool-like
pub fn BoolTensor(comptime T: type) type {
    if (!dtypes.isBool(T.dtype)) {
        @compileError("Must be bool datatype");
    }
    return TensorType(.bool, T.shape);
}

/// Utility function to enforce that T must be int-like
pub fn IntTensor(comptime T: type) type {
    if (!dtypes.isInt(T.dtype)) {
        @compileError("Must cast to int datatype first");
    }
    return TensorType(dtypes.default_int, T.shape);
}

pub fn Cast(comptime T: type, new_dtype: dtypes.DType) type {
    return TensorType(new_dtype, T.shape);
}

pub fn Pad(comptime T: type, padding: anytype) type {
    const padded_dims = padding.len;
    const padding_tuple: [padded_dims][2]u64 = padding;
    std.debug.assert(padded_dims <= T.ndims);
    var new_shape: [T.ndims]usize = T.shape;
    for (0..padded_dims) |dim| {
        new_shape[T.ndims - dim - 1] += padding_tuple[dim][0] + padding_tuple[dim][1];
    }
    return TensorType(T.dtype, new_shape);
}

pub fn View(comptime X: type, comptime new_shape: anytype) type {
    const T = AsTensorType(X);
    return TensorType(T.dtype, new_shape);
}

pub fn UnaryOpResult(comptime T: type, comptime op: ops.UnaryOp) type {
    return switch (op) {
        .exp2, .log2, .recip, .sin, .sqrt => FloatTensor(AsTensorType(T)),
        .neg => AsTensorType(T),
    };
}

pub fn BinaryOpResult(comptime A: type, comptime B: type, comptime op: ops.BinaryOp) type {
    const T1 = AsTensorType(A);
    const T2 = AsTensorType(B);
    const new_dtype: dtypes.DType = switch (op) {
        .eq, .lt => .bool,
        else => dtypes.resultDType(T1.dtype, T2.dtype),
    };
    return TensorType(new_dtype, utils.broadcastShape(T1.shape, T2.shape));
}

pub fn ReduceOpResult(comptime T: type, comptime reduce_dims: anytype) type {
    const reduced_shape: [T.ndims]u64 = switch (@typeInfo(@TypeOf(reduce_dims))) {
        .ComptimeInt, .Int => blk: {
            const dim = T.signedToUnsignedDim(reduce_dims);
            if (dim < 0 or dim >= T.ndims) {
                @compileError("Dimension index for single dimension reduce is out of bounds");
            }
            var reduced_shape: [T.ndims]u64 = T.shape;
            reduced_shape[dim] = 1;
            break :blk reduced_shape;
        },
        .Null, .Void => blk: {
            break :blk .{1} ** T.ndims;
        },
        else => blk: {
            const dims = reduce_dims;
            if (dims.len > T.ndims) {
                @compileError("Length of dimension index array for multi dimension reduce is out of bounds");
            }
            var reduce_dim_mask: [T.ndims]bool = [_]bool{false} ** T.ndims;
            var reduced_shape: [T.ndims]u64 = T.shape;
            for (0..dims.len) |d| {
                const norm = T.signedToUnsignedDim(d);
                if (reduce_dim_mask[norm]) {
                    @compileError("Cannot reuse dimension index for multi dimensional reduce");
                }
                reduce_dim_mask[d] = true;
                reduced_shape[d] = 1;
            }
            break :blk reduced_shape;
        },
    };
    return TensorType(T.dtype, reduced_shape);
}
