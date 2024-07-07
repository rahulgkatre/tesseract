const std = @import("std");
const tensor = @import("tensor.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const Tensor = tensor.Tensor;

const dtypes = @import("../dtypes.zig");

// =============================================================================
// Type level utilities specifically for Tensor types
// Equivalents for @as, @TypeOf, @Type
// =============================================================================

/// Like @TypeOf but any must either be a scalar (to construct a scalar Tensor)
/// or a tensor (to reconstruct the Tensor from its dtype shape and ndims)
pub fn TensorTypeOf(any: anytype) type {
    var Type = @TypeOf(any);
    switch (@typeInfo(Type)) {
        .Pointer => |info| Type = info.child,
        else => {},
    }
    switch (@typeInfo(Type)) {
        .Struct => {},
        .Int,
        .Float,
        .Bool,
        .ComptimeInt,
        .ComptimeFloat,
        => return Tensor(Type),
        else => @compileError(std.fmt.comptimePrint("Cannot convert {any} to a tensor type", .{Type})),
    }
    return TensorType(any.dtype, any.shape[0..any.ndims]);
}

/// Like @as but for casting to matching Tensor type
/// Used for wrapping immediate values in single size tensors with the same dtype as the current tensor
/// Will cause a compile error if any is not a Tensor or a scalar number.
pub fn asTensor(comptime any: anytype) TensorTypeOf(any) {
    return switch (@typeInfo(@TypeOf(any))) {
        .Pointer => |info| (if (info.child == AnyTensor) @as(*const AnyTensor, any).toTensor() else any).*,
        .Struct => if (@TypeOf(any) == AnyTensor) @as(AnyTensor, any).toTensor().* else any,
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => TensorTypeOf(any).full(any),
        else => unreachable,
    };
}

/// Like @Type but for constructing a Tensor type from its "type info"
/// Given dtype and shape, recreate the array type and return the corresponing Tensor type
pub fn TensorType(dtype: dtypes.DType, shape: anytype) type {
    var ArrayType = dtypes.ZigType(dtype);
    for (0..shape.len) |dim| {
        ArrayType = [shape[shape.len - dim - 1]]ArrayType;
    }
    return Tensor(ArrayType);
}

pub fn TensorTuple(comptime tensors: anytype) type {
    var types: [tensors.len]type = undefined;
    for (tensors, 0..) |in, i| {
        types[i] = TensorTypeOf(in);
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
