const std = @import("std");
const dtypes = @import("dtypes.zig");
const Tensor = @import("tensor.zig").Tensor;

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
pub fn asTensor(any: anytype) TensorTypeOf(any) {
    @setEvalBranchQuota(std.math.maxInt(u32));
    return switch (@typeInfo(@TypeOf(any))) {
        .Pointer => any.*,
        .Struct => any,
        .Int, .Float, .Bool, .ComptimeInt, .ComptimeFloat => TensorTypeOf(any).full(any),
        else => unreachable,
    };
}

/// Test if a type is a Tensor type
pub fn isTensorType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => Tensor(T.ArrayType()) == T,
        else => false,
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
