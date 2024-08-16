const std = @import("std");
const tensor = @import("tensor.zig");
const AnyTensor = @import("tensor.zig").AnyTensor;
const Tensor = tensor.Tensor;
const ops = @import("../ops.zig");
const utils = @import("../utils.zig");
const dtypes = @import("../dtypes.zig");

pub const Layout = struct {
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
};

pub const Labels = struct {
    name: ?[]const u8,
    dim_names: ?[]const ?[]const u8,
};

pub fn DimsEnumType(comptime maybe_dim_names: ?[]const ?[]const u8) type {
    if (maybe_dim_names) |dim_names| {
        var dim_enum_fields: [dim_names.len]std.builtin.Type.EnumField = undefined;
        var enum_idx: usize = 0;

        for (dim_names, 0..) |maybe_name, dim_idx| {
            if (maybe_name) |name| {
                dim_enum_fields[enum_idx] = std.builtin.Type.EnumField{ .name = name[0.. :0], .value = dim_idx };
                enum_idx += 1;
            }
        }
        return @Type(std.builtin.Type{ .Enum = .{ .fields = dim_enum_fields[0..enum_idx], .is_exhaustive = false, .tag_type = u8, .decls = &.{} } });
    } else {
        return void;
    }
}

pub const Json = struct {
    ptr: usize,
    dtype: dtypes.DType,
    ndims: u8,
    shape: []const u64,
    strides: []const u64,
    offset: u64,
};

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
    return TensorType(any.layout.dtype, any.layout.shape);
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

pub fn Pad(Self: type, padding: anytype) type {
    if (Self == AnyTensor) unreachable;
    const padded_dims = padding.len;
    const padding_tuple: [padded_dims][2]u64 = padding;
    std.debug.assert(padded_dims <= Self._ndims);
    var new_shape: [Self._ndims]usize = Self._shape;
    for (0..padded_dims) |dim| {
        new_shape[Self._ndims - dim - 1] += padding_tuple[dim][0] + padding_tuple[dim][1];
    }
    return View(Self, new_shape);
}

pub fn View(Self: type, new_shape: anytype) type {
    if (Self == AnyTensor) unreachable;
    return TensorType(Self._dtype, new_shape);
}

pub fn UnaryOpResult(Self: type, op: ops.UnaryOp) type {
    if (Self == AnyTensor) unreachable;
    return switch (op) {
        .exp2, .log2, .recip, .sin, .sqrt => FloatTensor(Self),
        .neg => TensorType(Self._dtype, Self._shape),
    };
}

pub fn BinaryOpResult(Self: type, Other: type, comptime op: ops.BinaryOp) type {
    if (Self == AnyTensor) unreachable;
    std.debug.assert(Other != AnyTensor);
    const new_dtype: dtypes.DType = switch (op) {
        .eq, .lt => .bool,
        else => dtypes.resultDType(Self._dtype, Other._dtype),
    };
    return TensorType(new_dtype, utils.broadcastShape(Self._shape, Other._shape));
}

pub fn ReduceOpResult(Self: type, reduce_dims: anytype) type {
    if (Self == AnyTensor) unreachable;
    const reduced_shape: [Self._ndims]u64 = switch (@typeInfo(@TypeOf(reduce_dims))) {
        .ComptimeInt, .Int => blk: {
            const dim = Self.signedToUnsignedDim(reduce_dims);
            if (dim < 0 or dim >= Self._ndims) {
                @compileError("Dimension index for single dimension reduce is out of bounds");
            }
            var reduced_shape: [Self._ndims]u64 = Self._shape;
            reduced_shape[dim] = 1;
            break :blk reduced_shape;
        },
        .Null, .Void => blk: {
            break :blk .{1} ** Self._ndims;
        },
        else => blk: {
            const dims = reduce_dims;
            if (dims.len > Self._ndims) {
                @compileError("Length of dimension index array for multi dimension reduce is out of bounds");
            }
            var reduce_dim_mask: [Self._ndims]bool = [_]bool{false} ** Self._ndims;
            var reduced_shape: [Self._ndims]u64 = Self._shape;
            for (0..dims.len) |d| {
                const norm = Self.signedToUnsignedDim(d);
                if (reduce_dim_mask[norm]) {
                    @compileError("Cannot reuse dimension index for multi dimensional reduce");
                }
                reduce_dim_mask[d] = true;
                reduced_shape[d] = 1;
            }
            break :blk reduced_shape;
        },
    };
    return TensorType(Self._dtype, reduced_shape);
}

/// Utility function to enforce that T must be float-like
pub fn FloatTensor(comptime Self: type) type {
    if (Self == AnyTensor) unreachable;
    if (dtypes.isFloat(Self._dtype)) return Self;
    return TensorType(dtypes.default_float, Self._shape);
}

/// Utility function to enforce that T must be bool-like
pub fn BoolTensor(comptime Self: type) type {
    if (Self == AnyTensor) unreachable;
    if (!dtypes.isBool(Self._dtype)) @compileError("Must be bool datatype");

    return TensorType(.bool, Self._shape);
}

/// Utility function to enforce that T must be int-like
pub fn IntTensor(comptime Self: type) type {
    if (Self == AnyTensor) unreachable;
    if (!dtypes.isInt(Self._dtype)) @compileError("Must cast to int datatype first");

    return TensorType(dtypes.default_int, Self._shape);
}
