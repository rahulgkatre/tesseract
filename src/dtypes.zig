const std = @import("std");
const utils = @import("utils.zig");
const tensor = @import("tensor/tensor.zig");

// For dtypes without a corresponding zig type, they are represented
// as packed structs
// A unsigned int of the same bitsize would also work but the labels
// like "bf16" would be squashed, so this is a better solution

/// Brain floating point
pub const bf16 = packed struct {
    sign: u1 = 0,
    exponent: u8 = 0,
    mantissa: u7 = 0,
};

test bf16 {
    try std.testing.expectEqual(tensor.Tensor([2][3]bf16)._dtype, .bf16);
}

/// Nvidia Tensor Float
pub const TF32 = packed struct {
    sign: u1 = 0,
    exponent: u8 = 0,
    mantissa: u10 = 0,
};

pub const DType = enum(u8) {
    anyopaque,
    comptime_int,
    comptime_float,
    bool,
    u8,
    i8,
    u16,
    i16,
    f16,
    bf16,
    u32,
    i32,
    f32,
    TF32,
    u64,
    i64,
    f64,
    u128,
    i128,
    f128,
};

pub const default_float: DType = .f32;
pub const default_int: DType = .i32;

pub fn isComptime(t: DType) bool {
    return switch (t) {
        .anyopaque, .bool, .comptime_float, .comptime_int => true,
        else => false,
    };
}

pub fn isFloat(t: DType) bool {
    return switch (t) {
        .comptime_float, .f16, .bf16, .f32, .TF32, .f64, .f128 => true,
        else => false,
    };
}

pub fn isSigned(t: DType) bool {
    if (isBool(t)) {
        return false;
    } else if (isFloat(t)) {
        return true;
    } else {
        return switch (t) {
            .comptime_int, .i8, .i16, .i32, .i64, .i128 => true,
            else => false,
        };
    }
}

pub fn isInt(t: DType) bool {
    return !isFloat(t) and !isBool(t);
}

pub fn isBool(t: DType) bool {
    return switch (t) {
        .bool => true,
        else => false,
    };
}

pub fn bits(t: DType) u8 {
    return switch (t) {
        .anyopaque, .comptime_int, .comptime_float => 0,
        .bool => 1,
        .u8, .i8 => 8,
        .u16, .i16, .f16 => 16,
        .u32, .i32, .f32 => 32,
        .u64, .i64, .f64 => 64,
        .u128, .i128, .f128 => 128,
        else => @typeInfo(ZigType(t)).Struct.backing_integer,
    };
}

/// Each dtype has a corresponding Zig type that can be used for the array definition
/// This function provides a way to get that type
pub fn ZigType(comptime dtype: DType) type {
    return switch (dtype) {
        .anyopaque => anyopaque,
        .comptime_float => comptime_float,
        .comptime_int => comptime_int,
        .bool => bool,
        .u8, .i8, .u16, .i16, .u32, .i32, .u64, .i64, .u128, .i128 => std.meta.Int(
            if (isSigned(dtype)) .signed else .unsigned,
            bits(dtype),
        ),
        .f16, .f32, .f64, .f128 => std.meta.Float(bits(dtype)),
        .bf16 => f16,
        .TF32 => TF32,
    };
}

pub fn inferDType(comptime value: anytype) DType {
    return switch (@typeInfo(@TypeOf(value))) {
        .Bool => .bool,
        .ComptimeInt => .comptime_int,
        .ComptimeFloat => .comptime_float,
        .Int, .Float => @field(DType, utils.rawTypeName(@TypeOf(value))),
        else => @compileError(@typeName(@TypeOf(value)) ++ " is not a valid tensor data type"),
    };
}

pub fn resultDType(dtype1: DType, dtype2: DType) DType {
    if (dtype1 == dtype2) {
        return dtype1;
    }

    const is_bool1 = isBool(dtype1);
    const is_bool2 = isBool(dtype2);
    const is_float1 = isFloat(dtype1);
    const is_float2 = isFloat(dtype2);
    const is_int1 = isInt(dtype1);
    const is_int2 = isInt(dtype2);

    if (is_bool1 and is_bool2) {
        // bool, bool -> bool
        return .bool;
    }

    const bits1 = bits(dtype1);
    const bits2 = bits(dtype2);
    // If both are same (float or int) choose the one with more bits
    if ((is_float1 and is_float2) or (is_int1 and is_int2)) {
        if (bits1 > bits2) {
            return dtype1;
        } else {
            return dtype2;
        }
    }

    @compileError("Cannot combine " ++ utils.rawTagName(dtype1) ++ " and " ++ utils.rawTagName(dtype2));
}
