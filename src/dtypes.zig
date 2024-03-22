const std = @import("std");
pub const DType = enum { bool, u8, i8, u16, i16, f16, u32, i32, f32, u64, i64, f64, u128, i128, f128 };

pub fn isFloat(t: DType) bool {
    return switch (t) {
        .f16, .f32, .f64, .f128 => true,
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
            .i8, .i16, .i32, .i64, .i128 => true,
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

pub fn bits(t: DType) u16 {
    return switch (t) {
        .bool => 1,
        .u8, .i8 => 8,
        .u16, .i16, .f16 => 16,
        .u32, .i32, .f32 => 32,
        .u64, .i64, .f64 => 64,
        .u128, .i128, .f128 => 128,
    };
}

pub fn ZigType(comptime dtype: DType) type {
    return switch (dtype) {
        .bool => bool,
        .u8, .i8, .u16, .i16, .u32, .i32, .u64, .i64, .u128, .i128 => std.meta.Int(if (isSigned(dtype)) .signed else .unsigned, bits(dtype)),
        .f16, .f32, .f64, .f128 => std.meta.Float(bits(dtype)),
    };
}
