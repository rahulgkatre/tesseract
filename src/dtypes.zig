const std = @import("std");
const tensor = @import("tensor.zig");
pub const DType = enum(u8) {
    bool_literal,
    int_literal,
    float_literal,
    bool,
    u8,
    i8,
    u16,
    i16,
    f16,
    u32,
    i32,
    f32,
    u64,
    i64,
    f64,
    u128,
    i128,
    f128,
};

pub const default_float: DType = .f32;
pub const default_int: DType = .i32;

pub fn isLiteral(t: DType) bool {
    return switch (t) {
        .bool_literal, .float_literal, .int_literal => true,
        else => false,
    };
}

pub fn isFloat(t: DType) bool {
    return switch (t) {
        .float_literal, .f16, .f32, .f64, .f128 => true,
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
            .int_literal, .i8, .i16, .i32, .i64, .i128 => true,
            else => false,
        };
    }
}

pub fn isInt(t: DType) bool {
    return !isFloat(t) and !isBool(t);
}

pub fn isBool(t: DType) bool {
    return switch (t) {
        .bool, .bool_literal => true,
        else => false,
    };
}

pub fn bits(t: DType) u8 {
    return switch (t) {
        .bool_literal, .int_literal, .float_literal => 0,
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
        .float_literal => comptime_float,
        .int_literal => comptime_int,
        .bool, .bool_literal => bool,
        .u8, .i8, .u16, .i16, .u32, .i32, .u64, .i64, .u128, .i128 => std.meta.Int(
            if (isSigned(dtype)) .signed else .unsigned,
            bits(dtype),
        ),
        .f16, .f32, .f64, .f128 => std.meta.Float(bits(dtype)),
    };
}

pub fn inferDType(comptime value: anytype) DType {
    return switch (@typeInfo(@TypeOf(value))) {
        .Bool => .bool_literal,
        .ComptimeInt => .int_literal,
        .ComptimeFloat => .float_literal,
        .Int, .Float => @field(DType, @typeName(@TypeOf(value))),
        else => @compileError(@typeName(@TypeOf(value)) ++ " is not a valid tensor element type"),
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
    const is_comptime1 = isLiteral(dtype1);
    const is_comptime2 = isLiteral(dtype2);

    if (is_bool1 and is_float2 or is_float1 and is_bool2) {
        @compileError("Cannot combine a float and a bool");
    }

    if (is_bool1 and is_bool2) {
        // bool, bool -> bool
        return .bool;
    } else if ((is_bool1 and is_int2) or (is_int1 and is_float2)) {
        // bool, int -> int
        // int, float -> float
        if (is_comptime2 and is_float2) {
            return default_float;
        } else if (is_comptime2 and is_int2) {
            return default_int;
        }
        return dtype2;
    } else if ((is_int1 and is_bool2) or (is_float1 and is_int2)) {
        // int, bool -> int
        // float, int -> float
        if (is_comptime1 and is_float1) {
            return default_float;
        } else if (is_comptime1 and is_int1) {
            return default_int;
        }
        return dtype1;
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

    @compileLog(dtype1, dtype2);
    unreachable;
}

/// When a float data type is necessary (e.g.  for division) this function can be used
/// It will always return a float dtype, either the default (f32) or a satisfying one from dtype1 and dtype2
pub fn floatResultDType(dtype1: DType, dtype2: DType) DType {
    const result = resultDType(dtype1, dtype2);
    return if (isFloat(result)) result else default_float;
}

pub fn FloatTensor(comptime T: type) type {
    std.debug.assert(tensor.isTensor(T));
    const Tensor = tensor.Tensor(T.dtype, T.ndims, T.shape);
    if (!isFloat(Tensor.dtype)) {
        return tensor.Tensor(default_float, Tensor.ndims, Tensor.shape);
    } else {
        return Tensor;
    }
}

pub fn BoolTensor(comptime T: type) type {
    std.debug.assert(tensor.isTensor(T));
    const Tensor = tensor.Tensor(T.dtype, T.ndims, T.shape);
    if (!isBool(Tensor.dtype)) {
        @compileError("Must be bool datatype");
    } else {
        return Tensor;
    }
}

pub fn IntTensor(comptime T: type) type {
    std.debug.assert(tensor.isTensor(T));
    const Tensor = tensor.Tensor(T.dtype, T.ndims, T.shape);
    if (!isInt(Tensor.dtype)) {
        @compileError("Must cast to int datatype first");
    } else {
        return Tensor;
    }
}
