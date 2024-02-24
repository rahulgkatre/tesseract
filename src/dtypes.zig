pub const DType = enum {
    u1,
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

pub fn isFloat(t: DType) bool {
    return switch (t) {
        .f16, .f32, .f64, .f128 => true,
        else => false,
    };
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
        .u1, .bool => 1,
        .u8, .i8 => 8,
        .u16, .i16, .f16 => 16,
        .u32, .i32, .f32 => 32,
        .u64, .i64, .f64 => 64,
        .u128, .i128, .f128 => 128,
    };
}
