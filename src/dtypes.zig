pub const DType = enum {
    i8,
    i16,
    i32,
    i64,
    f16,
    f32,
    f64,
    bool,
};

pub fn isFloat(t: DType) bool {
    return switch (t) {
        .f16, .f32, .f64 => true,
        else => false,
    };
}

pub fn isInt(t: DType) bool {
    return switch (t) {
        .i8, .i16, .i32, .i64 => true,
        else => false,
    };
}

pub fn isBool(t: DType) bool {
    return switch (t) {
        .bool => true,
        else => false,
    };
}
