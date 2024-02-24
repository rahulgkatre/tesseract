// Map, Zip and Reduce ops are arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const MapOp = enum { Id, Neg, Log2, Exp2, Sqrt, Recip, Sin };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, LessThan, Equals, Xor };
pub const ReduceOp = enum { Sum, Max };
pub const TypeOp = enum { AsStrided, AsType, View };
pub const InitOp = enum { Input, Full, Rand, Range };

pub const InitValue = union(InitOp) {
    Input: void,
    Full: []const u8,
    Rand: @import("dtypes.zig").DType,
    Range: struct {
        start: []const u8,
        stop: []const u8,
    },
};

pub const MemOps = enum {
    Load, // Load data
    Store, // Store data
    Move, // Move between devices
};

pub const GraphOps = enum {
    MapOp,
    ZipOp,
    ReduceOp,
    TypeOp,
    InitOp,
};
