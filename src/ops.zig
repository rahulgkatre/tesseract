const DType = @import("dtypes.zig").DType;
// Map, Zip and Reduce ops are arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const MapOp = enum { Id, Neg, Log, Exp, Sqrt, Recip, Sin };
pub const ZipOp = enum { Add, Mul, Maximum, Mod, LessThan, Equals, Xor };
pub const ReduceOp = enum { Sum, Max };
pub const TypeOp = enum {
    AsStrided,
    AsType,
};
pub const InitOp = enum {
    pub const Args = union(InitOp) {
        Input: void,
        Full: []const u8,
        Rand: DType,
        Range: struct {
            start: []const u8,
            stop: []const u8,
        },
    };

    Input,
    Full,
    Rand,
    Range,
};

pub const MemOps = enum {
    Load, // Load data
    Store, // Store data
    Move, // Move between devices
};

pub const OpTypes = enum { MapOp, ZipOp, ReduceOp, TypeOp, InitOp };
