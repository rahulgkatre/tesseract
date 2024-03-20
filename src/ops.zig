const dtypes = @import("dtypes.zig");
// Map, Zip and Reduce ops are arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const MapOp = enum { Id, Neg, Log, Exp, Sqrt, Recip, Sin };
// LessThan, Equals, Xor will produce a bool tensor which can be used in mask based operations later on
pub const ZipOp = enum { Add, Mul, Maximum, Mod, LessThan, Equals, Xor };
pub const ReduceOp = enum { Sum, Max };
pub const TypeOp = enum { AsStrided, AsType };
pub const InitOp = enum {
    pub const Args = union(InitOp) {
        Input: void,
        Full: []const u8,
        Rand: void,
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
// Where is an if statement that takes in a mask boolean (the operand tensor), a true branch value, and a false branch value
pub const TernaryOp = enum { Where };
pub const OpTypes = enum { MapOp, ZipOp, ReduceOp, TypeOp, InitOp, TernaryOp };
