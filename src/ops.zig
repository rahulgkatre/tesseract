const ops = @This();
const dtypes = @import("dtypes.zig");
// Arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const UnaryOp = enum { Id, Neg, Log2, Exp2, Sqrt, Rcp, Sin };
// Lt, Eq, Xor will produce a bool tensor which can be used in mask based operations later on
pub const BinaryOp = enum { Add, Mul, Max, Mod, Lt, Eq, Xor };
// Ternary ops take in 3 arguments which can have different purposes
pub const TernaryOp = enum { Where };
// ReduceOps are just recurrently applied binary ops
pub const ReduceOp = enum {
    Add,
    Mul,
    Max,
    Xor,

    pub fn binaryOp(comptime reduceOp: ReduceOp) BinaryOp {
        return @field(BinaryOp, @tagName(reduceOp));
    }
};
// Array ops do not have runtime dependencies as they are consumed by the code generator
pub const ArrayOp = enum { View, Cast, Pad, Expand, Shrink, Contiguous };
pub const InitOp = enum {
    pub const Args = union(InitOp) {
        Empty: void,
        Input: void,
        Parameter: void,
        Full: []const u8,
        Rand: void,
        Range: struct {
            start: []const u8,
            stop: []const u8,
        },
    };
    Empty,
    Input,
    Parameter,
    Full,
    Rand,
    Range,
};
pub const OpTypes = enum { UnaryOp, BinaryOp, ReduceOp, ArrayOp, InitOp, TernaryOp };
