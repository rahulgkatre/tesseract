const dtypes = @import("dtypes.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;

// Arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const UnaryOp = enum {
    pub const Info = struct {
        op: UnaryOp,
        in: [1]*const AnyTensor,
    };
    pub const Args = void;
    pub const Json = struct {
        op: UnaryOp,
        in: [1]usize,
        out: usize,
    };

    Id,
    Neg,
    Log2,
    Exp2,
    Sqrt,
    Rcp,
    Sin,
};
// Lt, Eq, Xor will produce a bool tensor which can be used in mask based operations later on
pub const BinaryOp = enum {
    pub const Info = struct {
        op: BinaryOp,
        in: [2]*const AnyTensor,
    };
    pub const Args = void;
    pub const Json = struct {
        op: BinaryOp,
        in: [2]usize,
        out: usize,
    };

    Add,
    Mul,
    Max,
    Mod,
    Lt,
    Eq,
    Xor,
};
// Ternary ops take in 3 arguments which can have different purposes
pub const TernaryOp = enum {
    pub const Info = struct {
        op: TernaryOp,
        in: [3]*const AnyTensor,
    };
    pub const Args = void;
    pub const Json = struct {
        op: TernaryOp,
        in: [3]usize,
        out: usize,
    };
    Where,
};
// ReduceOps are just recurrently applied binary ops
pub const ReduceOp = enum {
    pub const Info = struct {
        op: ReduceOp,
        in: [1]*const AnyTensor,
        args: Args,
    };
    pub const Args = struct {
        dims: []const u16,
        mask: []const bool,
    };
    pub const Json = struct {
        op: ReduceOp,
        in: [1]usize,
        args: Args,
        out: usize,
    };

    Add,
    Mul,
    Max,
    Xor,

    pub fn binaryOp(comptime reduceOp: ReduceOp) BinaryOp {
        return @field(BinaryOp, @tagName(reduceOp));
    }
};
// Buffer ops do not have runtime dependencies as they are consumed by the code generator
pub const BufferOp = enum {
    pub const Info = struct {
        op: BufferOp,
        in: [1]*const AnyTensor,
    };
    pub const Args = void;
    pub const Json = struct {
        op: BufferOp,
        in: [1]usize,
        out: usize,
    };

    View,
    Cast,
    Pad,
    Expand,
    Shrink,
    Contiguous,
};
pub const InitOp = enum {
    pub const Info = struct {
        op: InitOp,
        args: InitOp.Args,
    };
    pub const Args = union(InitOp) {
        Empty: void,
        Input: void,
        Parameter: void,
        Full: struct {
            value: []const u8,
        },
        Rand: void,
        Range: struct {
            start: []const u8,
            stop: []const u8,
        },
    };
    pub const Json = struct {
        op: InitOp,
        args: Args,
        out: usize,
    };
    Empty,
    Input,
    Parameter,
    Full,
    Rand,
    Range,
};
pub const OpTypes = enum { UnaryOp, BinaryOp, ReduceOp, BufferOp, InitOp, TernaryOp };
