const dtypes = @import("dtypes.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;

// Arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const UnaryOp = enum {
    pub const Info = struct {
        op: UnaryOp,
        in: [1]*const AnyTensor,
        args: Args = {},
    };
    pub const Args = void;
    pub const Json = struct {
        op: UnaryOp,
        in: [1]usize,
        out: usize,
    };

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
        args: Args = {},
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
        args: Args = {},
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
// TypeOps mutate the type of the tensor, in Tesseract's case this not only changes
// the dtype but also the shape, so any shape affecting ops are TypeOps

pub const PadModeEnum = enum { constant, reflect, replicate, circular };

pub const TypeOp = enum {
    pub const Info = struct {
        op: TypeOp,
        in: [1]*const AnyTensor,
        args: Args,
    };
    pub const Args = union(TypeOp) {
        View: void,
        Cast: void,
        Pad: struct {
            padding: []const [2]u64,
            mode: union(PadModeEnum) {
                constant: []const u8,
                reflect: void,
                replicate: void,
                circular: void,
            },
        },
        Contiguous: void,
    };
    pub const Json = struct {
        op: TypeOp,
        in: [1]usize,
        out: usize,
    };

    View,
    Cast,
    Pad,
    Contiguous,
};
pub const InitOp = enum {
    pub const Info = struct {
        op: InitOp,
        in: [0]*const AnyTensor,
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
pub const OpTypes = enum { UnaryOp, BinaryOp, ReduceOp, TypeOp, InitOp, TernaryOp };
