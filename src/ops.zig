const AnyTensor = @import("anytensor.zig").AnyTensor;

// Arithmetic operations for unary functions, binary functions,
// and reducing a dimension of a tensor to a single value by applying some binary function
pub const UnaryOp = enum {
    pub const Instr = struct {
        op: UnaryOp,
        src: [1]*const AnyTensor,
        args: Args = {},
    };
    pub const Args = void;
    pub const Json = struct {
        op: UnaryOp,
        src: [1]usize,
        dst: usize,
    };

    neg,
    log2,
    exp2,
    sqrt,
    recip,
    sin,
};
// Lt, Eq, Xor will produce a bool tensor which can be used in mask based operations later on
pub const BinaryOp = enum {
    pub const Instr = struct {
        op: BinaryOp,
        src: [2]*const AnyTensor,
        args: Args = {},
    };
    pub const Args = void;
    pub const Json = struct {
        op: BinaryOp,
        src: [2]usize,
        dst: usize,
    };

    add,
    mul,
    max,
    mod,
    less_than,
    equals,
    xor,
};
// Ternary ops take in 3 arguments which can have different purposes
pub const TernaryOp = enum {
    pub const Instr = struct {
        op: TernaryOp,
        src: [3]*const AnyTensor,
        args: Args = {},
    };
    pub const Args = void;
    pub const Json = struct {
        op: TernaryOp,
        src: [3]usize,
        dst: usize,
    };
    where,
};
// ReduceOps are just recurrently applied binary ops
pub const ReduceOp = enum {
    pub const Instr = struct {
        op: ReduceOp,
        src: [1]*const AnyTensor,
        args: Args,
    };
    pub const Args = struct {
        dims: []const u16,
        mask: []const bool,
    };
    pub const Json = struct {
        op: ReduceOp,
        src: [1]usize,
        args: Args,
        dst: usize,
    };

    add,
    mul,
    max,
    xor,

    pub fn binaryOp(reduceOp: ReduceOp) BinaryOp {
        return @field(BinaryOp, @tagName(reduceOp));
    }
};
// TypeOps mutate the type of the tensor, in Tesseract's case this not only changes
// the dtype but also the shape, so any shape affecting ops are TypeOps

pub const TypeOp = enum {
    pub const Instr = struct {
        op: TypeOp,
        src: [1]*const AnyTensor,
        args: Args,
    };
    pub const Args = union(TypeOp) {
        pub const Pad = struct {
            pub const Mode = enum {
                constant,
                reflect,
                replicate,
                circular,
            };

            padding: []const [2]u64,
            mode: union(Mode) {
                constant: []const u8,
                reflect: void,
                replicate: void,
                circular: void,
            },
        };
        view: void,
        cast: void,
        pad: Pad,
        contiguous: void,
    };
    pub const Json = struct {
        op: TypeOp,
        src: [1]usize,
        dst: usize,
    };

    view,
    cast,
    pad,
    contiguous,
};
pub const InitOp = enum {
    pub const Instr = struct {
        op: InitOp,
        src: [0]*const AnyTensor = .{},
        args: InitOp.Args,
    };
    pub const Args = union(InitOp) {
        pub const full = []const u8;
        pub const range = struct {
            start: []const u8,
            stop: []const u8,
        };

        empty: void,
        input: void,
        parameter: void,
        full: full,
        random: void,
        range: range,
    };
    pub const Json = struct {
        op: InitOp,
        args: Args,
        dst: usize,
    };
    empty,
    input,
    parameter,
    full,
    random,
    range,
};
pub const OpTypes = enum {
    InitOp,
    UnaryOp,
    BinaryOp,
    TernaryOp,
    ReduceOp,
    TypeOp,
};

pub const Instruction = union(OpTypes) {
    InitOp: InitOp.Instr,
    UnaryOp: UnaryOp.Instr,
    BinaryOp: BinaryOp.Instr,
    TernaryOp: TernaryOp.Instr,
    ReduceOp: ReduceOp.Instr,
    TypeOp: TypeOp.Instr,

    pub const Json = union(OpTypes) {
        InitOp: InitOp.Json,
        UnaryOp: UnaryOp.Json,
        BinaryOp: BinaryOp.Json,
        TernaryOp: TernaryOp.Json,
        ReduceOp: ReduceOp.Json,
        TypeOp: TypeOp.Json,
    };
};
