const ops = @import("ops.zig");

/// Expression in the body of the loop of the form y = f(x)
/// y can either be a location in an array or a temporary variable
pub const Expr = union(ops.OpTypes) {
    InitOp: struct {
        op: ops.InitOp,
    },
    MapOp: struct {
        op: ops.MapOp,
        x_id: usize,
        x_strides: []const usize,
        out_id: usize,
        out_strides: []const usize,
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a_id: usize,
        a_strides: []const usize,
        b_id: usize,
        b_strides: []const usize,
        out_id: usize,
        out_strides: []const usize,
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x_id: usize,
        x_strides: []const usize,
        out_id: usize,
        out_strides: []const usize,
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x_id: usize,
        x_strides: []const usize,
        out_id: usize,
        out_strides: []const usize,
    },
};

/// Abstractions for lowering Graph.Node into a Loop which can be codegened
/// Loop structs will be stored in a list (program) where order is exact order of code
/// Loops are defined as a grammar, every loop has a header and a body
pub const Loop = struct {
    header: Header,
    acc: bool = false,
    body: Body,
};

/// Loop header defines the upper bound of the loop and the loop variable
/// Lower bound will always be 0
const Header = struct {
    upper_bound: usize,
    loop_var: []const u8,
};

/// Loop body can either be another loop (normal or accumulating) or an expression
/// Expression can just reuse Graph.Link as it has access to all needed information
const Body = union(enum) {
    InnerLoop: Loop,
    Expr: []Expr,
};
