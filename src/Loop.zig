const ops = @import("ops.zig");

const Expr = union(ops.OpTypes) {
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

// Abstractions for lowering Graph.Node into a Loop which can be codegened for a specific language
// Loop structs will be stored in a list where order is exact order of code
// Loops are defined as a grammar, ever loop has a header and a body
pub const Loop = struct {
    header: Header,
    body: Body,
};

// Loop header defines the bounds of the loop and the loop variable
// Loop variable will almost always be i_X where X is a number so just store the id
const Header = struct {
    lower_bound: usize,
    upper_bound: usize,
    loop_var_id: usize,
};

// Loop body can either be another loop (normal or accumulating) or an expression
// Expression can just reuse Graph.Link as it has access to all needed information
const Body = union(enum) {
    InnerLoop: Loop,
    InnerAccLoop: AccLoop,
    Expr: []Expr,
};

// Accumulating loop is special because it needs an accumulator (var acc_X)
// To accumulate over multiple dimensions it can also be nested with inner loops
const AccLoop = struct {
    header: Header,
    body: Body,
    acc_var_id: usize,
};
