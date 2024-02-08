const ops = @import("ops.zig");

const Loop = struct {
    var_id: usize,
    upper_bound: usize,
    body: Body,
};

const Body = union(enum) {
    Loop: Loop,
    Expr: Expr,
};

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
