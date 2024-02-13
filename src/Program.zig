const ops = @import("ops.zig");
const std = @import("std");
/// Expression in the body of the loop of the form y = f(x)
/// y can either be a location in an array or a temporary variable
pub const Expr = union(ops.GraphOps) {
    MapOp: struct {
        op: ops.MapOp,
        x_id: usize,
        x_strides: []const usize,
        out_id: usize,
        out_strides: []const usize,

        pub fn log(self: *const @This()) void {
            std.debug.print("\t" ** 0 ++ "T{d} = {s}(T{d})\n", .{ self.out_id, @tagName(self.op), self.x_id });
        }
    },
    ZipOp: struct {
        op: ops.ZipOp,
        a_id: usize,
        a_strides: []const usize,
        b_id: usize,
        b_strides: []const usize,
        out_id: usize,
        out_strides: []const usize,

        pub fn log(self: *const @This()) void {
            std.debug.print("\t" ** 0 ++ "T{d} = {s}(T{d}, T{d})\n", .{ self.out_id, @tagName(self.op), self.a_id, self.b_id });
        }
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        x_id: usize,
        x_strides: []const usize,
        out_id: usize,
        out_strides: []const usize,

        pub fn log(self: *const @This()) void {
            std.debug.print("\t" ** 0 ++ "T{d} = {s}(T{d})\n", .{ self.out_id, @tagName(self.op), self.x_id });
        }
    },
    TypeOp: struct {
        op: ops.TypeOp,
        x_id: usize,
        x_strides: []const usize,
        out_id: usize,
        out_strides: []const usize,

        pub fn log(self: *const @This()) void {
            std.debug.print("\t" ** 0 ++ "T{d} = {s}(T{d})\n", .{ self.out_id, @tagName(self.op), self.x_id });
        }
    },
    InitOp: struct {
        op: ops.InitOp,

        pub fn log(_: *const @This()) void {}
    },
};

/// Abstractions for lowering Graph.Node into a loop which can be codegened
/// loop structs will be stored in a list (program) where order is exact order of code
/// loops are defined as a grammar, every loop has a header and a body
pub const AffineLoop = struct {
    upper_bound: usize,
    loop_var: []const u8,
    acc: bool = false,
    body: LoopBody,
    prev: ?*AffineLoop,

    pub fn log(self: *const AffineLoop) void {
        if (self.prev != null) {
            self.prev.?.log();
        }
        std.debug.print("\t" ** 0 ++ "for (0..{d}) |{s}| {{\n", .{ self.upper_bound, self.loop_var });
        self.body.log();
        std.debug.print("\t" ** 0 ++ "}}\n", .{});
    }
};

/// Affine loop body can either be another loop (normal or accumulating) or an expression
/// Expression can just reuse Graph.Link as it has access to all needed information
const LoopBody = struct {
    inner_loops: std.ArrayList(*AffineLoop),
    exprs: std.ArrayList(Expr),

    pub fn log(self: *const LoopBody) void {
        for (self.inner_loops.items) |loop| {
            loop.log();
        }
        for (self.exprs.items) |expr| {
            switch (expr) {
                inline else => |expr_| expr_.log(),
            }
        }
    }
};
