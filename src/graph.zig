const ops = @import("ops.zig");
const std = @import("std");

// TODO: Add a backend pointer here
pub const GraphTensor = struct {
    const Self = @This();
    debug_info_fn: *const fn (ptr: *const Self) void,
    eval_map_fn: *const fn (ptr: *const Self, op_call: ops.OpCall) void,
    eval_zip_fn: *const fn (ptr: *const Self, op_call: ops.OpCall) void,
    eval_reduce_fn: *const fn (ptr: *const Self, op_call: ops.OpCall) void,
    last_op: ?ops.OpCall = null,
    // TODO: Remove after debugging is done
    pub fn debug_info(self: *const Self) void {
        self.debug_info_fn(self);
    }
    pub fn eval_map(self: *const Self, op_call: ops.OpCall) void {
        return self.eval_map_fn(self, op_call);
    }
    pub fn eval_zip(self: *const Self, op_call: ops.OpCall) void {
        return self.eval_zip_fn(self, op_call);
    }
    pub fn eval_reduce(self: *const Self, op_call: ops.OpCall) void {
        return self.eval_reduce_fn(self, op_call);
    }
    // TODO: Remove after debugging is done
    pub fn debug_graph(self: *const Self) void {
        std.debug.print("\ncurrent: ", .{});
        self.debug_info();

        if (self.last_op != null) {
            switch (self.last_op.?) {
                .MapOp => |op_call| {
                    std.debug.print("\n\top: {any}\n\tinput: ", .{op_call.op});
                    op_call.a.debug_info();
                    std.debug.print("\n", .{});
                    op_call.a.debug_graph();
                },
                .ZipOp => |op_call| {
                    std.debug.print("\n\top: {any}\n\tinput1: ", .{op_call.op});
                    op_call.a.debug_info();
                    std.debug.print("\n\tinput2: ", .{});
                    op_call.b.debug_info();
                    std.debug.print("\n", .{});
                    op_call.a.debug_graph();
                    op_call.b.debug_graph();
                },
                .ReduceOp => |op_call| {
                    std.debug.print("\n\top: {any}\n\tdim: {any}\n\tinput: ", .{ op_call.op, op_call.reduce_dim });
                    op_call.a.debug_info();
                    std.debug.print("\n", .{});
                    op_call.a.debug_graph();
                },
            }
        } else {
            std.debug.print("\n\treached leaf in graph\n", .{});
        }
    }
};
