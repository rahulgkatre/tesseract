const ops = @import("ops.zig");
const std = @import("std");

pub const GraphTensor = struct {
    const Self = @This();
    // permute_fn: *const fn (comptime ptr: *const Self, comptime perm: []u8) Self,
    // tensor_type: type,
    print_info_fn: *const fn (ptr: *const Self) void,
    eval_map: *const fn (ptr: *const Self, op_call: ops.OpCall) void,
    eval_zip: *const fn (ptr: *const Self, op_call: ops.OpCall) void,
    eval_reduce: *const fn (ptr: *const Self, op_call: ops.OpCall) void,
    last_op: ?ops.OpCall = null,
    // TODO: Remove after debugging is done
    pub fn print_info(self: *const Self) void {
        self.print_info_fn(self);
    }
    pub fn map(self: *const Self, op_call: ops.OpCall) void {
        return self.eval_map(self, op_call);
    }
    pub fn zip(self: *const Self, op_call: ops.OpCall) void {
        return self.eval_zip(self, op_call);
    }
    pub fn reduce(self: *const Self, op_call: ops.OpCall) void {
        return self.eval_reduce(self, op_call);
    }
    // TODO: Remove after debugging is done
    pub fn print_graph(self: *const Self) void {
        std.debug.print("\ncurrent: ", .{});
        self.print_info();

        if (self.last_op != null) {
            switch (self.last_op.?) {
                .MapOp => |op_call| {
                    std.debug.print("\n\top: {any}\n\tinput: ", .{op_call.op});
                    op_call.a.print_info();
                    std.debug.print("\n", .{});
                    op_call.a.print_graph();
                },
                .ZipOp => |op_call| {
                    std.debug.print("\n\top: {any}\n\tinput1: ", .{op_call.op});
                    op_call.a.print_info();
                    std.debug.print("\n\tinput2: ", .{});
                    op_call.b.print_info();
                    std.debug.print("\n", .{});
                    op_call.a.print_graph();
                    op_call.b.print_graph();
                },
                .ReduceOp => |op_call| {
                    std.debug.print("\n\top: {any}\n\tdim: {any}\n\tinput: ", .{ op_call.op, op_call.reduce_dim });
                    op_call.a.print_info();
                    std.debug.print("\n", .{});
                    op_call.a.print_graph();
                },
            }
        } else {
            std.debug.print("\n\treached leaf in graph\n", .{});
        }
    }
};
