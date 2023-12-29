const ops = @import("ops.zig");
const std = @import("std");


pub const GraphTensor = struct {
    const Self = @This();
    // permute_fn: *const fn (comptime ptr: *const Self, comptime perm: []u8) Self,
    print_info_fn: *const fn (comptime ptr: *const Self) void,
    map_fn: *const fn (comptime ptr: *const Self, comptime map_op: ops.MapOp) Self,
    zip_fn: *const fn (comptime ptr: *const Self, comptime zip_op: ops.ZipOp, comptime b: anytype) Self,
    reduce_fn: *const fn (comptime ptr: *const Self, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: u8) Self,
    last_op: ?ops.OpCall = null,
    // pub fn permute(comptime self: *const Self, comptime perm :[]u8) Self {
    //     return self.permute_fn(self, perm);
    // }
    // TODO: Remove after debugging is done
    pub fn print_info(comptime self: *const Self) void {
        self.print_info_fn(self);
    }
    pub fn map(comptime self: *const Self, comptime map_op: ops.MapOp) Self {
        return self.map_fn(self, map_op);
    }
    pub fn zip(comptime self: *const Self, comptime zip_op: ops.ZipOp, comptime other: anytype) Self {
        return self.zip_fn(self, zip_op, other);
    }
    pub fn reduce(comptime self: *const Self, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: u8) Self {
        return self.reduce_fn(self, reduce_op, reduce_dim);
    }
    // TODO: Remove after debugging is done
    pub fn print_graph(comptime self: *const Self) void {
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

