const ops = @import("ops.zig");
const std = @import("std");

pub const GraphTensor = struct {
    const Self = @This();
    // TODO: Can we make the args of these not comptime?
    // permute_fn: *const fn (comptime ptr: *const Self, comptime perm: []u8) Self,
    map_fn: *const fn (comptime ptr: *const Self, comptime map_op: ops.MapOp) Self,
    zip_fn: *const fn (comptime ptr: *const Self, comptime zip_op: ops.ZipOp, comptime other_ptr: anytype) Self,
    reduce_fn: *const fn (comptime ptr: *const Self, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: u8) Self,
    history: ?History,
    // pub fn permute(comptime self: *const Self, comptime perm :[]u8) Self {
    //     return self.permute_fn(self, perm);
    // }
    pub fn map(comptime self: *const Self, comptime map_op: ops.MapOp) Self {
        return self.map_fn(self, map_op);
    }
    pub fn zip(comptime self: *const Self, comptime zip_op: ops.ZipOp, comptime other_ptr: anytype) Self {
        return self.zip_fn(self, zip_op, other_ptr);
    }
    pub fn reduce(comptime self: *const Self, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: u8) Self {
        return self.reduce_fn(self, reduce_op, reduce_dim);
    }
    pub fn print_graph(comptime self: *const Self) void {
        if (self.history != null) {
            switch (self.history.?.op) {
                .MapOp => |op| {
                    const t1 = self.history.?.args.MapOp.self_ptr;
                    std.debug.print("\nop:{any}\ninput:{any}\n", .{op,t1});
                    t1.print_graph();
                },
                .ZipOp => |op| {
                    const t1 = self.history.?.args.ZipOp.self_ptr;
                    const t2 = self.history.?.args.ZipOp.other_ptr;
                    std.debug.print("\nop:{any}\ninput1:{any}\ninput2:{any}\n", .{op,t1,t2});
                    t1.print_graph();
                    t2.print_graph();
                },
                .ReduceOp => |op| {
                    const t1 = self.history.?.args.ReduceOp.self_ptr;
                    const rd = self.history.?.args.ReduceOp.reduce_dim;
                    std.debug.print("\nop:{any}\ninput:{any}\ndim:{any}\n", .{op,t1,rd});
                    t1.print_graph();
                },
            }
        }
    }
};

const History = struct { op: ops.Op, args: ops.OpArgs };

