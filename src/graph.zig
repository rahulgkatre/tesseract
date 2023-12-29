const ops = @import("ops.zig");
const std = @import("std");

pub const GraphTensor = struct {
    const Self = @This();
    // permute_fn: *const fn (comptime ptr: *const Self, comptime perm: []u8) Self,
    print_info_fn: *const fn (comptime ptr: *const Self) void,
    map_fn: *const fn (comptime ptr: *const Self, comptime map_op: ops.MapOp) Self,
    zip_fn: *const fn (comptime ptr: *const Self, comptime zip_op: ops.ZipOp, comptime other_ptr: anytype) Self,
    reduce_fn: *const fn (comptime ptr: *const Self, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: u8) Self,
    history: ?History,
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
    pub fn zip(comptime self: *const Self, comptime zip_op: ops.ZipOp, comptime other_ptr: anytype) Self {
        return self.zip_fn(self, zip_op, other_ptr);
    }
    pub fn reduce(comptime self: *const Self, comptime reduce_op: ops.ReduceOp, comptime reduce_dim: u8) Self {
        return self.reduce_fn(self, reduce_op, reduce_dim);
    }
    // TODO: Remove after debugging is done
    pub fn print_graph(comptime self: *const Self) void {
        std.debug.print("\ncurrent: ", .{});
        self.print_info();

        if (self.history != null) {
            switch (self.history.?.op) {
                .MapOp => |op| {
                    const t1 = self.history.?.args.MapOp.self_ptr;
                    std.debug.print("\n\top: {any}\n\tinput:", .{op});
                    t1.print_info();
                    std.debug.print("\n", .{});
                    t1.print_graph();
                },
                .ZipOp => |op| {
                    const t1 = self.history.?.args.ZipOp.self_ptr;
                    const t2 = self.history.?.args.ZipOp.other_ptr;
                    std.debug.print("\n\top: {any}\n\tinput1:", .{op});
                    t1.print_info();
                    std.debug.print("\n\tinput2:", .{});
                    t2.print_info();
                    std.debug.print("\n", .{});
                    t1.print_graph();
                    t2.print_graph();
                },
                .ReduceOp => |op| {
                    const t1 = self.history.?.args.ReduceOp.self_ptr;
                    const rd = self.history.?.args.ReduceOp.reduce_dim;
                    std.debug.print("\n\top: {any}\n\tdim: {any}\n\tinput:", .{ op, rd });
                    t1.print_info();
                    std.debug.print("\n", .{});
                    t1.print_graph();
                },
            }
        } else {
            std.debug.print("\n\treached leaf in graph\n", .{});
        }
    }
};

const History = struct { op: ops.Op, args: ops.OpArgs };
