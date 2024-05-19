const ops = @import("ops.zig");
const std = @import("std");
const AnyTensor = @import("anytensor.zig").AnyTensor;

pub const OpTracker = union(ops.OpTypes) {
    UnaryOp: struct {
        op: ops.UnaryOp,
        a: *const AnyTensor,
    },
    BinaryOp: struct {
        op: ops.BinaryOp,
        a: *const AnyTensor,
        b: *const AnyTensor,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .a = @intFromPtr(self.a), .b = @intFromPtr(self.b) });
        }
    },
    ReduceOp: struct {
        op: ops.ReduceOp,
        a: *const AnyTensor,
        dims: []const bool,
    },
    ArrayOp: struct {
        op: ops.ArrayOp,
        a: *const AnyTensor,
        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .a = @intFromPtr(self.a) });
        }
    },
    InitOp: struct {
        op: ops.InitOp,
        args: ops.InitOp.Args,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .args = @intFromPtr(self.args) });
        }
    },
    TernaryOp: struct {
        op: ops.TernaryOp,
        a: *const AnyTensor,
        b: *const AnyTensor,
        c: *const AnyTensor,

        pub fn jsonStringify(self: @This(), write_stream: anytype) !void {
            try write_stream.write(.{ .op = self.op, .a = @intFromPtr(self.a), .b = @intFromPtr(self.b), .c = @intFromPtr(self.c) });
        }
    },

    pub fn init(
        comptime tag: ops.OpTypes,
        comptime op: @field(ops, @tagName(tag)),
        inputs: switch (tag) {
            .TernaryOp => [3]*const AnyTensor,
            .BinaryOp => [2]*const AnyTensor,
            .UnaryOp, .ArrayOp, .ReduceOp => [1]*const AnyTensor,
            .InitOp => void,
        },
        args: switch (tag) {
            .ReduceOp => []const bool,
            .InitOp => ops.InitOp.Args,
            else => void,
        },
    ) OpTracker {
        return @unionInit(OpTracker, @tagName(tag), switch (tag) {
            .TernaryOp => .{
                .op = op,
                .a = inputs[0],
                .b = inputs[1],
                .c = inputs[2],
            },
            .BinaryOp => .{
                .op = op,
                .a = inputs[0],
                .b = inputs[1],
            },
            .ReduceOp => .{
                .op = op,
                .a = inputs[0],
                .dims = args,
            },
            .UnaryOp, .ArrayOp => .{
                .op = op,
                .a = inputs[0],
            },
            .InitOp => .{
                .op = op,
                .args = args,
            },
        });
    }

    pub fn toJsonFormat(self: *const OpTracker, out: *const AnyTensor) JsonFormat {
        return switch (self.*) {
            .UnaryOp => |op_tracker| .{ .UnaryOp = .{
                .op = op_tracker.op,
                .a = op_tracker.a,
                .out = @intFromPtr(out),
            } },
            .BinaryOp => |op_tracker| .{ .BinaryOp = .{
                .op = op_tracker.op,
                .a = op_tracker.a,
                .b = op_tracker.b,
                .out = @intFromPtr(out),
            } },
            .ReduceOp => |op_tracker| .{ .ReduceOp = .{
                .op = op_tracker.op,
                .a = op_tracker.a,
                .dims = op_tracker.dims,
                .out = @intFromPtr(out),
            } },
            .ArrayOp => |op_tracker| .{ .ArrayOp = .{
                .op = op_tracker.op,
                .a = op_tracker.a,
                .out = @intFromPtr(out),
            } },
            .InitOp => |op_tracker| .{ .InitOp = .{
                .op = op_tracker.op,
                .args = op_tracker.args,
                .out = @intFromPtr(out),
            } },
            .TernaryOp => |op_tracker| .{ .TernaryOp = .{
                .op = op_tracker.op,
                .a = op_tracker.a,
                .b = op_tracker.b,
                .c = op_tracker.c,
                .out = @intFromPtr(out),
            } },
        };
    }

    pub const JsonFormat = union(ops.OpTypes) {
        UnaryOp: struct {
            op: ops.UnaryOp,
            a: *const AnyTensor,
            out: usize,
        },
        BinaryOp: struct {
            op: ops.BinaryOp,
            a: *const AnyTensor,
            b: *const AnyTensor,
            out: usize,
        },
        ReduceOp: struct {
            op: ops.ReduceOp,
            a: *const AnyTensor,
            dims: []const bool,
            out: usize,
        },
        ArrayOp: struct {
            op: ops.ArrayOp,
            a: *const AnyTensor,
            out: usize,
        },
        InitOp: struct {
            op: ops.InitOp,
            args: ops.InitOp.Args,
            out: usize,
        },
        TernaryOp: struct {
            op: ops.TernaryOp,
            a: *const AnyTensor,
            b: *const AnyTensor,
            c: *const AnyTensor,
            out: usize,
        },
    };
};

pub const BlockTracker = extern struct {
    pub const Block = struct {
        name: []const u8,
        outer: ?*const Block,
        id: usize,
    };

    curr_block: ?*const Block = null,
    next_block: ?*const Block = null,
    next_id: usize = 0,

    pub fn enter(bt: BlockTracker, block_name: []const u8) BlockTracker {
        return .{
            .curr_block = bt.curr_block,
            .next_block = &Block{
                .name = block_name,
                .outer = bt.next_block,
                .id = bt.next_id,
            },
            .next_id = bt.next_id + 1,
        };
    }

    pub fn join(bt: BlockTracker, block: ?*const Block) BlockTracker {
        return .{
            .curr_block = bt.curr_block,
            .next_block = block,
            .next_id = if (block) |b| @max(bt.next_id + 1, b.id) else bt.next_id + 1,
        };
    }

    pub fn update(bt: BlockTracker) BlockTracker {
        return .{
            .curr_block = bt.next_block,
            .next_block = bt.next_block,
            .next_id = bt.next_id + 1,
        };
    }

    pub fn leave(bt: BlockTracker) BlockTracker {
        return .{
            .curr_block = bt.curr_block,
            .next_block = if (bt.next_block) |next_block| next_block.outer else @compileError("No block to end"),
            .next_id = bt.next_id + 1,
        };
    }
};
