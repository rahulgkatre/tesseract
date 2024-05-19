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
            try write_stream.write(.{ .op = self.op, .args = self.args });
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
                .a = @intFromPtr(op_tracker.a),
                .out = @intFromPtr(out),
            } },
            .BinaryOp => |op_tracker| .{ .BinaryOp = .{
                .op = op_tracker.op,
                .a = @intFromPtr(op_tracker.a),
                .b = @intFromPtr(op_tracker.b),
                .out = @intFromPtr(out),
            } },
            .ReduceOp => |op_tracker| .{ .ReduceOp = .{
                .op = op_tracker.op,
                .a = @intFromPtr(op_tracker.a),
                .dims = op_tracker.dims,
                .out = @intFromPtr(out),
            } },
            .ArrayOp => |op_tracker| .{ .ArrayOp = .{
                .op = op_tracker.op,
                .a = @intFromPtr(op_tracker.a),
                .out = @intFromPtr(out),
            } },
            .InitOp => |op_tracker| .{ .InitOp = .{
                .op = op_tracker.op,
                .args = op_tracker.args,
                .out = @intFromPtr(out),
            } },
            .TernaryOp => |op_tracker| .{ .TernaryOp = .{
                .op = op_tracker.op,
                .a = @intFromPtr(op_tracker.a),
                .b = @intFromPtr(op_tracker.b),
                .c = @intFromPtr(op_tracker.c),
                .out = @intFromPtr(out),
            } },
        };
    }

    pub const JsonFormat = union(ops.OpTypes) {
        UnaryOp: struct {
            op: ops.UnaryOp,
            a: usize,
            out: usize,
        },
        BinaryOp: struct {
            op: ops.BinaryOp,
            a: usize,
            b: usize,
            out: usize,
        },
        ReduceOp: struct {
            op: ops.ReduceOp,
            a: usize,
            dims: []const bool,
            out: usize,
        },
        ArrayOp: struct {
            op: ops.ArrayOp,
            a: usize,
            out: usize,
        },
        InitOp: struct {
            op: ops.InitOp,
            args: ops.InitOp.Args,
            out: usize,
        },
        TernaryOp: struct {
            op: ops.TernaryOp,
            a: usize,
            b: usize,
            c: usize,
            out: usize,
        },
    };
};

/// Structure for tracking groups of operations (i.e. functions)
/// This is used for autodifferentiation, where a simpler operation expression can be written
/// instead of a more complicated expression derived through the backwards pass
/// Group name should always correspond to a function name so that a corresponding
/// gradient function "d_<function name>" and be looked up
pub const OpGroupTracker = extern struct {
    pub const OpGroup = struct {
        name: []const u8,
        outer: ?*const OpGroup,
        id: usize,
    };

    curr: ?*const OpGroup = null,
    next: ?*const OpGroup = null,
    next_id: usize = 0,

    pub fn startGroup(self: OpGroupTracker, name: []const u8) OpGroupTracker {
        return .{
            .curr = self.curr,
            .next = &OpGroup{
                .name = name,
                .outer = self.next,
                .id = self.next_id,
            },
            .next_id = self.next_id + 1,
        };
    }

    pub fn joinGroup(self: OpGroupTracker, group: ?*const OpGroup) OpGroupTracker {
        return .{
            .curr = self.curr,
            .next = group,
            .next_id = if (group) |g| @max(self.next_id + 1, g.id + 1) else self.next_id + 1,
        };
    }

    pub fn keepGroup(self: OpGroupTracker) OpGroupTracker {
        return .{
            .curr = self.next,
            .next = self.next,
            .next_id = self.next_id + 1,
        };
    }

    pub fn endGroup(self: OpGroupTracker) OpGroupTracker {
        return .{
            .curr = self.curr,
            // Next group is the outer one (end of the function scope)
            .next = if (self.next) |next| next.outer else @compileError("No op group to end"),
            .next_id = self.next_id + 1,
        };
    }
};
