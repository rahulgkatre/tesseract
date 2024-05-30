const ops = @import("ops.zig");
const std = @import("std");
const AnyTensor = @import("anytensor.zig").AnyTensor;

pub const OpTracker = union(ops.OpTypes) {
    UnaryOp: ops.UnaryOp.Info,
    BinaryOp: ops.BinaryOp.Info,
    ReduceOp: ops.ReduceOp.Info,
    TypeOp: ops.TypeOp.Info,
    InitOp: ops.InitOp.Info,
    TernaryOp: ops.TernaryOp.Info,

    pub fn init(
        comptime tag: ops.OpTypes,
        comptime op: @field(ops, @tagName(tag)),
        in: switch (tag) {
            .TernaryOp => [3]*const AnyTensor,
            .BinaryOp => [2]*const AnyTensor,
            .UnaryOp, .TypeOp, .ReduceOp => [1]*const AnyTensor,
            .InitOp => [0]*const AnyTensor,
        },
        args: @field(@TypeOf(op), "Args"),
    ) OpTracker {
        return @unionInit(
            OpTracker,
            @tagName(tag),
            .{
                .op = op,
                .in = in,
                .args = args,
            },
        );
    }

    pub fn toJson(self: *const OpTracker, out: *const AnyTensor) Json {
        return switch (self.*) {
            .UnaryOp => |info| .{ .UnaryOp = .{
                .op = info.op,
                .in = .{@intFromPtr(info.in[0])},
                .out = @intFromPtr(out),
            } },
            .BinaryOp => |info| .{ .BinaryOp = .{
                .op = info.op,
                .in = .{ @intFromPtr(info.in[0]), @intFromPtr(info.in[1]) },
                .out = @intFromPtr(out),
            } },
            .TernaryOp => |info| .{ .TernaryOp = .{
                .op = info.op,
                .in = .{ @intFromPtr(info.in[0]), @intFromPtr(info.in[1]), @intFromPtr(info.in[2]) },
                .out = @intFromPtr(out),
            } },
            .ReduceOp => |info| .{ .ReduceOp = .{
                .op = info.op,
                .in = .{@intFromPtr(info.in[0])},
                .args = info.args,
                .out = @intFromPtr(out),
            } },
            .TypeOp => |info| .{ .TypeOp = .{
                .op = info.op,
                .in = .{@intFromPtr(info.in[0])},
                .out = @intFromPtr(out),
            } },
            .InitOp => |info| .{ .InitOp = .{
                .op = info.op,
                .args = info.args,
                .out = @intFromPtr(out),
            } },
        };
    }

    pub const Json = union(ops.OpTypes) {
        UnaryOp: ops.UnaryOp.Json,
        BinaryOp: ops.BinaryOp.Json,
        ReduceOp: ops.ReduceOp.Json,
        TypeOp: ops.TypeOp.Json,
        InitOp: ops.InitOp.Json,
        TernaryOp: ops.TernaryOp.Json,
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

    pub fn foldIntoGroup(self: OpGroupTracker, group: ?*const OpGroup) OpGroupTracker {
        if (group == null) {
            return self;
        }

        return .{
            .curr = group,
            .next = group,
            .next_id = if (group) |g| @max(self.next_id + 1, g.id + 1) else self.next_id + 1,
        };
    }

    pub fn keepGroup(self: OpGroupTracker) OpGroupTracker {
        if (self.next == null and self.curr == null) {
            return self;
        }
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
