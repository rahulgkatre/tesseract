const ops = @import("ops.zig");
const std = @import("std");
const utils = @import("utils.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const tensor = @import("tensor.zig");

/// Metadata for tensors, shared between the shape-typed Tensor and AnyTensor
pub const Metadata = struct {
    op_tracker: OpTracker,
    op_group_tracker: OpGroupTracker,
    constant: bool,
    label: ?[]const u8,
};

fn ArgsType(comptime tag: ops.OpTypes, comptime op: @field(ops, utils.rawTagName(tag))) type {
    const Args = @field(@TypeOf(op), "Args");
    switch (@typeInfo(Args)) {
        .Void, .Struct => return Args,
        .Union => {
            const field = @field(Args, utils.rawTagName(op));
            return switch (@typeInfo(@TypeOf(field))) {
                .Type => field,
                else => void,
            };
        },
        else => unreachable,
    }
}

pub const OpTracker = union(ops.OpTypes) {
    UnaryOp: ops.UnaryOp.Info,
    BinaryOp: ops.BinaryOp.Info,
    ReduceOp: ops.ReduceOp.Info,
    TypeOp: ops.TypeOp.Info,
    InitOp: ops.InitOp.Info,
    TernaryOp: ops.TernaryOp.Info,

    pub fn init(
        comptime tag: ops.OpTypes,
        comptime op: @field(ops, utils.rawTagName(tag)),
        in: switch (tag) {
            .TernaryOp => [3]*const AnyTensor,
            .BinaryOp => [2]*const AnyTensor,
            .UnaryOp, .TypeOp, .ReduceOp => [1]*const AnyTensor,
            .InitOp => [0]*const AnyTensor,
        },
        args: ArgsType(tag, op),
    ) OpTracker {
        return @unionInit(
            OpTracker,
            @tagName(tag),
            .{
                .op = op,
                .in = in,
                .args = switch (@typeInfo(@field(@TypeOf(op), "Args"))) {
                    .Struct => args,
                    .Union => @unionInit(@field(@TypeOf(op), "Args"), utils.rawTagName(op), args),
                    .Void => {},
                    else => unreachable,
                },
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

    pub fn joinGroup(self: OpGroupTracker, group: ?*const OpGroup) OpGroupTracker {
        return .{
            .curr = group,
            .next = group,
            .next_id = if (group) |g| @max(self.next_id + 1, g.id + 1) else self.next_id + 1,
        };
    }

    pub fn nextGroup(self: OpGroupTracker) OpGroupTracker {
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

/// This is used to determine which higher level function the tensor is part of
/// which is useful for finding a better gradient implementation if one exists
/// than the one that would be found through simple backtracking of the graph.
/// By default the op_group is null, so setting the op_group isn't necessary
/// for simple functions with trivial gradients (e.g suogtraction)
pub fn startGroup(comptime input: anytype, name: []const u8) tensor.TensorTypeOf(input) {
    // OpGroup will not modify the computation graph so pass the other fields unmodified
    const t = tensor.asTensor(input);
    return .{
        .meta = &.{
            .op_tracker = t.meta.op_tracker,
            .op_group_tracker = t.meta.op_group_tracker.startGroup(name ++ if (t.meta.label) |label| "_" ++ label else ""),
            .constant = t.meta.constant,
            .label = t.meta.label,
        },
        .strides = t.strides,
        .offset = t.offset,
    };
}

pub fn startGroupFromInputs(comptime name: []const u8, comptime tensors: anytype) tensor.TensorTuple(tensors) {
    var grouped: tensor.TensorTuple(tensors) = undefined;
    for (tensors, 0..) |in, i| {
        grouped[i] = startGroup(in, name);
    }
    return grouped;
}

pub fn joinGroup(comptime input: anytype, other: anytype) tensor.TensorTypeOf(input) {
    const a = tensor.asTensor(input);
    const b = tensor.asTensor(other);
    return a.updateMetadata(&.{
        .op_tracker = a.meta.op_tracker,
        .op_group_tracker = a.meta.op_group_tracker.joinGroup(b.meta.op_group_tracker.next),
        .constant = a.meta.constant,
        .label = a.meta.label,
    });
}

/// End the current op_group by setting it to the outer op_group.
/// Compile error if the current op_group is null.
pub fn endGroup(comptime input: anytype) tensor.TensorTypeOf(input) {
    const t = tensor.asTensor(input);
    return t.updateMetadata(&.{
        .op_tracker = t.meta.op_tracker,
        .op_group_tracker = t.meta.op_group_tracker.endGroup(),
        .constant = t.meta.constant,
        .label = t.meta.label,
    });
}

pub fn endGroupFromOutputs(comptime tensors: anytype) tensor.TensorTuple(tensors) {
    var ungrouped: tensor.TensorTuple(tensors) = undefined;
    for (tensors, 0..) |out, i| {
        const t = tensor.asTensor(out);
        ungrouped[i] = t.updateMetadata(&.{
            .op_tracker = t.meta.op_tracker,
            .op_group_tracker = t.meta.op_group_tracker.endGroup(),
            .constant = t.meta.constant,
            .label = t.meta.label,
        });
    }
    return ungrouped;
}