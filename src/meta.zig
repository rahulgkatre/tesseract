const ops = @import("ops.zig");
const std = @import("std");
const utils = @import("utils.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const tensor = @import("tensor.zig");
const graph = @import("graph.zig");

/// Metadata for tensors, shared between the shape-typed Tensor and AnyTensor
pub const Metadata = struct {
    const Forward = *const fn (comptime anytype, *graph.Graph) anyerror!void;
    instr: ops.Instruction,
    forward: *const anyopaque,
    backward: *const anyopaque,
    // forward: *const fn (*const AnyTensor, *graph.Graph) anyerror!void,
    // backward: ?*const fn (*const AnyTensor, *graph.Graph) anyerror!void = null,
    constant: bool,
    label: ?[]const u8,
};
