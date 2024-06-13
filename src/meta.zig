const ops = @import("ops.zig");
const std = @import("std");
const utils = @import("utils.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const tensor = @import("tensor.zig");
const graph = @import("graph.zig");

/// Metadata for tensors, shared between the shape-typed Tensor and AnyTensor
pub const Metadata = struct {
    instr: ops.Instruction,
    backward: *const anyopaque,
    // backward: ?*const fn (*const AnyTensor, *graph.Graph) anyerror!void = null,
    // const getBackwardReturnType: *const fn () type = comptime @ptrCast(grad_out.meta.backwardReturnType);
    // const backwardImpl: *const fn (comptime anytype) getBackwardReturnType() = comptime @ptrCast(grad_out.meta.backward);
    // try backwardImpl(grad_out, graph);
    constant: bool,
    label: ?[]const u8,
};
