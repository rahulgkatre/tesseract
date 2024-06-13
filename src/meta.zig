const ops = @import("ops.zig");
const std = @import("std");
const utils = @import("utils.zig");
const AnyTensor = @import("anytensor.zig").AnyTensor;
const tensor = @import("tensor.zig");
const graph = @import("graph.zig");
const autograd = @import("autograd.zig");

/// Metadata for tensors, shared between the shape-typed Tensor and AnyTensor
pub const Metadata = struct {
    instr: ops.Instruction,

    /// *const fn (comptime anytype) get_grad_fn_return_type()
    grad_fn: *const anyopaque = autograd.NoBackwardReturnType,
    /// *const fn (comptime anytype) type
    get_grad_fn_return_type: *const anyopaque = autograd.noBackwardFn,
    constant: bool,
    label: ?[]const u8,
};
